"""Motor de inferencia sobre video.

Un hilo lee frames, infiere y deja el ultimo resultado en una ranura de tamano uno.
Los clientes HTTP leen esa ranura.

**Se descartan frames, no se encolan.** Si la inferencia va mas lenta que el video y se
encolara, la latencia crecería sin techo hasta que el stream fuera minutos por detras de
la realidad. Con una ranura de uno, un cliente lento solo pierde frames intermedios y
siempre ve lo mas reciente (CLAUDE.md §9.7).

Los tiempos se miden **por etapa**: sin separarlos no se sabe si el techo lo pone
TensorRT o el decodificador, y en video suele ponerlo el decodificador.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sira.infer import RuntimeTensorRT, postprocesar
from sira.infer.postprocess import CONF_DEFECTO, IOU_DEFECTO, Deteccion
from sira.infer.preprocess import preparar_tensor

logger = logging.getLogger(__name__)

VENTANA_METRICAS = 120
COLOR_CAJA = (0, 220, 0)


@dataclass
class Tiempos:
    """Tiempos de una etapa, en milisegundos, sobre una ventana deslizante."""

    decode: deque = field(default_factory=lambda: deque(maxlen=VENTANA_METRICAS))
    preproceso: deque = field(default_factory=lambda: deque(maxlen=VENTANA_METRICAS))
    inferencia: deque = field(default_factory=lambda: deque(maxlen=VENTANA_METRICAS))
    postproceso: deque = field(default_factory=lambda: deque(maxlen=VENTANA_METRICAS))
    anotado: deque = field(default_factory=lambda: deque(maxlen=VENTANA_METRICAS))
    encode: deque = field(default_factory=lambda: deque(maxlen=VENTANA_METRICAS))

    def resumen(self) -> dict[str, dict[str, float]]:
        salida: dict[str, dict[str, float]] = {}
        for nombre, valores in vars(self).items():
            if not valores:
                continue
            arr = np.fromiter(valores, dtype=np.float64)
            salida[nombre] = {
                "p50": round(float(np.percentile(arr, 50)), 3),
                "p95": round(float(np.percentile(arr, 95)), 3),
            }
        return salida


@dataclass
class EstadoMotor:
    frames_leidos: int = 0
    frames_procesados: int = 0
    frames_descartados: int = 0
    fps: float = 0.0
    detecciones: list[Deteccion] = field(default_factory=list)
    corriendo: bool = False


class MotorVideo:
    """Bucle de captura e inferencia en un hilo propio."""

    def __init__(
        self,
        ruta_video: Path,
        ruta_engine: Path,
        conf: float = CONF_DEFECTO,
        iou: float = IOU_DEFECTO,
        bucle: bool = True,
        calidad_jpeg: int = 80,
        fps_objetivo: float | None = None,
    ) -> None:
        self.ruta_video = ruta_video
        self.ruta_engine = ruta_engine
        self.conf = conf
        self.iou = iou
        self.bucle = bucle
        self.calidad_jpeg = int(calidad_jpeg)
        self.fps_objetivo = fps_objetivo

        self.tiempos = Tiempos()
        self.estado = EstadoMotor()

        self._runtime: RuntimeTensorRT | None = None
        self._hilo: threading.Thread | None = None
        self._parar = threading.Event()
        # Ranura de tamano uno: el frame nuevo pisa al anterior.
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._frame_nuevo = threading.Condition(self._lock)
        self._contador_publicado = 0
        self.fps_video = 0.0
        self.resolucion = (0, 0)

    # -- ciclo de vida ------------------------------------------------------

    def arrancar(self) -> None:
        if self._hilo is not None:
            return
        self._runtime = RuntimeTensorRT(self.ruta_engine)
        self._parar.clear()
        self._hilo = threading.Thread(target=self._bucle, name="motor-video", daemon=True)
        self._hilo.start()
        logger.info("Motor arrancado sobre %s", self.ruta_video.name)

    def parar(self) -> None:
        self._parar.set()
        if self._hilo is not None:
            self._hilo.join(timeout=5)
            self._hilo = None
        if self._runtime is not None:
            self._runtime.cerrar()
            self._runtime = None
        self.estado.corriendo = False
        logger.info("Motor detenido")

    # -- lectura por parte de los clientes ----------------------------------

    def ultimo_jpeg(self, contador_visto: int, timeout: float = 5.0) -> tuple[bytes | None, int]:
        """Bloquea hasta que haya un frame mas nuevo que `contador_visto`.

        Devolver el contador permite al cliente no reenviar el mismo frame: sin el, un
        cliente rapido quemaria CPU reenviando imagenes identicas.
        """
        with self._frame_nuevo:
            if self._contador_publicado <= contador_visto:
                self._frame_nuevo.wait(timeout=timeout)
            return self._jpeg, self._contador_publicado

    def instantanea(self) -> dict[str, Any]:
        return {
            "corriendo": self.estado.corriendo,
            "engine": self.ruta_engine.name,
            "video": self.ruta_video.name,
            "resolucion": list(self.resolucion),
            "fps_video": round(self.fps_video, 2),
            "fps": round(self.estado.fps, 2),
            "frames_leidos": self.estado.frames_leidos,
            "frames_procesados": self.estado.frames_procesados,
            "frames_descartados": self.estado.frames_descartados,
            "detecciones": [d.a_dict() for d in self.estado.detecciones],
            "tiempos_ms": self.tiempos.resumen(),
        }

    # -- bucle interno ------------------------------------------------------

    def _bucle(self) -> None:
        runtime = self._runtime
        assert runtime is not None

        captura = cv2.VideoCapture(str(self.ruta_video))
        if not captura.isOpened():
            logger.error("No se pudo abrir %s", self.ruta_video)
            return

        self.fps_video = float(captura.get(cv2.CAP_PROP_FPS) or 0.0)
        self.resolucion = (
            int(captura.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        periodo = 0.0
        objetivo = self.fps_objetivo or self.fps_video
        if objetivo and objetivo > 0:
            periodo = 1.0 / objetivo

        self.estado.corriendo = True
        siguiente = time.perf_counter()
        marca_fps = time.perf_counter()
        contados = 0

        while not self._parar.is_set():
            t0 = time.perf_counter()
            ok, frame = captura.read()
            if not ok:
                if self.bucle:
                    captura.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            self.tiempos.decode.append((time.perf_counter() - t0) * 1000)
            self.estado.frames_leidos += 1

            # Si vamos por detras del ritmo objetivo, saltar este frame en vez de
            # acumular retraso: es la diferencia entre perder frames y perder el directo.
            ahora = time.perf_counter()
            if periodo and ahora > siguiente + periodo:
                self.estado.frames_descartados += 1
                siguiente = ahora
                continue

            t1 = time.perf_counter()
            tensor, params = preparar_tensor(frame, runtime.hw, runtime.dtype_entrada)
            self.tiempos.preproceso.append((time.perf_counter() - t1) * 1000)

            t2 = time.perf_counter()
            salida = runtime.inferir(tensor)
            self.tiempos.inferencia.append((time.perf_counter() - t2) * 1000)

            t3 = time.perf_counter()
            detecciones = postprocesar(
                salida, params, names=runtime.names, conf=self.conf, iou=self.iou
            )
            self.tiempos.postproceso.append((time.perf_counter() - t3) * 1000)

            t4 = time.perf_counter()
            anotado = self._anotar(frame, detecciones)
            self.tiempos.anotado.append((time.perf_counter() - t4) * 1000)

            t5 = time.perf_counter()
            ok, buffer = cv2.imencode(
                ".jpg", anotado, [int(cv2.IMWRITE_JPEG_QUALITY), self.calidad_jpeg]
            )
            self.tiempos.encode.append((time.perf_counter() - t5) * 1000)

            if ok:
                with self._frame_nuevo:
                    self._jpeg = buffer.tobytes()
                    self._contador_publicado += 1
                    self._frame_nuevo.notify_all()

            self.estado.detecciones = detecciones
            self.estado.frames_procesados += 1
            contados += 1

            transcurrido = time.perf_counter() - marca_fps
            if transcurrido >= 1.0:
                self.estado.fps = contados / transcurrido
                contados = 0
                marca_fps = time.perf_counter()

            if periodo:
                siguiente += periodo
                espera = siguiente - time.perf_counter()
                if espera > 0:
                    time.sleep(espera)

        captura.release()
        self.estado.corriendo = False

    def _anotar(self, frame: np.ndarray, detecciones: list[Deteccion]) -> np.ndarray:
        anotado = frame.copy()
        for d in detecciones:
            x1, y1, x2, y2 = (int(v) for v in d.bbox)
            cv2.rectangle(anotado, (x1, y1), (x2, y2), COLOR_CAJA, 2)
            etiqueta = f"{d.class_name} {d.score:.2f}"
            cv2.putText(
                anotado, etiqueta, (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CAJA, 2, cv2.LINE_AA,
            )
        return anotado
