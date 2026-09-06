"""Medicion de latencia, throughput y divergencia entre configuraciones.

Reglas que hacen que las cifras signifiquen algo (CLAUDE.md §9.8):

- **Frames precargados en RAM.** Si se decodifica dentro del bucle se acaba midiendo el
  decodificador, que en este video cuesta mas que la inferencia.
- **Calentamiento descartado.** Las primeras iteraciones incluyen compilacion de kernels
  y subida de pesos, y contaminan la mediana.
- **Percentiles, no medias.** La media esconde la varianza termica.
- **Latencia de inferencia pura separada de la de extremo a extremo.** Sin separarlas se
  atribuye a TensorRT un techo que impone la CPU.
- **Sin etiquetas no hay mAP.** Lo que se reporta es *divergencia* respecto a la
  referencia FP32, nunca precision absoluta.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from sira.infer.postprocess import CONF_DEFECTO, IOU_DEFECTO, Deteccion, postprocesar
from sira.infer.preprocess import preparar_tensor

logger = logging.getLogger(__name__)

CALENTAMIENTO = 50


@dataclass
class Medicion:
    configuracion: str
    detalle: str
    frames: int
    entrada: list[int]
    latencia_inferencia_ms: dict[str, float]
    latencia_e2e_ms: dict[str, float]
    fps_sostenido: float
    vram_pico_mib: float
    detecciones_totales: int
    detecciones_por_frame: float
    tiempos_etapa_ms: dict[str, dict[str, float]] = field(default_factory=dict)
    divergencia: dict[str, Any] | None = None

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)


def _percentiles(muestras: list[float]) -> dict[str, float]:
    arr = np.asarray(muestras, dtype=np.float64)
    return {
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
        "min": round(float(arr.min()), 3),
        "max": round(float(arr.max()), 3),
    }


def cargar_frames(ruta_video: Path, n: int, salto: int = 1) -> list[np.ndarray]:
    """Precarga `n` frames en RAM. Ver la nota del modulo sobre por que."""
    captura = cv2.VideoCapture(str(ruta_video))
    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir {ruta_video}")
    frames: list[np.ndarray] = []
    indice = 0
    while len(frames) < n:
        ok, frame = captura.read()
        if not ok:
            break
        if indice % salto == 0:
            frames.append(frame)
        indice += 1
    captura.release()
    if not frames:
        raise RuntimeError(f"No se pudo leer ningun frame de {ruta_video}")
    logger.info("Precargados %d frames de %s", len(frames), ruta_video.name)
    return frames


def _vram_usada_mib() -> float:
    from cuda.bindings import runtime as cudart

    err, libre, total = cudart.cudaMemGetInfo()
    if int(err) != 0:
        return 0.0
    return (total - libre) / 2**20


def medir_tensorrt(
    ruta_engine: Path,
    frames: list[np.ndarray],
    conf: float = CONF_DEFECTO,
    iou: float = IOU_DEFECTO,
) -> tuple[Medicion, list[list[Deteccion]]]:
    """Mide un engine sobre los frames dados y devuelve tambien sus detecciones."""
    from sira.infer import RuntimeTensorRT

    with RuntimeTensorRT(ruta_engine) as runtime:
        # Calentamiento: kernels compilados y pesos ya en device.
        tensor, params = preparar_tensor(frames[0], runtime.hw, runtime.dtype_entrada)
        for _ in range(CALENTAMIENTO):
            runtime.inferir(tensor)

        vram_base = _vram_usada_mib()
        lat_inf: list[float] = []
        lat_e2e: list[float] = []
        t_pre: list[float] = []
        t_post: list[float] = []
        todas: list[list[Deteccion]] = []
        vram_pico = vram_base

        inicio = time.perf_counter()
        for frame in frames:
            t_a = time.perf_counter()
            tensor, params = preparar_tensor(frame, runtime.hw, runtime.dtype_entrada)
            t_b = time.perf_counter()
            salida = runtime.inferir(tensor)
            t_c = time.perf_counter()
            dets = postprocesar(salida, params, names=runtime.names, conf=conf, iou=iou)
            t_d = time.perf_counter()

            t_pre.append((t_b - t_a) * 1000)
            lat_inf.append((t_c - t_b) * 1000)
            t_post.append((t_d - t_c) * 1000)
            lat_e2e.append((t_d - t_a) * 1000)
            todas.append(dets)
            vram_pico = max(vram_pico, _vram_usada_mib())
        total = time.perf_counter() - inicio

        medicion = Medicion(
            configuracion=ruta_engine.stem,
            detalle=f"TensorRT {ruta_engine.name}",
            frames=len(frames),
            entrada=[int(v) for v in runtime.forma_entrada],
            latencia_inferencia_ms=_percentiles(lat_inf),
            latencia_e2e_ms=_percentiles(lat_e2e),
            fps_sostenido=round(len(frames) / total, 2),
            vram_pico_mib=round(vram_pico, 1),
            detecciones_totales=sum(len(d) for d in todas),
            detecciones_por_frame=round(sum(len(d) for d in todas) / len(todas), 3),
            tiempos_etapa_ms={
                "preproceso": _percentiles(t_pre),
                "inferencia": _percentiles(lat_inf),
                "postproceso": _percentiles(t_post),
            },
        )
    return medicion, todas


def medir_pytorch(
    ruta_pt: Path,
    frames: list[np.ndarray],
    hw: tuple[int, int],
    conf: float = CONF_DEFECTO,
    iou: float = IOU_DEFECTO,
) -> tuple[Medicion, list[list[Deteccion]]]:
    """Baseline: el modelo tal como salio de entrenamiento, sobre la misma GPU.

    Se le pasa la **misma forma de entrada** que al engine y `rect=False`, o la
    comparacion mediria dos preprocesos distintos en vez de dos runtimes.
    """
    import torch
    from ultralytics import YOLO

    modelo = YOLO(str(ruta_pt))
    kwargs = dict(
        conf=conf, iou=iou, imgsz=list(hw), rect=False, verbose=False, device=0
    )

    for _ in range(min(CALENTAMIENTO, 20)):
        modelo.predict(frames[0], **kwargs)
    torch.cuda.synchronize()

    lat: list[float] = []
    todas: list[list[Deteccion]] = []
    vram_pico = _vram_usada_mib()

    inicio = time.perf_counter()
    for frame in frames:
        t_a = time.perf_counter()
        resultado = modelo.predict(frame, **kwargs)[0]
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t_a) * 1000)

        todas.append([
            Deteccion(
                bbox=tuple(float(v) for v in caja.xyxy[0].tolist()),
                score=float(caja.conf[0]),
                class_id=int(caja.cls[0]),
                class_name=modelo.names.get(int(caja.cls[0]), str(int(caja.cls[0]))),
            )
            for caja in resultado.boxes
        ])
        vram_pico = max(vram_pico, _vram_usada_mib())
    total = time.perf_counter() - inicio

    medicion = Medicion(
        configuracion="pytorch",
        detalle=f"PyTorch/Ultralytics {ruta_pt.name}",
        frames=len(frames),
        entrada=[1, 3, hw[0], hw[1]],
        # Ultralytics no expone la inferencia pura por separado: su predict incluye
        # pre y postproceso. Se reporta el mismo valor en ambos campos y se dice.
        latencia_inferencia_ms=_percentiles(lat),
        latencia_e2e_ms=_percentiles(lat),
        fps_sostenido=round(len(frames) / total, 2),
        vram_pico_mib=round(vram_pico, 1),
        detecciones_totales=sum(len(d) for d in todas),
        detecciones_por_frame=round(sum(len(d) for d in todas) / len(todas), 3),
    )
    return medicion, todas


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def comparar_detecciones(
    referencia: list[list[Deteccion]], candidata: list[list[Deteccion]]
) -> dict[str, Any]:
    """Divergencia frente a la referencia FP32.

    **No es mAP.** Sin etiquetas no hay precision absoluta: esto mide cuanto se aparta
    una configuracion de la que se toma como referencia, y nada mas.
    """
    ious: list[float] = []
    deltas: list[float] = []
    sin_pareja_ref = 0
    sin_pareja_cand = 0
    cambios_clase = 0

    for ref_frame, cand_frame in zip(referencia, candidata):
        libres = list(range(len(cand_frame)))
        for det_ref in ref_frame:
            mejor, mejor_iou = None, 0.0
            for k in libres:
                solape = _iou(det_ref.bbox, cand_frame[k].bbox)
                if solape > mejor_iou:
                    mejor, mejor_iou = k, solape
            if mejor is not None and mejor_iou >= 0.5:
                libres.remove(mejor)
                ious.append(mejor_iou)
                deltas.append(abs(det_ref.score - cand_frame[mejor].score))
                if det_ref.class_id != cand_frame[mejor].class_id:
                    cambios_clase += 1
            else:
                sin_pareja_ref += 1
        sin_pareja_cand += len(libres)

    if not ious:
        return {
            "emparejadas": 0,
            "solo_en_referencia": sin_pareja_ref,
            "solo_en_candidata": sin_pareja_cand,
            "nota": "sin parejas: las configuraciones no son comparables",
        }

    arr_iou = np.asarray(ious)
    arr_delta = np.asarray(deltas)
    return {
        "emparejadas": len(ious),
        "solo_en_referencia": sin_pareja_ref,
        "solo_en_candidata": sin_pareja_cand,
        "cambios_de_clase": cambios_clase,
        "iou_medio": round(float(arr_iou.mean()), 5),
        "iou_p05": round(float(np.percentile(arr_iou, 5)), 5),
        "iou_minimo": round(float(arr_iou.min()), 5),
        "delta_score_medio": round(float(arr_delta.mean()), 5),
        "delta_score_maximo": round(float(arr_delta.max()), 5),
        "nota": "divergencia frente a la referencia FP32; NO es mAP",
    }
