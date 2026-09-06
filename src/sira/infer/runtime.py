"""Ejecucion de un engine TensorRT.

Reserva los buferes de device una sola vez y los reutiliza en cada inferencia: en un
servidor de video se ejecuta decenas de veces por segundo, y reservar y liberar en cada
frame domina el tiempo y fragmenta la memoria.

El `IExecutionContext` **no es seguro para uso concurrente**. Esta clase no se protege
sola: quien la use desde varios hilos debe serializar el acceso o crear una instancia por
hilo (CLAUDE.md §4.4).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from sira.engine.builder import EngineError, cargar_engine

logger = logging.getLogger(__name__)


def _comprobar(err: Any, que: str) -> None:
    if int(err) != 0:
        raise EngineError(f"CUDA fallo en {que}: {err}")


class RuntimeTensorRT:
    """Envoltorio de un engine listo para inferir sobre tensores de numpy."""

    def __init__(self, ruta_engine: Path, comprobar_compatibilidad: bool = True) -> None:
        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        self._trt = trt
        self._cudart = cudart

        self.ruta_engine = ruta_engine
        self.engine = cargar_engine(ruta_engine, comprobar=comprobar_compatibilidad)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise EngineError(f"No se pudo crear el contexto de {ruta_engine}")

        self.meta = self._leer_meta(ruta_engine)
        self.nc: int = int(self.meta.get("nc", 0))
        self.names: dict[int, str] = {
            int(k): str(v) for k, v in (self.meta.get("names") or {}).items()
        }

        self._nombre_entrada = ""
        self._nombre_salida = ""
        for i in range(self.engine.num_io_tensors):
            nombre = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(nombre) == trt.TensorIOMode.INPUT:
                self._nombre_entrada = self._nombre_entrada or nombre
            else:
                self._nombre_salida = self._nombre_salida or nombre

        self.forma_entrada = tuple(self.engine.get_tensor_shape(self._nombre_entrada))
        # El alto y el ancho se leen del engine, no de los metadatos: el engine es la
        # verdad sobre lo que acepta, y con entradas rectangulares un `imgsz` escalar
        # de los metadatos seria ambiguo.
        self.hw: tuple[int, int] = (int(self.forma_entrada[2]), int(self.forma_entrada[3]))
        self.forma_salida = tuple(self.engine.get_tensor_shape(self._nombre_salida))
        # El dtype sale del engine: uno construido desde un ONNX FP16 espera FLOAT16, y
        # alimentarlo con float32 produce basura silenciosamente.
        self.dtype_entrada = trt.nptype(
            self.engine.get_tensor_dtype(self._nombre_entrada)
        )
        self.dtype_salida = trt.nptype(self.engine.get_tensor_dtype(self._nombre_salida))

        err, self._stream = cudart.cudaStreamCreate()
        _comprobar(err, "cudaStreamCreate")

        self._bytes_entrada = int(np.prod(self.forma_entrada)) * np.dtype(self.dtype_entrada).itemsize
        self._bytes_salida = int(np.prod(self.forma_salida)) * np.dtype(self.dtype_salida).itemsize

        err, self._d_entrada = cudart.cudaMalloc(self._bytes_entrada)
        _comprobar(err, "cudaMalloc entrada")
        err, self._d_salida = cudart.cudaMalloc(self._bytes_salida)
        _comprobar(err, "cudaMalloc salida")

        self.context.set_tensor_address(self._nombre_entrada, int(self._d_entrada))
        self.context.set_tensor_address(self._nombre_salida, int(self._d_salida))

        self._salida_host = np.empty(self.forma_salida, dtype=self.dtype_salida)
        self._cerrado = False

        logger.info(
            "Engine %s cargado: %s %s -> %s %s, nc=%d",
            ruta_engine.name, self.forma_entrada, np.dtype(self.dtype_entrada).name,
            self.forma_salida, np.dtype(self.dtype_salida).name, self.nc,
        )

    @staticmethod
    def _leer_meta(ruta_engine: Path) -> dict[str, Any]:
        ruta = ruta_engine.with_suffix(ruta_engine.suffix + ".json")
        if not ruta.is_file():
            return {}
        return json.loads(ruta.read_text(encoding="utf-8"))

    def inferir(self, tensor: np.ndarray) -> np.ndarray:
        """Ejecuta el engine sobre un tensor ya preprocesado."""
        if self._cerrado:
            raise EngineError("El runtime ya esta cerrado")

        if tensor.dtype != np.dtype(self.dtype_entrada):
            tensor = tensor.astype(self.dtype_entrada)
        tensor = np.ascontiguousarray(tensor)

        cudart = self._cudart
        kind = cudart.cudaMemcpyKind

        err = cudart.cudaMemcpyAsync(
            self._d_entrada, tensor.ctypes.data, self._bytes_entrada,
            kind.cudaMemcpyHostToDevice, self._stream,
        )[0]
        _comprobar(err, "memcpy H2D")

        if not self.context.execute_async_v3(int(self._stream)):
            raise EngineError("execute_async_v3 devolvio False")

        err = cudart.cudaMemcpyAsync(
            self._salida_host.ctypes.data, self._d_salida, self._bytes_salida,
            kind.cudaMemcpyDeviceToHost, self._stream,
        )[0]
        _comprobar(err, "memcpy D2H")

        err = cudart.cudaStreamSynchronize(self._stream)[0]
        _comprobar(err, "stream sync")

        return self._salida_host

    def cerrar(self) -> None:
        if self._cerrado:
            return
        self._cerrado = True
        for puntero in (getattr(self, "_d_entrada", None), getattr(self, "_d_salida", None)):
            if puntero is not None:
                self._cudart.cudaFree(puntero)
        if getattr(self, "_stream", None) is not None:
            self._cudart.cudaStreamDestroy(self._stream)

    def __enter__(self) -> "RuntimeTensorRT":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.cerrar()
