"""Construccion y carga de engines TensorRT.

**TensorRT 11 no tiene flags de precision.** No existen `BuilderFlag.FP16` ni
`BuilderFlag.INT8`: toda red es *strongly typed* y la precision la determinan los tipos
del grafo ONNX. Tampoco existe `IInt8Calibrator`. Por eso no se construyen varias
precisiones desde un mismo ONNX: se parte de **un ONNX por precision** (CLAUDE.md §4.3).

Verificado sobre la instalacion real en `docs/fase-0-preparacion-host.md`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WORKSPACE_MB_DEFECTO = 2048


class EngineError(RuntimeError):
    """Fallo construyendo o cargando un engine."""


class EngineIncompatibleError(EngineError):
    """El engine se construyo para otra version de TensorRT u otra GPU.

    Un plan de TensorRT esta ligado a la version de la libreria y a la arquitectura de
    GPU. Cargarlo a ciegas produce fallos oscuros o resultados incorrectos, asi que se
    comprueba antes (CLAUDE.md §9.5).
    """


@dataclass(frozen=True)
class InfoGPU:
    nombre: str
    compute_cap: str
    vram_mib: int


@dataclass(frozen=True)
class MetadatosEngine:
    """Sidecar de cada engine. Se escribe junto al `.engine` y se valida al cargar."""

    engine: str
    precision: str
    trt_version: str
    gpu_nombre: str
    compute_cap: str
    onnx: str
    onnx_sha256: str
    imgsz: int
    nc: int
    names: dict[str, str]
    entrada: dict[str, Any]
    salida: dict[str, Any]
    workspace_mb: int
    build_segundos: float
    engine_bytes: int
    creado_en: str

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)


def info_gpu(indice: int = 0) -> InfoGPU:
    """Nombre, compute capability y VRAM del dispositivo."""
    from cuda.bindings import runtime as cudart

    err, props = cudart.cudaGetDeviceProperties(indice)
    if int(err) != 0:
        raise EngineError(f"No se pudo consultar la GPU {indice}: {err}")
    nombre = props.name.decode() if isinstance(props.name, bytes) else str(props.name)
    return InfoGPU(
        nombre=nombre.strip("\x00").strip(),
        compute_cap=f"{props.major}.{props.minor}",
        vram_mib=props.totalGlobalMem // 2**20,
    )


class _Logger:
    """Adaptador entre el ILogger de TensorRT y el logging del proyecto.

    Los avisos del parser son la principal pista cuando un ONNX no encaja, asi que no se
    descartan: van al log como cualquier otra traza.
    """

    def __new__(cls):
        import tensorrt as trt

        nivel = {
            trt.Logger.INTERNAL_ERROR: logging.CRITICAL,
            trt.Logger.ERROR: logging.ERROR,
            trt.Logger.WARNING: logging.WARNING,
            trt.Logger.INFO: logging.DEBUG,
            trt.Logger.VERBOSE: logging.DEBUG,
        }

        class Adaptador(trt.ILogger):
            def __init__(self) -> None:
                trt.ILogger.__init__(self)

            def log(self, severity, msg) -> None:  # noqa: D102
                logger.log(nivel.get(severity, logging.INFO), "[TRT] %s", msg)

        return Adaptador()


def _sha256(ruta: Path) -> str:
    sha = hashlib.sha256()
    with ruta.open("rb") as fh:
        for trozo in iter(lambda: fh.read(1024 * 1024), b""):
            sha.update(trozo)
    return sha.hexdigest()


def _describir_tensores(engine) -> tuple[dict[str, Any], dict[str, Any]]:
    """Entrada y salida del engine, leidas del propio plan."""
    import tensorrt as trt

    entrada = salida = None
    for i in range(engine.num_io_tensors):
        nombre = engine.get_tensor_name(i)
        desc = {
            "nombre": nombre,
            "forma": list(engine.get_tensor_shape(nombre)),
            "dtype": str(engine.get_tensor_dtype(nombre)).rsplit(".", 1)[-1],
        }
        if engine.get_tensor_mode(nombre) == trt.TensorIOMode.INPUT:
            entrada = entrada or desc
        else:
            salida = salida or desc
    if entrada is None or salida is None:
        raise EngineError("El engine no expone una entrada y una salida reconocibles")
    return entrada, salida


def construir_engine(
    ruta_onnx: Path,
    destino: Path,
    precision: str,
    workspace_mb: int = WORKSPACE_MB_DEFECTO,
    ruta_timing_cache: Path | None = None,
) -> MetadatosEngine:
    """Compila un ONNX a un plan de TensorRT serializado.

    `precision` es solo una **etiqueta**: en TensorRT 11 la precision real la fijan los
    tipos del ONNX de entrada, no el builder. Sirve para nombrar el artefacto y para que
    el benchmark sepa que esta comparando.
    """
    import tensorrt as trt

    if not ruta_onnx.is_file():
        raise EngineError(f"No existe el ONNX {ruta_onnx}")

    meta_onnx: dict[str, Any] = {}
    ruta_meta_onnx = ruta_onnx.with_suffix(ruta_onnx.suffix + ".json")
    if ruta_meta_onnx.is_file():
        meta_onnx = json.loads(ruta_meta_onnx.read_text(encoding="utf-8"))
    else:
        logger.warning(
            "Sin %s: el engine no llevara nc ni names, y el runtime los necesita.",
            ruta_meta_onnx.name,
        )

    trt_logger = _Logger()
    builder = trt.Builder(trt_logger)
    # En TensorRT 11 toda red es strongly typed; no hay flags de precision que pasar.
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, trt_logger)

    logger.info("Parseando %s", ruta_onnx)
    if not parser.parse(ruta_onnx.read_bytes()):
        errores = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise EngineError(
            "El parser de ONNX fallo:\n" + "\n".join(f"  - {e}" for e in errores)
        )

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024
    )

    # El cache de tactics ahorra mucho tiempo al reconstruir variantes del mismo modelo.
    cache = None
    if ruta_timing_cache is not None:
        datos = ruta_timing_cache.read_bytes() if ruta_timing_cache.is_file() else b""
        cache = config.create_timing_cache(datos)
        config.set_timing_cache(cache, ignore_mismatch=False)

    gpu = info_gpu()
    logger.info(
        "Construyendo engine %s para %s (cc %s), workspace %d MB. Puede tardar.",
        precision, gpu.nombre, gpu.compute_cap, workspace_mb,
    )
    inicio = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    transcurrido = time.perf_counter() - inicio

    if plan is None:
        raise EngineError(
            "build_serialized_network devolvio None. Revisa los avisos [TRT] de arriba."
        )

    if cache is not None and ruta_timing_cache is not None:
        ruta_timing_cache.parent.mkdir(parents=True, exist_ok=True)
        ruta_timing_cache.write_bytes(memoryview(cache.serialize()))

    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_bytes(memoryview(plan))
    logger.info(
        "Engine %s escrito en %s (%.1f MB) en %.1f s",
        precision, destino, destino.stat().st_size / 1e6, transcurrido,
    )

    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise EngineError("El engine recien construido no se puede deserializar")
    entrada, salida = _describir_tensores(engine)

    return MetadatosEngine(
        engine=destino.name,
        precision=precision,
        trt_version=trt.__version__,
        gpu_nombre=gpu.nombre,
        compute_cap=gpu.compute_cap,
        onnx=ruta_onnx.name,
        onnx_sha256=meta_onnx.get("onnx_sha256") or _sha256(ruta_onnx),
        imgsz=int(meta_onnx.get("imgsz", 0)),
        nc=int(meta_onnx.get("nc", 0)),
        names={str(k): str(v) for k, v in (meta_onnx.get("names") or {}).items()},
        entrada=entrada,
        salida=salida,
        workspace_mb=workspace_mb,
        build_segundos=round(transcurrido, 2),
        engine_bytes=destino.stat().st_size,
        creado_en=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def cargar_engine(ruta_engine: Path, comprobar: bool = True):
    """Deserializa un engine, validando antes que corresponde a este entorno.

    Un plan construido con otra version de TensorRT o para otra GPU no debe cargarse a
    ciegas: falla de formas poco claras o, peor, no falla.
    """
    import tensorrt as trt

    if not ruta_engine.is_file():
        raise EngineError(f"No existe el engine {ruta_engine}")

    if comprobar:
        ruta_meta = ruta_engine.with_suffix(ruta_engine.suffix + ".json")
        if not ruta_meta.is_file():
            raise EngineIncompatibleError(
                f"Falta {ruta_meta.name}: no se puede verificar para que version de "
                "TensorRT ni para que GPU se construyo este engine."
            )
        meta = json.loads(ruta_meta.read_text(encoding="utf-8"))
        gpu = info_gpu()
        problemas = []
        if meta.get("trt_version") != trt.__version__:
            problemas.append(
                f"TensorRT {meta.get('trt_version')} != {trt.__version__} actual"
            )
        if meta.get("compute_cap") != gpu.compute_cap:
            problemas.append(
                f"compute capability {meta.get('compute_cap')} != {gpu.compute_cap} actual"
            )
        if problemas:
            raise EngineIncompatibleError(
                f"{ruta_engine.name} no corresponde a este entorno: "
                + "; ".join(problemas)
                + ". Reconstruyelo."
            )

    runtime = trt.Runtime(_Logger())
    engine = runtime.deserialize_cuda_engine(ruta_engine.read_bytes())
    if engine is None:
        raise EngineError(f"No se pudo deserializar {ruta_engine}")
    return engine
