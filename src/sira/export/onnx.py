"""Export de `best.pt` a ONNX.

Unico punto del proyecto donde interviene Ultralytics, y es inevitable: un `.pt` es un
pickle que serializa el objeto `DetectionModel` con referencias a clases Python, de modo
que deserializarlo exige el interprete y el paquete. Ademas `torch.onnx.export` no tiene
equivalente fuera de Python.

A partir del ONNX, el resto del pipeline es propio: construccion del engine, preproceso,
postproceso y NMS (CLAUDE.md §1).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExportError(RuntimeError):
    """Fallo exportando el checkpoint."""


@dataclass(frozen=True)
class InfoCheckpoint:
    """Lo que el checkpoint dice de si mismo.

    `nc` y `names` se leen de aqui y **nunca se hardcodean** (CLAUDE.md §9.2): el
    catalogo del repositorio de entrenamiento no permite deducirlos porque filtra clases
    con `keep_classes`.
    """

    nc: int
    names: dict[int, str]
    task: str
    imgsz_entrenamiento: int | None
    parametros: int
    bytes_checkpoint: int


@dataclass(frozen=True)
class ResultadoExport:
    """Metadatos del ONNX generado. Se persisten junto al fichero."""

    onnx: str
    onnx_sha256: str
    onnx_bytes: int
    imgsz: int
    opset: int
    nms_incrustado: bool
    dinamico: bool
    half: bool
    nc: int
    names: dict[int, str]
    entrada: dict[str, Any]
    salida: dict[str, Any]
    origen_pt: str
    origen_pt_sha256: str
    exportado_en: str

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(ruta: Path) -> str:
    sha = hashlib.sha256()
    with ruta.open("rb") as fh:
        for trozo in iter(lambda: fh.read(1024 * 1024), b""):
            sha.update(trozo)
    return sha.hexdigest()


def inspeccionar_checkpoint(ruta_pt: Path) -> InfoCheckpoint:
    """Abre el checkpoint y devuelve lo que declara sobre si mismo."""
    from ultralytics import YOLO  # import perezoso: arrastra torch

    if not ruta_pt.is_file():
        raise ExportError(f"No existe el checkpoint {ruta_pt}")

    modelo = YOLO(str(ruta_pt))
    names = {int(k): str(v) for k, v in modelo.names.items()}
    parametros = sum(p.numel() for p in modelo.model.parameters())

    args = getattr(modelo.model, "args", None) or {}
    if not isinstance(args, dict):
        args = vars(args) if hasattr(args, "__dict__") else {}
    imgsz = args.get("imgsz")
    if isinstance(imgsz, (list, tuple)):
        imgsz = imgsz[0]

    return InfoCheckpoint(
        nc=len(names),
        names=names,
        task=getattr(modelo, "task", "") or "",
        imgsz_entrenamiento=int(imgsz) if isinstance(imgsz, (int, float)) else None,
        parametros=int(parametros),
        bytes_checkpoint=ruta_pt.stat().st_size,
    )


def _describir_onnx(ruta_onnx: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Forma y tipo de la entrada y la salida del ONNX, leidas del grafo."""
    import onnx

    modelo = onnx.load(str(ruta_onnx))

    def describir(valor) -> dict[str, Any]:
        tipo = valor.type.tensor_type
        forma = [
            d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
            for d in tipo.shape.dim
        ]
        return {
            "nombre": valor.name,
            "forma": forma,
            "dtype": onnx.TensorProto.DataType.Name(tipo.elem_type),
        }

    iniciales = {t.name for t in modelo.graph.initializer}
    entradas = [v for v in modelo.graph.input if v.name not in iniciales]
    return describir(entradas[0]), describir(modelo.graph.output[0])


def exportar_onnx(
    ruta_pt: Path,
    destino: Path,
    imgsz: int = 640,
    opset: int | None = None,
    half: bool = False,
    simplify: bool = True,
) -> ResultadoExport:
    """Exporta el checkpoint a ONNX y devuelve los metadatos del resultado.

    `nms=False` siempre: el NMS se hace en el postproceso propio. Si fuera incrustado en
    el grafo, el postproceso dejaria de ser nuestro y se perderia el control sobre
    umbrales y sobre la comparacion con el baseline (CLAUDE.md §9.2).

    `dynamic=False`: forma de entrada fija. Simplifica el engine y el runtime, y el
    servidor siempre alimenta frames del mismo tamano tras el letterbox.
    """
    from ultralytics import YOLO

    info = inspeccionar_checkpoint(ruta_pt)
    logger.info(
        "Checkpoint: task=%s, nc=%d, %.2f M parametros, imgsz de entrenamiento=%s",
        info.task, info.nc, info.parametros / 1e6, info.imgsz_entrenamiento,
    )

    if info.imgsz_entrenamiento and info.imgsz_entrenamiento != imgsz:
        logger.warning(
            "Exportando a imgsz=%d pero el modelo se entreno con %d. "
            "El preproceso debe usar el mismo valor que este export.",
            imgsz, info.imgsz_entrenamiento,
        )

    modelo = YOLO(str(ruta_pt))
    kwargs: dict[str, Any] = {
        "format": "onnx",
        "imgsz": imgsz,
        "nms": False,
        "dynamic": False,
        "simplify": simplify,
        "half": half,
        "device": "cpu",
        "verbose": False,
    }
    if opset is not None:
        kwargs["opset"] = opset

    logger.info("Exportando a ONNX (%s)", ", ".join(f"{k}={v}" for k, v in kwargs.items()))
    generado = Path(modelo.export(**kwargs))

    destino.parent.mkdir(parents=True, exist_ok=True)
    if generado.resolve() != destino.resolve():
        generado.replace(destino)

    entrada, salida = _describir_onnx(destino)
    logger.info("ONNX entrada=%s salida=%s", entrada, salida)

    # La salida de YOLO sin NMS es [1, 4+nc, N]. Si el eje no cuadra con las clases del
    # checkpoint, el postproceso interpretaria mal el tensor y las cajas saldrian con la
    # clase cambiada, sin ningun error visible.
    forma = salida["forma"]
    if len(forma) == 3 and isinstance(forma[1], int) and forma[1] != 4 + info.nc:
        logger.warning(
            "La salida tiene %s canales pero el checkpoint declara nc=%d (se esperaban %d). "
            "Revisar antes de escribir el postproceso.",
            forma[1], info.nc, 4 + info.nc,
        )

    opset_real = opset
    if opset_real is None:
        import onnx
        opset_real = max(o.version for o in onnx.load(str(destino)).opset_import)

    return ResultadoExport(
        onnx=destino.name,
        onnx_sha256=_sha256(destino),
        onnx_bytes=destino.stat().st_size,
        imgsz=imgsz,
        opset=int(opset_real),
        nms_incrustado=False,
        dinamico=False,
        half=half,
        nc=info.nc,
        names=info.names,
        entrada=entrada,
        salida=salida,
        origen_pt=ruta_pt.name,
        origen_pt_sha256=_sha256(ruta_pt),
        exportado_en=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
