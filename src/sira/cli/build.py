"""Construye engines TensorRT a partir de los ONNX de `models/`.

En TensorRT 11 la precision la fija el ONNX, no el builder: cada precision necesita su
propio ONNX de partida (CLAUDE.md §4.3).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sira.common.config import Config, ConfigIncompletaError
from sira.common.logging import configurar
from sira.engine import EngineError, construir_engine
from sira.engine.builder import WORKSPACE_MB_DEFECTO

logger = logging.getLogger("sira.build")

# precision -> nombre del ONNX de partida
ONNX_POR_PRECISION = {
    "fp32": "model.onnx",
    "fp16": "model_fp16.onnx",
    "int8": "model_int8.onnx",
}


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--precision",
        default="fp32",
        choices=sorted(ONNX_POR_PRECISION),
        help="Precision del engine (determina de que ONNX se parte)",
    )
    p.add_argument("--onnx", type=Path, help="ONNX de entrada (sobreescribe el de la precision)")
    p.add_argument("--destino", type=Path, help="Ruta del .engine de salida")
    p.add_argument("--workspace-mb", type=int, default=WORKSPACE_MB_DEFECTO)
    p.add_argument(
        "--sin-timing-cache",
        action="store_true",
        help="No reutilizar el cache de tactics entre construcciones",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    configurar()
    args = construir_parser().parse_args(argv)

    try:
        config = Config.desde_entorno()
    except ConfigIncompletaError as exc:
        logger.error("%s", exc)
        return 2

    ruta_onnx = args.onnx or (config.dir_modelos / ONNX_POR_PRECISION[args.precision])
    destino = args.destino or (config.dir_modelos / f"model_{args.precision}.engine")
    timing = None if args.sin_timing_cache else (config.dir_modelos / "timing.cache")

    if not ruta_onnx.is_file():
        logger.error(
            "No existe %s. Genera primero el ONNX de %s (ver `export`).",
            ruta_onnx, args.precision,
        )
        return 1

    try:
        meta = construir_engine(
            ruta_onnx=ruta_onnx,
            destino=destino,
            precision=args.precision,
            workspace_mb=args.workspace_mb,
            ruta_timing_cache=timing,
        )
    except EngineError as exc:
        logger.error("%s", exc)
        return 1

    ruta_meta = destino.with_suffix(destino.suffix + ".json")
    ruta_meta.write_text(
        json.dumps(meta.a_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Metadatos escritos en %s", ruta_meta)

    resumen = meta.a_dict()
    resumen["names"] = f"<{len(meta.names)} clases>"
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
