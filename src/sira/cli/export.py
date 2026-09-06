"""Exporta `models/best.pt` a ONNX.

Deja en `models/`:
  - `model.onnx`
  - `model.onnx.json` con los metadatos del export (nc, names, formas, hashes)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sira.common.config import Config, ConfigIncompletaError
from sira.common.logging import configurar
from sira.export import ExportError, exportar_onnx, inspeccionar_checkpoint

logger = logging.getLogger("sira.export")


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pt", type=Path, help="Checkpoint de entrada")
    p.add_argument("--destino", type=Path, help="Ruta del ONNX de salida")
    p.add_argument(
        "--imgsz", type=int, nargs="+", default=[640],
        help="Lado de la entrada, o alto y ancho (p. ej. --imgsz 256 640)",
    )
    p.add_argument("--opset", type=int, help="Opset ONNX (por defecto, el de ultralytics)")
    p.add_argument("--half", action="store_true", help="Exportar en FP16 (requiere GPU)")
    p.add_argument("--device", help="Dispositivo de export: cpu o indice de GPU")
    p.add_argument("--sin-simplify", action="store_true", help="No simplificar el grafo")
    p.add_argument(
        "--solo-inspeccionar",
        action="store_true",
        help="Solo abre el checkpoint y muestra nc, names y parametros",
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

    ruta_pt = args.pt or (config.dir_modelos / "best.pt")

    try:
        if args.solo_inspeccionar:
            info = inspeccionar_checkpoint(ruta_pt)
            print(json.dumps({
                "nc": info.nc,
                "task": info.task,
                "parametros": info.parametros,
                "imgsz_entrenamiento": info.imgsz_entrenamiento,
                "bytes_checkpoint": info.bytes_checkpoint,
                "names": info.names,
            }, indent=2, ensure_ascii=False))
            return 0

        sufijo = "_fp16" if args.half else ""
        destino = args.destino or (config.dir_modelos / f"model{sufijo}.onnx")

        resultado = exportar_onnx(
            ruta_pt=ruta_pt,
            destino=destino,
            imgsz=args.imgsz if len(args.imgsz) > 1 else args.imgsz[0],
            opset=args.opset,
            half=args.half,
            simplify=not args.sin_simplify,
            device=args.device,
        )
    except ExportError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - frontera del CLI
        logger.exception("Fallo exportando: %s", exc)
        return 1

    meta = destino.with_suffix(destino.suffix + ".json")
    meta.write_text(
        json.dumps(resultado.a_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Metadatos escritos en %s", meta)

    resumen = resultado.a_dict()
    resumen["names"] = f"<{len(resultado.names)} clases>"
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
