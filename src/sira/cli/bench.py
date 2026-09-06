"""Benchmark: PyTorch vs TensorRT FP32 vs TensorRT FP16.

Cada configuracion se mide sobre los mismos frames, precargados en RAM. El engine FP32
es la referencia frente a la que se mide la divergencia de las demas.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sira.bench import comparar_detecciones, medir_pytorch, medir_tensorrt
from sira.bench.runner import cargar_frames
from sira.common.config import Config, ConfigIncompletaError
from sira.common.logging import configurar
from sira.infer.postprocess import CONF_DEFECTO, IOU_DEFECTO

logger = logging.getLogger("sira.bench")


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, help="Video de entrada")
    p.add_argument("--frames", type=int, default=300, help="Frames a medir")
    p.add_argument("--salto", type=int, default=1, help="Tomar 1 de cada N frames")
    p.add_argument(
        "--engines", nargs="+", default=["model_r_fp32", "model_r_fp16"],
        help="Engines a medir, sin extension. El primero es la referencia",
    )
    p.add_argument("--sin-pytorch", action="store_true", help="Omitir el baseline PyTorch")
    p.add_argument("--conf", type=float, default=CONF_DEFECTO)
    p.add_argument("--iou", type=float, default=IOU_DEFECTO)
    p.add_argument("--salida", type=Path, help="Fichero JSON de resultados")
    return p


def main(argv: list[str] | None = None) -> int:
    configurar()
    args = construir_parser().parse_args(argv)

    try:
        config = Config.desde_entorno()
    except ConfigIncompletaError as exc:
        logger.error("%s", exc)
        return 2

    ruta_video = args.video
    if ruta_video is None:
        candidatos = sorted(config.dir_datos.glob("*.mp4"))
        if not candidatos:
            logger.error("No hay ningun .mp4 en %s. Pasa --video.", config.dir_datos)
            return 1
        ruta_video = candidatos[0]

    frames = cargar_frames(ruta_video, args.frames, args.salto)

    mediciones = []
    referencia = None
    hw = None

    for i, nombre in enumerate(args.engines):
        ruta = config.dir_modelos / f"{nombre}.engine"
        if not ruta.is_file():
            logger.warning("No existe %s, se omite", ruta)
            continue
        logger.info("Midiendo %s", nombre)
        medicion, dets = medir_tensorrt(ruta, frames, conf=args.conf, iou=args.iou)
        if referencia is None:
            referencia = dets
            hw = (medicion.entrada[2], medicion.entrada[3])
        else:
            medicion.divergencia = comparar_detecciones(referencia, dets)
        mediciones.append(medicion)

    if not mediciones:
        logger.error("No se pudo medir ningun engine")
        return 1

    if not args.sin_pytorch:
        logger.info("Midiendo baseline PyTorch")
        medicion, dets = medir_pytorch(
            config.dir_modelos / "best.pt", frames, hw, conf=args.conf, iou=args.iou
        )
        medicion.divergencia = comparar_detecciones(referencia, dets)
        mediciones.append(medicion)

    resultados = {
        "video": ruta_video.name,
        "frames": len(frames),
        "conf": args.conf,
        "iou": args.iou,
        "referencia_divergencia": args.engines[0],
        "mediciones": [m.a_dict() for m in mediciones],
    }

    salida = args.salida or (config.dir_modelos.parent / "results" / "bench.json")
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Resultados en %s", salida)

    print(f"\n{'configuracion':22s} {'entrada':>14s} {'infer p50':>10s} {'p99':>8s} "
          f"{'e2e p50':>9s} {'FPS':>8s} {'VRAM MiB':>9s} {'det/frame':>10s}")
    print("-" * 96)
    for m in mediciones:
        print(f"{m.configuracion:22s} {str(m.entrada[2])+'x'+str(m.entrada[3]):>14s} "
              f"{m.latencia_inferencia_ms['p50']:>10.2f} {m.latencia_inferencia_ms['p99']:>8.2f} "
              f"{m.latencia_e2e_ms['p50']:>9.2f} {m.fps_sostenido:>8.1f} "
              f"{m.vram_pico_mib:>9.0f} {m.detecciones_por_frame:>10.3f}")
    print()
    for m in mediciones:
        if m.divergencia:
            d = m.divergencia
            print(f"{m.configuracion}: divergencia vs {args.engines[0]} -> "
                  f"IoU medio {d.get('iou_medio')}, min {d.get('iou_minimo')}, "
                  f"delta score medio {d.get('delta_score_medio')}, "
                  f"sin pareja {d.get('solo_en_referencia')}/{d.get('solo_en_candidata')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
