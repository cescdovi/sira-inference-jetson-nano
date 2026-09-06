"""Sirve el video anotado por HTTP con una interfaz web minima."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from sira.common.config import RAIZ, Config, ConfigIncompletaError
from sira.common.logging import configurar
from sira.infer.postprocess import CONF_DEFECTO, IOU_DEFECTO
from sira.server import crear_app
from sira.video import MotorVideo

logger = logging.getLogger("sira.serve")


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, help="Engine a servir")
    p.add_argument("--video", type=Path, help="Video de entrada")
    p.add_argument("--host", help="Interfaz de escucha")
    p.add_argument("--port", type=int, help="Puerto")
    p.add_argument("--conf", type=float, default=CONF_DEFECTO)
    p.add_argument("--iou", type=float, default=IOU_DEFECTO)
    p.add_argument("--calidad-jpeg", type=int, default=80)
    p.add_argument("--fps", type=float, help="Ritmo objetivo (por defecto, el del video)")
    p.add_argument("--sin-bucle", action="store_true", help="No repetir el video al acabar")
    return p


def main(argv: list[str] | None = None) -> int:
    configurar()
    args = construir_parser().parse_args(argv)

    try:
        config = Config.desde_entorno()
    except ConfigIncompletaError as exc:
        logger.error("%s", exc)
        return 2

    import os

    ruta_engine = args.engine or (config.dir_modelos / "model_r_fp16.engine")
    if not ruta_engine.is_file():
        logger.error("No existe el engine %s. Construyelo primero con `build`.", ruta_engine)
        return 1

    ruta_video = args.video
    if ruta_video is None:
        candidatos = sorted(config.dir_datos.glob("*.mp4"))
        if not candidatos:
            logger.error("No hay ningun .mp4 en %s. Pasa --video.", config.dir_datos)
            return 1
        ruta_video = candidatos[0]

    motor = MotorVideo(
        ruta_video=ruta_video,
        ruta_engine=ruta_engine,
        conf=args.conf,
        iou=args.iou,
        bucle=not args.sin_bucle,
        calidad_jpeg=args.calidad_jpeg,
        fps_objetivo=args.fps,
    )

    app = crear_app(motor, RAIZ / "web")

    host = args.host or os.environ.get("SIRA_SERVE_HOST", "127.0.0.1")
    port = args.port or int(os.environ.get("SIRA_SERVE_PORT", "8080"))

    if host not in {"127.0.0.1", "localhost"}:
        logger.warning(
            "Escuchando en %s: el servicio queda expuesto en la red y no tiene "
            "autenticacion.", host,
        )

    import uvicorn

    logger.info("Sirviendo %s con %s en http://%s:%d", ruta_video.name, ruta_engine.name, host, port)
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
