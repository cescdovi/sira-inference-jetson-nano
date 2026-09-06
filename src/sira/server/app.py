"""API HTTP y streaming del video anotado.

El transporte es **MJPEG** (`multipart/x-mixed-replace`): funciona en un `<img>` sin
JavaScript, no necesita negociacion y es trivial de depurar con `curl`. WebRTC daria
menos latencia, pero para un prototipo la simplicidad gana.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from sira.video import MotorVideo

logger = logging.getLogger(__name__)

SEPARADOR = "sirastream"


def crear_app(motor: MotorVideo, dir_web: Path) -> FastAPI:
    app = FastAPI(title="SIRA — inferencia sobre video", docs_url="/api/docs")

    @app.on_event("startup")
    def _arrancar() -> None:
        motor.arrancar()

    @app.on_event("shutdown")
    def _parar() -> None:
        motor.parar()

    @app.get("/health")
    def health() -> JSONResponse:
        datos = motor.instantanea()
        return JSONResponse(datos, status_code=200 if datos["corriendo"] else 503)

    @app.get("/detections")
    def detections() -> JSONResponse:
        datos = motor.instantanea()
        return JSONResponse(
            {"detecciones": datos["detecciones"], "fps": datos["fps"]}
        )

    @app.get("/metrics")
    def metrics() -> JSONResponse:
        datos = motor.instantanea()
        return JSONResponse(
            {
                "fps": datos["fps"],
                "frames_leidos": datos["frames_leidos"],
                "frames_procesados": datos["frames_procesados"],
                "frames_descartados": datos["frames_descartados"],
                "tiempos_ms": datos["tiempos_ms"],
            }
        )

    @app.get("/stream")
    def stream() -> StreamingResponse:
        def generar() -> Iterator[bytes]:
            visto = -1
            while True:
                jpeg, visto = motor.ultimo_jpeg(visto)
                if jpeg is None:
                    break
                yield (
                    b"--" + SEPARADOR.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )

        return StreamingResponse(
            generar(),
            media_type=f"multipart/x-mixed-replace; boundary={SEPARADOR}",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/")
    def indice() -> FileResponse:
        return FileResponse(dir_web / "index.html")

    if dir_web.is_dir():
        app.mount("/static", StaticFiles(directory=dir_web), name="static")

    return app
