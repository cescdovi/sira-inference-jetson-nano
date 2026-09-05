"""Preproceso: letterbox y normalizacion.

Debe reproducir **exactamente** lo que hace Ultralytics. Cualquier divergencia aqui
desplaza las cajas, y el benchmark acabaria midiendo el bug en vez del modelo
(CLAUDE.md §9.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

RELLENO = (114, 114, 114)


@dataclass(frozen=True)
class Letterbox:
    """Parametros de la transformacion, necesarios para revertir las cajas."""

    escala: float
    pad_x: float
    pad_y: float
    alto_original: int
    ancho_original: int


def letterbox(
    imagen: np.ndarray, imgsz: int = 640, relleno: tuple[int, int, int] = RELLENO
) -> tuple[np.ndarray, Letterbox]:
    """Redimensiona conservando la relacion de aspecto y rellena hasta `imgsz`.

    El relleno se reparte a ambos lados (imagen centrada), como en Ultralytics: si se
    pusiera todo a un lado, las cajas saldrian desplazadas justo la mitad del padding.
    """
    alto, ancho = imagen.shape[:2]
    escala = min(imgsz / alto, imgsz / ancho)

    sin_pad = (int(round(ancho * escala)), int(round(alto * escala)))
    dw = (imgsz - sin_pad[0]) / 2
    dh = (imgsz - sin_pad[1]) / 2

    if (ancho, alto) != sin_pad:
        imagen = cv2.resize(imagen, sin_pad, interpolation=cv2.INTER_LINEAR)

    arriba, abajo = int(round(dh - 0.1)), int(round(dh + 0.1))
    izq, der = int(round(dw - 0.1)), int(round(dw + 0.1))
    imagen = cv2.copyMakeBorder(
        imagen, arriba, abajo, izq, der, cv2.BORDER_CONSTANT, value=relleno
    )

    return imagen, Letterbox(
        escala=escala, pad_x=dw, pad_y=dh, alto_original=alto, ancho_original=ancho
    )


def preparar_tensor(imagen_bgr: np.ndarray, imgsz: int = 640, dtype=np.float32):
    """BGR uint8 -> tensor NCHW normalizado, mas los parametros del letterbox.

    `dtype` sale del engine: uno construido desde un ONNX FP16 espera FLOAT16, y pasarle
    float32 produce basura sin ningun error.
    """
    acolchada, params = letterbox(imagen_bgr, imgsz)
    rgb = cv2.cvtColor(acolchada, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=dtype)
    tensor /= dtype(255.0) if hasattr(dtype, "__call__") else 255.0
    return tensor, params
