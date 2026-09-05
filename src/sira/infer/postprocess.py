"""Postproceso: decodificacion de la salida cruda y NMS.

El ONNX se exporta con `nms=False` a proposito, asi que el NMS es nuestro. Todo se
escribe generico sobre `nc` aunque el modelo actual tenga una sola clase: `nc` se lee de
los metadatos del engine, y un reentrenamiento no debe obligar a tocar esto
(CLAUDE.md §9.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sira.infer.preprocess import Letterbox

CONF_DEFECTO = 0.45
IOU_DEFECTO = 0.5


@dataclass(frozen=True)
class Deteccion:
    """Una deteccion en coordenadas del frame original, en pixeles."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float
    class_id: int
    class_name: str

    def a_dict(self) -> dict[str, Any]:
        return {
            "bbox": [round(float(v), 2) for v in self.bbox],
            "score": round(float(self.score), 4),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
        }


def nms_por_clase(
    cajas: np.ndarray, scores: np.ndarray, clases: np.ndarray, iou: float
) -> np.ndarray:
    """NMS independiente por clase. Devuelve los indices que sobreviven.

    Separar por clase importa: dos objetos de clases distintas pueden solaparse
    legitimamente, y un NMS global eliminaria uno de los dos.
    """
    conservados: list[int] = []
    for clase in np.unique(clases):
        (idx,) = np.where(clases == clase)
        idx = idx[np.argsort(-scores[idx])]
        while idx.size:
            actual = idx[0]
            conservados.append(int(actual))
            if idx.size == 1:
                break
            solape = _iou(cajas[actual], cajas[idx[1:]])
            idx = idx[1:][solape <= iou]
    return np.array(sorted(conservados), dtype=np.int64)


def _iou(caja: np.ndarray, otras: np.ndarray) -> np.ndarray:
    x1 = np.maximum(caja[0], otras[:, 0])
    y1 = np.maximum(caja[1], otras[:, 1])
    x2 = np.minimum(caja[2], otras[:, 2])
    y2 = np.minimum(caja[3], otras[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (caja[2] - caja[0]) * (caja[3] - caja[1])
    areas = (otras[:, 2] - otras[:, 0]) * (otras[:, 3] - otras[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def postprocesar(
    salida: np.ndarray,
    params: Letterbox,
    names: dict[int, str] | None = None,
    conf: float = CONF_DEFECTO,
    iou: float = IOU_DEFECTO,
) -> list[Deteccion]:
    """Convierte la salida cruda `[1, 4+nc, N]` en detecciones sobre el frame original."""
    pred = np.asarray(salida, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[0]
    # [4+nc, N] -> [N, 4+nc]
    pred = pred.T

    cajas_cxcywh = pred[:, :4]
    scores_clase = pred[:, 4:]
    if scores_clase.size == 0:
        return []

    mejores = scores_clase.max(axis=1)
    clases = scores_clase.argmax(axis=1)

    guarda = mejores >= conf
    if not guarda.any():
        return []
    cajas_cxcywh, mejores, clases = cajas_cxcywh[guarda], mejores[guarda], clases[guarda]

    # cx,cy,w,h -> x1,y1,x2,y2
    mitad = cajas_cxcywh[:, 2:4] / 2
    cajas = np.concatenate(
        [cajas_cxcywh[:, :2] - mitad, cajas_cxcywh[:, :2] + mitad], axis=1
    )

    conservados = nms_por_clase(cajas, mejores, clases, iou)
    cajas, mejores, clases = cajas[conservados], mejores[conservados], clases[conservados]

    # Revertir el letterbox: quitar el padding y deshacer la escala.
    cajas[:, [0, 2]] -= params.pad_x
    cajas[:, [1, 3]] -= params.pad_y
    cajas /= params.escala
    cajas[:, [0, 2]] = cajas[:, [0, 2]].clip(0, params.ancho_original)
    cajas[:, [1, 3]] = cajas[:, [1, 3]].clip(0, params.alto_original)

    names = names or {}
    return [
        Deteccion(
            bbox=(float(c[0]), float(c[1]), float(c[2]), float(c[3])),
            score=float(s),
            class_id=int(k),
            class_name=names.get(int(k), str(int(k))),
        )
        for c, s, k in zip(cajas, mejores, clases)
    ]
