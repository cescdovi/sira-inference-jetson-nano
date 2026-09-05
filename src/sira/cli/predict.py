"""Ejecuta un engine sobre una imagen o un frame de video.

Con `--comparar` contrasta el resultado contra Ultralytics sobre el `.pt` original. Es
el criterio de aceptacion del pre/postproceso propio: si diverge, el benchmark estaria
midiendo un bug en vez del modelo.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from sira.common.config import Config, ConfigIncompletaError
from sira.common.logging import configurar
from sira.engine import EngineError
from sira.infer import RuntimeTensorRT, postprocesar
from sira.infer.postprocess import CONF_DEFECTO, IOU_DEFECTO, Deteccion
from sira.infer.preprocess import preparar_tensor

logger = logging.getLogger("sira.predict")


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, help="Engine a usar")
    p.add_argument("--imagen", type=Path, help="Imagen de entrada")
    p.add_argument("--video", type=Path, help="Video del que extraer un frame")
    p.add_argument("--frame", type=int, default=0, help="Indice del frame a extraer")
    p.add_argument("--conf", type=float, default=CONF_DEFECTO)
    p.add_argument("--iou", type=float, default=IOU_DEFECTO)
    p.add_argument("--salida-anotada", type=Path, help="Guardar la imagen con las cajas")
    p.add_argument("--comparar", action="store_true", help="Contrastar con Ultralytics sobre el .pt")
    p.add_argument("--pt", type=Path, help="Checkpoint para --comparar")
    return p


def cargar_frame(args: argparse.Namespace) -> np.ndarray:
    if args.imagen:
        imagen = cv2.imread(str(args.imagen))
        if imagen is None:
            raise SystemExit(f"No se pudo leer la imagen {args.imagen}")
        return imagen
    if not args.video:
        raise SystemExit("Hace falta --imagen o --video")

    captura = cv2.VideoCapture(str(args.video))
    if not captura.isOpened():
        raise SystemExit(f"No se pudo abrir el video {args.video}")
    if args.frame:
        captura.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, imagen = captura.read()
    captura.release()
    if not ok:
        raise SystemExit(f"No se pudo leer el frame {args.frame} de {args.video}")
    return imagen


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def comparar_con_ultralytics(
    imagen: np.ndarray,
    ruta_pt: Path,
    mias: list[Deteccion],
    conf: float,
    iou: float,
    imgsz: int = 640,
) -> dict:
    """Empareja mis detecciones con las de Ultralytics por IoU maximo.

    `rect=False` es imprescindible para que la comparacion sea justa: por defecto
    Ultralytics hace inferencia **rectangular**, ajustando la entrada a la relacion de
    aspecto del frame, mientras que el ONNX exportado tiene entrada cuadrada fija. Sin
    forzarlo, las cajas difieren por el preproceso y parece un fallo del postproceso.
    """
    from ultralytics import YOLO

    modelo = YOLO(str(ruta_pt))
    resultado = modelo.predict(
        imagen, conf=conf, iou=iou, imgsz=imgsz, rect=False, verbose=False, device=0
    )[0]
    ref = [
        (
            tuple(float(v) for v in caja.xyxy[0].tolist()),
            float(caja.conf[0]),
            int(caja.cls[0]),
        )
        for caja in resultado.boxes
    ]

    emparejadas = []
    sin_usar = list(range(len(ref)))
    for det in mias:
        mejor, mejor_iou = None, 0.0
        for k in sin_usar:
            solape = _iou(det.bbox, ref[k][0])
            if solape > mejor_iou:
                mejor, mejor_iou = k, solape
        if mejor is not None and mejor_iou > 0.5:
            sin_usar.remove(mejor)
            emparejadas.append({
                "iou": round(mejor_iou, 4),
                "score_mio": round(det.score, 4),
                "score_ultralytics": round(ref[mejor][1], 4),
                "delta_score": round(abs(det.score - ref[mejor][1]), 4),
                "misma_clase": det.class_id == ref[mejor][2],
            })

    return {
        "detecciones_mias": len(mias),
        "detecciones_ultralytics": len(ref),
        "emparejadas": len(emparejadas),
        "solo_mias": len(mias) - len(emparejadas),
        "solo_ultralytics": len(sin_usar),
        "iou_minimo": round(min((e["iou"] for e in emparejadas), default=0.0), 4),
        "delta_score_maximo": round(max((e["delta_score"] for e in emparejadas), default=0.0), 4),
        "clases_coinciden": all(e["misma_clase"] for e in emparejadas),
        "parejas": emparejadas[:10],
    }


def main(argv: list[str] | None = None) -> int:
    configurar()
    args = construir_parser().parse_args(argv)

    try:
        config = Config.desde_entorno()
    except ConfigIncompletaError as exc:
        logger.error("%s", exc)
        return 2

    ruta_engine = args.engine or (config.dir_modelos / "model_fp32.engine")
    imagen = cargar_frame(args)

    try:
        with RuntimeTensorRT(ruta_engine) as runtime:
            tensor, params = preparar_tensor(imagen, runtime.imgsz, runtime.dtype_entrada)
            salida = runtime.inferir(tensor)
            detecciones = postprocesar(
                salida, params, names=runtime.names, conf=args.conf, iou=args.iou
            )
    except EngineError as exc:
        logger.error("%s", exc)
        return 1

    informe: dict = {
        "engine": ruta_engine.name,
        "resolucion": [imagen.shape[1], imagen.shape[0]],
        "detecciones": [d.a_dict() for d in detecciones],
    }

    if args.comparar:
        ruta_pt = args.pt or (config.dir_modelos / "best.pt")
        informe["comparacion"] = comparar_con_ultralytics(
            imagen, ruta_pt, detecciones, args.conf, args.iou, imgsz=runtime.imgsz
        )

    if args.salida_anotada:
        anotada = imagen.copy()
        for d in detecciones:
            x1, y1, x2, y2 = (int(v) for v in d.bbox)
            cv2.rectangle(anotada, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(
                anotada, f"{d.class_name} {d.score:.2f}", (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA,
            )
        args.salida_anotada.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.salida_anotada), anotada)
        informe["anotada"] = str(args.salida_anotada)

    print(json.dumps(informe, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
