"""Descarga el modelo champion del MLflow Registry.

Deja en `models/`:
  - el artefacto (`best.pt` por defecto)
  - `provenance.json` con que version exactamente se ha traido
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sira.common.config import Config, ConfigIncompletaError
from sira.common.logging import configurar
from sira.registry import ClienteRegistry, RegistryError

logger = logging.getLogger("sira.fetch")


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alias", help="Alias a resolver (por defecto, el de la config)")
    p.add_argument("--especialidad", help="Especialidad (por defecto, la de la config)")
    p.add_argument("--artefacto", default="best.pt", help="Fichero a descargar")
    p.add_argument("--destino", type=Path, help="Ruta de salida del artefacto")
    p.add_argument(
        "--forzar",
        action="store_true",
        help="Descarga aunque ya exista un artefacto con la misma procedencia",
    )
    return p


def _procedencia_vigente(ruta: Path, modelo: str, alias: str, version: str) -> bool:
    """True si `provenance.json` ya describe exactamente esta version."""
    if not ruta.is_file():
        return False
    try:
        previa = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        previa.get("modelo_registrado") == modelo
        and previa.get("alias") == alias
        and previa.get("version") == version
    )


def main(argv: list[str] | None = None) -> int:
    configurar()
    args = construir_parser().parse_args(argv)

    try:
        config = Config.desde_entorno()
    except ConfigIncompletaError as exc:
        logger.error("%s", exc)
        return 2

    especialidad = args.especialidad or config.especialidad
    modelo = config.modelo_registrado
    if args.especialidad:
        modelo = modelo.replace(config.especialidad, especialidad)
    alias = args.alias or config.alias

    destino = args.destino or (config.dir_modelos / args.artefacto)
    ruta_procedencia = config.dir_modelos / "provenance.json"

    cliente = ClienteRegistry(
        tracking_uri=config.tracking_uri,
        usuario=config.usuario,
        password=config.password,
        api_prefix=config.api_prefix,
        artifact_prefix=config.artifact_prefix,
    )

    try:
        version = cliente.resolver_alias(modelo, alias)

        # Descargar 40 MB otra vez para obtener el mismo fichero no aporta nada, pero
        # hay que resolver el alias igualmente: el champion pudo cambiar.
        if (
            not args.forzar
            and destino.is_file()
            and _procedencia_vigente(ruta_procedencia, modelo, alias, version.version)
        ):
            logger.info(
                "Ya esta descargada la version %s de '%s' (%s). Usa --forzar para repetir.",
                version.version, modelo, destino,
            )
            return 0

        procedencia = cliente.traer_champion(
            version=version,
            destino=destino,
            artefacto=args.artefacto,
        )
    except RegistryError as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - frontera del CLI
        logger.error("Fallo hablando con el registry: %s", exc)
        return 1

    ruta_procedencia.parent.mkdir(parents=True, exist_ok=True)
    ruta_procedencia.write_text(
        json.dumps(procedencia.a_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    logger.info("Procedencia escrita en %s", ruta_procedencia)
    print(json.dumps(procedencia.a_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
