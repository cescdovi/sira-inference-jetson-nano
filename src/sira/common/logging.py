"""Configuracion de logging comun a todos los puntos de entrada."""

from __future__ import annotations

import logging
import os
import sys

_FORMATO = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def configurar(nivel: str | None = None) -> None:
    """Configura el logging raiz. Idempotente.

    El nivel sale de `SIRA_LOG_LEVEL` si no se pasa explicitamente.
    """
    if logging.getLogger().handlers:
        return
    nivel = nivel or os.environ.get("SIRA_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, nivel.upper(), logging.INFO),
        format=_FORMATO,
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
