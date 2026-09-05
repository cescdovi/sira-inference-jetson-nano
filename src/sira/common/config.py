"""Carga de configuracion desde entorno y `.env`.

Las credenciales nunca se hardcodean ni se escriben en logs (CLAUDE.md §7).
`Config.__repr__` enmascara la contrasena a proposito: es facil filtrarla sin querer
al depurar un fallo de autenticacion, que es justo cuando mas se imprime el objeto.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Raiz del repositorio: .../src/sira/common/config.py -> tres niveles arriba de src/
RAIZ = Path(__file__).resolve().parents[3]

DEFECTOS = {
    "MLFLOW_TRACKING_URI": "https://iacom.uv.es/sira-mlflow",
    "SIRA_SPECIALTY": "General",
    "SIRA_MODEL_ALIAS": "champion",
    # Este servidor publica la REST bajo /ajax-api, no bajo /api. Ver Config.api_prefix.
    "SIRA_API_PREFIX": "/ajax-api/2.0",
    "SIRA_ARTIFACT_PREFIX": "/ajax-api/2.0/mlflow-artifacts/artifacts",
    "SIRA_MODEL_TEMPLATE": "sira-yolo11n-{specialty}",
    "SIRA_SERVE_HOST": "127.0.0.1",
    "SIRA_SERVE_PORT": "8080",
}


def cargar_dotenv(ruta: Path | None = None) -> None:
    """Vuelca `.env` en `os.environ` sin pisar lo que ya venga del entorno.

    El entorno gana sobre el fichero para que se pueda sobreescribir puntualmente
    sin editar `.env`.
    """
    ruta = ruta or RAIZ / ".env"
    if not ruta.is_file():
        return
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, _, valor = linea.partition("=")
        os.environ.setdefault(clave.strip(), valor.strip().strip("'\""))


class ConfigIncompletaError(RuntimeError):
    """Falta una variable obligatoria. Se distingue de un fallo de red o de auth."""


@dataclass(frozen=True)
class Config:
    tracking_uri: str
    usuario: str
    password: str = field(repr=False)
    especialidad: str
    alias: str
    api_prefix: str
    artifact_prefix: str
    modelo_registrado: str
    dir_modelos: Path
    dir_datos: Path

    @classmethod
    def desde_entorno(cls) -> "Config":
        cargar_dotenv()

        def leer(clave: str) -> str:
            return os.environ.get(clave) or DEFECTOS.get(clave, "")

        usuario = leer("MLFLOW_TRACKING_USERNAME")
        password = leer("MLFLOW_TRACKING_PASSWORD")
        if not usuario or not password:
            raise ConfigIncompletaError(
                "Faltan MLFLOW_TRACKING_USERNAME o MLFLOW_TRACKING_PASSWORD. "
                "Copia .env.example a .env y rellenalos."
            )

        especialidad = leer("SIRA_SPECIALTY")
        return cls(
            tracking_uri=leer("MLFLOW_TRACKING_URI").rstrip("/"),
            usuario=usuario,
            password=password,
            especialidad=especialidad,
            alias=leer("SIRA_MODEL_ALIAS"),
            api_prefix="/" + leer("SIRA_API_PREFIX").strip("/"),
            artifact_prefix="/" + leer("SIRA_ARTIFACT_PREFIX").strip("/"),
            modelo_registrado=leer("SIRA_MODEL_TEMPLATE").format(specialty=especialidad),
            dir_modelos=RAIZ / "models",
            dir_datos=RAIZ / "data",
        )

    def __repr__(self) -> str:  # pragma: no cover - solo presentacion
        return (
            f"Config(tracking_uri={self.tracking_uri!r}, usuario={self.usuario!r}, "
            f"password='***', especialidad={self.especialidad!r}, alias={self.alias!r}, "
            f"modelo_registrado={self.modelo_registrado!r})"
        )
