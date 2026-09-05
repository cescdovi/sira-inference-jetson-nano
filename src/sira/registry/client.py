"""Cliente REST del MLflow Registry.

No se usa el cliente oficial de `mlflow` a proposito. Este despliegue publica la API
bajo `/ajax-api/2.0` en vez de `/api/2.0`, y el cliente oficial lleva esas rutas
cableadas: el repo de entrenamiento lo resuelve parcheando en caliente varias clases
internas de mlflow (`camma-laparoscopy/src/tracking/mlflow_tracker.py`), lo que funciona
pero se rompe con cada cambio de version de la libreria.

Aqui solo hay que resolver un alias y bajar un fichero, asi que se habla REST
directamente: sin dependencia de mlflow, sin monkey-patching y con los prefijos como
configuracion en vez de como parche.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import requests

logger = logging.getLogger(__name__)

TIEMPO_ESPERA = (10, 60)  # (conexion, lectura) en segundos
TROZO = 1024 * 1024  # 1 MiB


class RegistryError(RuntimeError):
    """Error generico hablando con el registry.

    Lleva el `error_code` de MLflow y el codigo HTTP para que quien llama pueda
    clasificar sin volver a parsear el cuerpo.
    """

    def __init__(self, mensaje: str, codigo: str = "", http: int = 0) -> None:
        super().__init__(mensaje)
        self.codigo = codigo
        self.http = http


class ModeloNoDisponibleError(RegistryError):
    """No existe el modelo registrado, o el alias no apunta a ninguna version.

    Es un caso legitimo (arranque en frio, especialidad sin promocionar todavia), no un
    fallo de red: por eso tiene tipo propio y no se enmascara como respuesta vacia.
    """


class ArtefactoNoEncontradoError(RegistryError):
    """La version existe pero el fichero pedido no esta entre sus artefactos."""


@dataclass(frozen=True)
class VersionModelo:
    """Version del modelo registrado a la que apunta un alias."""

    modelo_registrado: str
    alias: str
    version: str
    run_id: str
    source: str
    status: str


@dataclass(frozen=True)
class Procedencia:
    """Que se ha descargado exactamente. Se persiste junto al artefacto.

    Sin esto no se sabe que modelo se esta sirviendo ni el benchmark es reproducible
    (CLAUDE.md §9.5).
    """

    modelo_registrado: str
    alias: str
    version: str
    run_id: str
    artefacto: str
    artefacto_sha256: str
    artefacto_bytes: int
    tracking_uri: str
    descargado_en: str

    def a_dict(self) -> dict[str, Any]:
        return asdict(self)


class ClienteRegistry:
    """Lectura del MLflow Registry por REST.

    Solo lee: este repositorio consume lo que el pipeline de entrenamiento promociona,
    nunca registra ni promociona nada.
    """

    def __init__(
        self,
        tracking_uri: str,
        usuario: str,
        password: str,
        api_prefix: str = "/ajax-api/2.0",
        artifact_prefix: str = "/ajax-api/2.0/mlflow-artifacts/artifacts",
    ) -> None:
        self.tracking_uri = tracking_uri.rstrip("/")
        self.api_prefix = "/" + api_prefix.strip("/")
        self.artifact_prefix = "/" + artifact_prefix.strip("/")
        self._sesion = requests.Session()
        self._sesion.auth = (usuario, password)

    # -- HTTP ---------------------------------------------------------------

    def _get(self, ruta: str, **params: Any) -> dict[str, Any]:
        url = f"{self.tracking_uri}{self.api_prefix}/mlflow/{ruta.lstrip('/')}"
        respuesta = self._sesion.get(url, params=params or None, timeout=TIEMPO_ESPERA)

        if respuesta.status_code >= 400:
            # MLflow describe el motivo en el cuerpo. Un alias o un modelo inexistentes
            # llegan como 400 con RESOURCE_DOES_NOT_EXIST, no como 404, asi que mirar
            # solo el codigo HTTP los confundiria con una peticion mal formada.
            codigo, mensaje = self._detalle_error(respuesta)
            if codigo in {"RESOURCE_DOES_NOT_EXIST", "ENDPOINT_NOT_FOUND"} or \
                    respuesta.status_code == 404:
                raise ModeloNoDisponibleError(mensaje or f"No existe: {url}")
            raise RegistryError(
                f"{respuesta.status_code} en {url}: {mensaje or respuesta.reason}",
                codigo=codigo,
                http=respuesta.status_code,
            )

        return respuesta.json()

    @staticmethod
    def _detalle_error(respuesta: requests.Response) -> tuple[str, str]:
        """Extrae `(error_code, message)` del cuerpo de error de MLflow."""
        try:
            cuerpo = respuesta.json()
        except ValueError:
            return "", respuesta.text[:200]
        return cuerpo.get("error_code", ""), cuerpo.get("message", "")

    # -- Resolucion ---------------------------------------------------------

    def resolver_alias(self, modelo_registrado: str, alias: str) -> VersionModelo:
        """Resuelve `models:/<modelo>@<alias>` a una version concreta."""
        try:
            datos = self._get(
                "registered-models/alias", name=modelo_registrado, alias=alias
            )
        except ModeloNoDisponibleError:
            raise
        except RegistryError as exc:
            # MLflow reporta tanto un alias inexistente como un modelo inexistente con
            # INVALID_PARAMETER_VALUE, no con RESOURCE_DOES_NOT_EXIST (que si usa para
            # un run inexistente). Sin este caso especial, "el champion aun no esta
            # promocionado" seria indistinguible de una peticion mal formada.
            if exc.codigo == "INVALID_PARAMETER_VALUE" and "not found" in str(exc).lower():
                raise ModeloNoDisponibleError(
                    f"No hay alias '{alias}' para '{modelo_registrado}'. "
                    "O no esta promocionado todavia, o el nombre registrado no existe."
                ) from exc
            raise
        mv = datos.get("model_version")
        if not mv:
            raise ModeloNoDisponibleError(
                f"El alias '{alias}' no apunta a ninguna version de "
                f"'{modelo_registrado}'"
            )
        version = VersionModelo(
            modelo_registrado=modelo_registrado,
            alias=alias,
            version=str(mv["version"]),
            run_id=mv["run_id"],
            source=mv.get("source", ""),
            status=mv.get("status", ""),
        )
        logger.info(
            "Alias '%s' de '%s' -> version %s (run %s)",
            alias, modelo_registrado, version.version, version.run_id,
        )
        return version

    def artifact_uri_del_run(self, run_id: str) -> str:
        """URI base de artefactos del run, tal cual la reporta el servidor."""
        return self._get("runs/get", run_id=run_id)["run"]["info"]["artifact_uri"]

    def listar_artefactos(self, run_id: str, ruta: str = "") -> list[dict[str, Any]]:
        datos = self._get("artifacts/list", run_id=run_id, path=ruta) if ruta \
            else self._get("artifacts/list", run_id=run_id)
        return datos.get("files", [])

    # -- Descarga -----------------------------------------------------------

    def _url_artefacto(self, artifact_uri: str, ruta_relativa: str) -> str:
        """Traduce un `mlflow-artifacts:/...` a URL HTTP del proxy de artefactos.

        El servidor devuelve el `artifact_uri` con el esquema propio de MLflow; hay que
        mapearlo al prefijo HTTP que expone el reverse-proxy.
        """
        esquema = "mlflow-artifacts:/"
        if not artifact_uri.startswith(esquema):
            raise RegistryError(
                f"artifact_uri no proxificado ({artifact_uri!r}). Este cliente solo "
                "sabe descargar por el proxy de artefactos de MLflow."
            )
        base = artifact_uri[len(esquema):].strip("/")
        return (
            f"{self.tracking_uri}{self.artifact_prefix}/"
            f"{base}/{ruta_relativa.lstrip('/')}"
        )

    def descargar_artefacto(
        self,
        run_id: str,
        ruta_relativa: str,
        destino: Path,
        artifact_uri: str | None = None,
    ) -> tuple[str, int]:
        """Descarga un artefacto a `destino`. Devuelve `(sha256, bytes)`.

        Escribe primero a un fichero temporal y renombra al final: si la descarga se
        corta, no queda un `best.pt` truncado que parezca valido.
        """
        artifact_uri = artifact_uri or self.artifact_uri_del_run(run_id)
        url = self._url_artefacto(artifact_uri, ruta_relativa)

        destino.parent.mkdir(parents=True, exist_ok=True)
        temporal = destino.with_suffix(destino.suffix + ".parcial")

        logger.info("Descargando %s -> %s", ruta_relativa, destino)
        sha = hashlib.sha256()
        total = 0
        with self._sesion.get(url, stream=True, timeout=TIEMPO_ESPERA) as respuesta:
            if respuesta.status_code == 404:
                raise ArtefactoNoEncontradoError(
                    f"'{ruta_relativa}' no existe en los artefactos del run {run_id}"
                )
            respuesta.raise_for_status()
            with temporal.open("wb") as fh:
                for trozo in respuesta.iter_content(chunk_size=TROZO):
                    if not trozo:
                        continue
                    fh.write(trozo)
                    sha.update(trozo)
                    total += len(trozo)

        temporal.replace(destino)
        digest = sha.hexdigest()
        logger.info("Descargado %.1f MB, sha256=%s", total / 1e6, digest[:16])
        return digest, total

    # -- Operacion de alto nivel -------------------------------------------

    def traer_champion(
        self,
        version: VersionModelo,
        destino: Path,
        artefacto: str = "best.pt",
    ) -> Procedencia:
        """Descarga el artefacto de una version ya resuelta y devuelve su procedencia.

        Recibe la `VersionModelo` en vez de resolver el alias por su cuenta: quien llama
        necesita la version antes de decidir si hay que descargar, y resolverla dos veces
        seria una peticion de mas y una traza duplicada y confusa.
        """
        artifact_uri = self.artifact_uri_del_run(version.run_id)

        disponibles = {f["path"] for f in self.listar_artefactos(version.run_id)}
        if artefacto not in disponibles:
            raise ArtefactoNoEncontradoError(
                f"'{artefacto}' no esta entre los artefactos del run "
                f"{version.run_id}. Disponibles: {sorted(disponibles)}"
            )

        digest, total = self.descargar_artefacto(
            version.run_id, artefacto, destino, artifact_uri=artifact_uri
        )
        return Procedencia(
            modelo_registrado=version.modelo_registrado,
            alias=version.alias,
            version=version.version,
            run_id=version.run_id,
            artefacto=artefacto,
            artefacto_sha256=digest,
            artefacto_bytes=total,
            tracking_uri=self.tracking_uri,
            descargado_en=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
