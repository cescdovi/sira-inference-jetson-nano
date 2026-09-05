# Fase 1 — Cliente del MLflow Registry

## Objetivo

Resolver el alias `champion` del modelo registrado y descargar su artefacto, dejando
constancia de **qué versión exactamente** se ha traído.

## Estado

**completada** — 2026-09-05. Verificada contra el servidor real.

## Decisión: REST directo, no el cliente de MLflow

El repositorio de entrenamiento habla con este servidor a través del cliente oficial de
`mlflow`, pero para lograrlo tiene que **parchear en caliente varias clases internas de
la librería** (`camma-laparoscopy/src/tracking/mlflow_tracker.py:143-230`): reescribe
`RestStore._METHOD_TO_INFO`, `_V3_METHOD_TO_INFO`, las del registro de modelos, las de
webhooks, y además sustituye `MlflowArtifactsRepository.resolve_uri` porque lleva
`/api/2.0/mlflow-artifacts/artifacts` cableado.

Funciona, pero depende de detalles privados de mlflow y se rompe con cada actualización.

Aquí sólo hace falta **resolver un alias y bajar un fichero**, así que se habla REST
directamente con `requests`:

- Sin dependencia de `mlflow` (que arrastra medio ecosistema de ML a una máquina cuyo
  único cometido es servir inferencia).
- Sin monkey-patching.
- Los prefijos son **configuración** (`SIRA_API_PREFIX`, `SIRA_ARTIFACT_PREFIX`), no un
  parche.

## API real del servidor

Sondeada antes de escribir nada. Todas las rutas cuelgan de `/ajax-api/2.0`, no de
`/api/2.0`, y usan autenticación básica.

| Paso | Endpoint |
| --- | --- |
| Resolver alias | `GET /ajax-api/2.0/mlflow/registered-models/alias?name=<modelo>&alias=<alias>` |
| Datos del run | `GET /ajax-api/2.0/mlflow/runs/get?run_id=<id>` |
| Listar artefactos | `GET /ajax-api/2.0/mlflow/artifacts/list?run_id=<id>` |
| Descargar | `GET /ajax-api/2.0/mlflow-artifacts/artifacts/<base>/<fichero>` |

El `artifact_uri` llega como `mlflow-artifacts:/2/<run_id>/artifacts` y hay que traducirlo
al prefijo HTTP del proxy; eso lo hace `ClienteRegistry._url_artefacto`.

### Estado del registry

```
sira-yolo11m-General   aliases: []
sira-yolo11n-Common    aliases: []
sira-yolo11n-General   aliases: [(challenger, 21), (champion, 17), (previous, 16)]
sira-yolo11n-Tora      aliases: []
```

El champion de `General` es la **versión 17**, run `d6c3bb77458d47988be0bf0a143b877e`.
Artefactos del run: `best.pt` (40 480 620 B), `last.pt`, matrices de confusión,
`results.csv`, `results.png`, imágenes de validación y un directorio `diagnostics`.

## Resultados

```
$ python -m sira.cli.fetch
INFO sira.registry.client | Alias 'champion' de 'sira-yolo11n-General' -> version 17 (run d6c3bb77458d47988be0bf0a143b877e)
INFO sira.registry.client | Descargando best.pt -> models/best.pt
INFO sira.registry.client | Descargado 40.5 MB, sha256=fe77008ec059d9fd
INFO sira.fetch | Procedencia escrita en models/provenance.json
```

`models/provenance.json`:

```json
{
  "modelo_registrado": "sira-yolo11n-General",
  "alias": "champion",
  "version": "17",
  "run_id": "d6c3bb77458d47988be0bf0a143b877e",
  "artefacto": "best.pt",
  "artefacto_sha256": "fe77008ec059d9fd8a64f2365eb12561324008b484aa637e93a771ce8866e09c",
  "artefacto_bytes": 40480620,
  "tracking_uri": "https://iacom.uv.es/sira-mlflow",
  "descargado_en": "2026-09-05T19:03:25+00:00"
}
```

### Casos de error, verificados

| Caso | Comportamiento |
| --- | --- |
| Segunda ejecución | No redescarga; compara `provenance.json`. `--forzar` lo salta |
| Alias inexistente | `ModeloNoDisponibleError: No hay alias 'no_existe' para 'sira-yolo11n-General'...` |
| Especialidad inexistente | Mismo error tipado, con el nombre registrado que se buscó |
| Artefacto inexistente | `ArtefactoNoEncontradoError` **listando los artefactos disponibles** |
| Alias `previous` | Resuelve a versión 16, run `fb76af75d8b9...` |

## Verificación en la máquina de inferencia

No basta con que funcione en el Mac. Desplegado y ejecutado en `192.168.1.44`:

```
$ cd ~/sira-inference-jetson-nano && git checkout main
$ PYTHONPATH=src ~/sira-venv/bin/python -m sira.cli.fetch
INFO sira.registry.client | Alias 'champion' de 'sira-yolo11n-General' -> version 17 (run d6c3bb77458d47988be0bf0a143b877e)
INFO sira.registry.client | Descargando best.pt -> /home/francesc/sira-inference-jetson-nano/models/best.pt
INFO sira.registry.client | Descargado 40.5 MB, sha256=fe77008ec059d9fd
```

**Mismo `sha256` que en el Mac** (`fe77008ec059d9fd8a64f2365eb12561324008b484aa637e93a771ce8866e09c`):
la descarga es byte a byte idéntica en ambas máquinas.

El `.env` se creó en la máquina con permisos `600` y **no procede del repositorio**: está
en `.gitignore` y se transfirió aparte.

## Incidencias

**El alias se resolvía dos veces.** La primera versión pedía el alias en `main()` para
decidir si había que descargar y otra vez dentro de `traer_champion()`. Se veía en las
trazas: dos peticiones HTTP y dos líneas de log idénticas. Corregido pasando la
`VersionModelo` ya resuelta.

**MLflow no usa `RESOURCE_DOES_NOT_EXIST` para un alias inexistente.** Devuelve **HTTP
400** con `error_code: INVALID_PARAMETER_VALUE`:

```json
{"error_code": "INVALID_PARAMETER_VALUE", "message": "Registered model alias no_existe not found."}
```

Reserva `RESOURCE_DOES_NOT_EXIST` para otras cosas — un run inexistente, por ejemplo. Sin
tratarlo aparte, "el champion todavía no está promocionado" —que es un estado legítimo,
de arranque en frío— quedaba indistinguible de una petición mal formada. La clasificación
está en `ClienteRegistry.resolver_alias`, con el motivo comentado en el código porque
depende de un comportamiento del servidor que no es evidente.

**El repositorio tenía dos ramas y la de trabajo no era la por defecto.** El repositorio
de GitHub se creó con `master` como rama por defecto, conteniendo un único *Empty commit*.
El repositorio local estaba en `main`, así que el `push` creó una rama nueva en vez de
actualizar la existente, y el primer `git clone` en la máquina de inferencia se quedó en
`master` — vacío:

```
$ git ls-remote --heads origin
b45fba4...  refs/heads/main     <- el trabajo
e03cada...  refs/heads/master   <- Empty commit, rama por defecto
```

Resuelto en la máquina con `git checkout main`. **Queda pendiente** cambiar la rama por
defecto del repositorio a `main` en GitHub: mientras no se haga, cualquier clon nuevo
caerá en `master` vacío. Es un cambio en la configuración del repositorio, así que lo
decide el usuario.

## Detalles de implementación que importan

- **Descarga a fichero temporal y `replace` al final.** Si se corta la conexión no queda
  un `best.pt` truncado con pinta de válido. Con 40 MB por una wifi, es un caso real.
- **SHA-256 calculado durante el streaming**, sin releer el fichero.
- **La contraseña no aparece en `repr(Config)`.** Está enmascarada a propósito: el objeto
  de configuración se imprime justo cuando se depura un fallo de autenticación, que es el
  peor momento para filtrarla.
- **Sólo lectura.** Este repositorio consume lo que el pipeline de entrenamiento
  promociona; no registra ni promociona nada.

## Pendiente que arrastra a la fase siguiente

`models/best.pt` pesa **40 MB**, mucho más que los ~10 MB que ocuparían los pesos de un
YOLO11n en FP32. Es lo esperable en un checkpoint de entrenamiento (estado del
optimizador, EMA), pero **conviene confirmarlo al abrirlo** en la fase 2, junto con el
dato que sigue abierto en `CLAUDE.md` §9.2: **cuántas clases tiene realmente el modelo**.
El catálogo de `data_general.yaml` lista 133 entradas pero declara `keep_classes: [1]`, y
el número efectivo no es derivable del fichero de configuración. Se resolverá leyendo
`model.names` del propio checkpoint.
