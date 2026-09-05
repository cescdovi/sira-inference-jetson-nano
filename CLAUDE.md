# SIRA — Inferencia optimizada con TensorRT

Servir el modelo *champion* de detección optimizado con TensorRT sobre vídeo en tiempo
real, en una workstation Fedora con RTX 5060. El champion actual es un detector de **una
sola clase** con ~20 M de parámetros (§9.2), pese a estar registrado como `yolo11n`.

Este repositorio es el lado de **despliegue**. El entrenamiento, la evaluación y la
promoción viven en `../camma-laparoscopy`, que publica los modelos en el MLflow
Registry. Aquí solo se consume lo ya promocionado.

> **Historial de cambios de rumbo** (importante para no confundirse con material
> antiguo): el destino era una Jetson Nano con JetPack 4.6, y el runtime iba a ser C++
> porque aquella máquina traía Python 3.6. Ambas cosas se descartaron el 2026-09-05.
> Hoy: **workstation x86_64 y runtime en Python**. Todo lo que dependía de TensorRT 8.2,
> CUDA 10.2, memoria unificada o ausencia de INT8 ha dejado de aplicar. El nombre del
> repositorio conserva la referencia a la Jetson por motivos históricos.

---

## 1. Decisiones de partida

- **Python**, no C++. La única razón para C++ era el Python 3.6 de la Jetson; sin esa
  restricción no compensa.
- **Pipeline manual**, no Ultralytics de punta a punta. Ultralytics se usa
  exclusivamente para el paso `.pt` → ONNX (inevitable: hay que deserializar el
  checkpoint). A partir de ahí, construcción del engine con la API de TensorRT y
  pre/postproceso propios. Se gana control sobre precisión, calibración INT8 y
  batching, que es donde el proyecto tiene algo que demostrar.
- **Vídeo en tiempo real**, no imágenes sueltas. Es el caso de uso real y condiciona el
  diseño del servidor: streaming, backpressure y FPS sostenido, no latencia por
  petición.
- **Contenedores Ubuntu** sobre el host Fedora, partiendo de imágenes NGC de NVIDIA que
  ya traen TensorRT, CUDA y cuDNN emparejados.

---

## 2. Entorno de destino

Máquina de inferencia, verificada por ssh el 2026-09-05:

| Componente | Valor |
| --- | --- |
| Acceso | `francesc@192.168.1.44` (misma LAN), auth por contraseña |
| SO | Fedora Linux 43 (Workstation), kernel 6.19.14-200.fc43.x86_64 |
| GPU | NVIDIA GeForce RTX 5060, **compute capability 12.0 (sm_120)**, Blackwell |
| VRAM | 8151 MiB, dedicada |
| Driver | 580.159.03 (expone CUDA 13.0) |
| CUDA (host) | 13.0.88 en `/usr/local/cuda-13.0` |
| Python (host) | 3.14.4 por defecto; 3.13 y 3.11 también instalados |
| GCC (host) | 15.2.1; `g++-14` disponible |
| Docker | 29.4.3, engine nativo (contexto `default`, socket unix), Compose v5.0.0 |
| CPU / RAM | 16 núcleos / 31 GiB |
| Disco | 223 GB libres en `/`, 546 GB libres en `/home` |

Las versiones del **host** importan poco: el trabajo ocurre dentro del contenedor, que
fija su propio CUDA, Python y TensorRT. Lo que sí condiciona desde fuera es el driver
(580.159.03) y la arquitectura de GPU (sm_120).

### Pendiente de instalar en el host

- **nvidia-container-toolkit** — no está, y sin él `docker run --gpus` no funciona.
  Es el único requisito real del host y bloquea todo lo demás.

TensorRT, cuDNN y CUDA no hace falta instalarlos en el host: vienen en la imagen.

### Consecuencias de este hardware

- **FP16, INT8, FP8 y FP4 disponibles.** Blackwell los soporta. INT8 exige calibración
  con datos representativos; no es un flag que se activa y ya.
- **VRAM dedicada, 8 GB.** YOLO11n a 640×640 ocupa ~10 MB en FP32. La memoria no es
  preocupación en ningún punto del flujo.
- **El plan de TensorRT no es portable** entre versiones de TRT ni arquitecturas de GPU.
  Se construye en la máquina de destino y no se versiona en git.

---

## 3. Flujo de trabajo: Mac → GitHub → máquina de inferencia

```
Mac (edición, diseño, revisión)
  └─ git commit && git push  →  github.com/cescdovi/sira-inference-jetson-nano
                                   └─ 192.168.1.44: git pull && docker compose up
```

**En el Mac no se puede ejecutar nada de esto**: no hay GPU NVIDIA. Toda verificación
real se hace por ssh contra `192.168.1.44`.

- **Nunca afirmes que algo funciona sin haberlo ejecutado por ssh.** En el Mac como
  mucho puedes razonar sobre el código; no lo presentes como validado.
- El ciclo es: editar en el Mac → commit → `git pull` en la máquina → ejecutar allí.
  Aunque el cambio sea de una línea, pasa por commit: no se editan ficheros
  directamente en la máquina de inferencia.
- Los binarios pesados (`.pt`, `.onnx`, `.engine`, vídeos) **no van a git**.
- El acceso ssh es por contraseña. Instalar una clave pública ahorraría fricción, pero
  eso lo decide el usuario: no lo hagas por tu cuenta.

---

## 4. Arquitectura del pipeline

Cuatro etapas, cada una un punto de entrada independiente. Que estén separadas importa:
el engine se construye una vez y el servidor arranca sin reconstruirlo.

```
  [1] fetch     MLflow Registry ──> models/best.pt
  [2] export    best.pt ──ultralytics──> models/model.onnx
  [3] build     model.onnx ──API TensorRT──> models/model_{fp32,fp16,int8}.engine
  [4] serve     engine + FastAPI + web/  ──> stream anotado en el navegador
  [5] bench     .pt vs engines ──> informe de latencia y FPS
```

### [1] `fetch` — cliente del MLflow Registry

Resuelve el alias `champion` y descarga el artefacto. Parámetros del servidor, tomados
de `../camma-laparoscopy/config/services/mlflow_params.yaml`:

- Tracking URI: `https://iacom.uv.es/sira-mlflow`
- Modelo registrado: `sira-yolo11n-{specialty}`. Alias: `champion` (existe también
  `previous`).
- El artefacto está en la **raíz** de los artefactos del run: `best.pt`, no
  `weights/best.pt`.

⚠️ **El cliente estándar de `mlflow` no funciona tal cual contra este servidor.** La
API REST se publica bajo `/ajax-api/2.0` y `/ajax-api/3.0`, y las rutas por defecto
devuelven 404. El repo de entrenamiento ya resuelve esto reescribiendo las URLs:
**reutiliza el enfoque de `../camma-laparoscopy/src/tracking/mlflow_tracker.py:148-220`**
en vez de reinventarlo o pelearte con el cliente.

Debe registrar en disco la **procedencia**: modelo registrado, versión, `run_id` y hash
del artefacto. Sin eso no se sabe qué se está sirviendo ni el benchmark es reproducible.

### [2] `export` — `.pt` → ONNX

Único punto donde interviene Ultralytics, y es inevitable: un `.pt` es un pickle que
serializa el objeto `DetectionModel` con referencias a clases Python, así que
deserializarlo exige el intérprete y el paquete.

- Export con `imgsz` explícito coherente con el preproceso, y `simplify=True`.
- **`nms=False`**: el NMS se hace en el postproceso propio, no incrustado en el grafo.
- Versiones fijadas en el `requirements` del contenedor.

Entrada 640×640, coherente con el `imgsz` de entrenamiento leído del propio checkpoint.
Las características reales del modelo están en §9.2: no las asumas a partir del nombre
del modelo registrado, que no corresponde con su tamaño.

### [3] `build` — ONNX → engine TensorRT

⚠️ **TensorRT 11 no se controla con flags de precisión.** Verificado sobre la instalación
real (`docs/fase-0-preparacion-host.md`): no existen `BuilderFlag.FP16` ni
`BuilderFlag.INT8`, toda red es *strongly typed*, y `IInt8Calibrator` y sus subclases han
sido eliminadas. **La precisión la determina el grafo ONNX, no el builder.**

Por tanto no se construyen tres engines desde un mismo ONNX: se prepara **un ONNX por
precisión**.

| Precisión | Cómo se obtiene |
| --- | --- |
| FP32 | ONNX tal cual → build directo |
| FP16 | ONNX convertido a precisión mixta (ModelOpt AutoCast) → build |
| INT8 | ONNX con nodos Q/DQ insertados y calibrados → build |

La calibración vive en la herramienta de cuantización, no en TensorRT. Los datos de
calibración salen del `.mp4` (§5), con tramos disjuntos para calibrar y evaluar.

Independiente de la precisión:

- `ILogger` propio volcado al log del proyecto; los avisos del parser son la principal
  pista cuando el ONNX no encaja.
- `EXPLICIT_BATCH` ya no es un flag: el batch explícito es el único modo.
- Serializa cada plan con metadatos (versión de TensorRT, precisión, hash del ONNX de
  origen) y **valida esos metadatos al cargar**: un engine de otra versión no debe
  cargarse a ciegas.
- La construcción es cara; se cachea y **el servidor nunca la repite al arrancar**.

### [4] `serve` — streaming de vídeo con inferencia

Es la etapa con más decisiones de diseño, y donde es fácil equivocarse midiendo.

- **FastAPI** como servidor. UI en HTML/CSS/JS estático servido desde `web/`, sin build
  step ni npm.
- **Transporte:** MJPEG sobre HTTP (`multipart/x-mixed-replace`) es lo más simple y
  funciona en un `<img>` sin JavaScript. WebSocket con frames JPEG es la alternativa si
  hace falta enviar también metadatos por frame.
- **Decodificación:** `cv2.VideoCapture` es lo directo, pero decodifica en CPU y puede
  convertirse en el cuello de botella. **Mide el tiempo de decodificación por separado
  del de inferencia**, o atribuirás a TensorRT un límite que impone el decodificador.
  Si domina, NVDEC por hardware es la salida.
- **Backpressure: descartar frames, nunca encolarlos.** Si la inferencia va más lenta
  que el vídeo y encolas, la latencia crece sin límite hasta que el stream va minutos
  por detrás. Cola de tamaño 1 y descarte del frame viejo. Es el error clásico de este
  tipo de servidor.
- **Anotación:** con MJPEG lo natural es dibujar las cajas en el servidor y emitir el
  frame ya anotado.
- El engine se carga **una vez** al arrancar; contexto de ejecución y búferes se
  reservan una vez y se reutilizan. El contexto de ejecución no es seguro para uso
  concurrente: serializa el acceso o usa uno por hilo.

Endpoints mínimos: el stream anotado, `GET /health` (estado y engine cargado), `GET /`
(UI). Por defecto escucha en `127.0.0.1` y se accede desde el Mac por túnel ssh
(`ssh -L 8080:localhost:8080 francesc@192.168.1.44`). Exponerlo en la LAN es una
decisión explícita: no hay autenticación.

### [5] `bench` — comparativa

Matriz de cuatro configuraciones sobre el mismo vídeo:

| Configuración | Qué aporta |
| --- | --- |
| PyTorch `.pt` (ultralytics) | Baseline: el modelo tal como salió de entrenamiento |
| TensorRT FP32 | Aísla fusión de capas y autotuning, a precisión constante |
| TensorRT FP16 | Precisión reducida sin calibración |
| TensorRT INT8 | Cuantización calibrada |

Los tramos intermedios son lo que hace útil el benchmark: con solo dos puntos obtienes
un agregado que no sabes descomponer.

**Qué medir:** latencia de inferencia pura y extremo a extremo **por separado**;
p50/p95/p99 en vez de medias; FPS sostenido sobre el vídeo completo; frames descartados;
VRAM pico; y tiempo de construcción de cada engine.

**Paridad de detecciones — leer §5 antes de diseñar esto.**

Protocolo: calentar y descartar las primeras iteraciones, fijar el reloj de la GPU si se
quiere reducir varianza, un proceso por configuración, y el mismo vídeo en el mismo
orden.

---

## 5. Datos: un único `.mp4`, sin etiquetas

Los datos disponibles en la máquina serán **un vídeo `.mp4` copiado por scp**. Esto
tiene dos consecuencias que condicionan el benchmark y que no se pueden esquivar:

1. **No hay ground truth, así que no hay mAP.** No se puede afirmar "INT8 pierde X
   puntos de mAP". Lo que sí se puede medir es **divergencia respecto al engine FP32**
   tomado como referencia: coincidencia de cajas por IoU, acuerdo de clase, diferencias
   de score y de número de detecciones por frame. Repórtalo como lo que es —divergencia
   entre configuraciones, no precisión absoluta— y no lo disfraces de métrica de
   calidad.
2. **Calibrar y evaluar con segmentos distintos del vídeo.** La calibración INT8 no
   necesita etiquetas, solo frames representativos, así que el vídeo sirve. Pero si
   calibras y mides sobre los mismos frames, el resultado está contaminado. Parte el
   vídeo explícitamente y deja constancia de qué tramo fue a cada cosa.

Si en algún momento hay acceso al split de validación etiquetado (vía DVC/MinIO, como en
el repo de entrenamiento), el mAP absoluto pasa a ser posible y merece la pena
replantear esta sección.

---

## 6. Estructura del repositorio

Orientativa; ajústala si el código lo pide, pero mantén las etapas separadas.

```
pyproject.toml          # dependencias y entry points
docker/                 # Dockerfile (base NGC) y compose
src/sira/
  common/               # config, logging, carga de .env
  registry/             # cliente MLflow (con el rewrite de /ajax-api)
  export/               # .pt -> onnx
  engine/               # builder de engines + calibrador INT8
  infer/                # preproceso (letterbox), postproceso, NMS, runtime TRT
  video/                # decodificación, pipeline de frames, backpressure
  server/               # FastAPI, streaming, endpoints
  bench/                # medición y generación del informe
web/                    # index.html, app.js, style.css
models/                 # gitignored: .pt, .onnx, .engine, metadatos
data/                   # gitignored: el .mp4
scripts/                # utilidades de despliegue
```

Sin prefijos de proyecto en carpetas ni ficheros: el repo ya identifica el contexto. Los
puntos de entrada se llaman `fetch`, `export`, `build`, `serve`, `bench`.

---

## 7. Secretos y configuración

Credenciales y parámetros en `.env` (gitignored), con `.env.example` como plantilla,
siguiendo la convención de `../camma-laparoscopy`. **Nunca** se hardcodean ni se
escriben en logs.

Variables: `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`,
`MLFLOW_TRACKING_PASSWORD`, `SIRA_SPECIALTY`, `SIRA_MODEL_ALIAS`, `SIRA_SERVE_HOST`,
`SIRA_SERVE_PORT`.

Las credenciales del MLflow son de **administrador**. No las incluyas en código,
ejemplos, mensajes de commit ni salidas de depuración.

---

## 8. Fases

0. ✅ **Preparar el host** — completada (`docs/fase-0-preparacion-host.md`). TensorRT
   11.2.1.2 en `~/sira-venv`, verificado construyendo un engine real sobre la GPU.
1. **Cliente del registry** — resolver `champion` y descargar `best.pt` con procedencia.
2. **Export a ONNX** — con `nms=False` e `imgsz` explícito. Un ONNX por precisión (§4.3).
3. **Builder de engines** — FP32 y FP16 primero; INT8 con cuantización explícita después.
4. **Runtime de inferencia** — pre/postproceso propios, verificados contra las
   detecciones de referencia de Ultralytics sobre el `.pt`.
5. **Servidor de vídeo** — streaming, backpressure, UI.
6. **Benchmark** — la matriz de §4.5.

Cierra cada fase con algo ejecutable y verificado en la máquina antes de pasar a la
siguiente. Es un prototipo: prioriza el camino completo funcionando sobre la
sofisticación de cada pieza.

---

## 9. Especificaciones

Contratos concretos. Cuando algo aquí choque con una suposición durante la
implementación, gana esto; si resulta estar mal, corrígelo aquí antes de seguir.

### 9.1 Stack fijado

**No se usan contenedores.** El usuario de la máquina no tiene sudo, así que no se puede
instalar `nvidia-container-toolkit` ni un Docker Engine nativo (solo hay Docker Desktop,
que además no expone la GPU a los contenedores). Ver `docs/fase-0-preparacion-host.md`.

El stack se instala en **espacio de usuario, en un venv**, con los wheels de PyPI. Se
obtiene la misma versión de TensorRT que traía el contenedor NGC previsto, los wheels son
manylinux —así que da igual que Fedora no esté soportado por NVIDIA— y arrastran sus
propias dependencias de CUDA.

| | |
| --- | --- |
| Entorno | venv en `~/sira-venv` en la máquina de inferencia |
| Python | **3.13** (`python3.13 -m venv`) |
| **TensorRT** | **11.2.1.2** (`pip install tensorrt==11.2.1.2` → `tensorrt_cu13`) |
| CUDA | 13.x vía wheels; el sistema tiene 13.0.88 |
| Driver del host | 580.159.03, expone CUDA 13.0 |
| GPU | RTX 5060, sm_120 |

⚠️ **Python 3.13, no 3.14.** Los wheels de TensorRT llegan hasta 3.13, y el `python3` por
defecto de la máquina es 3.14. Crear el venv con `python3.13` explícitamente.

Dependencias sobre el venv: `ultralytics` (solo export), `fastapi`, `uvicorn`,
`opencv-python-headless`, `numpy`, `cuda-python`. Versiones fijadas en `requirements.txt`.

Si en algún momento hay acceso root, el camino de contenedores (`nvcr.io/nvidia/tensorrt:26.08-py3`,
misma TensorRT 11.2.1.2) sigue siendo válido y no obligaría a cambiar el código.

**Decidido quedarse en TensorRT 11**, no bajar a 10.x. Razones: la 11.2.1.2 ya está
verificada sobre este driver y esta GPU (bajar reabriría esa incertidumbre, incluido el
soporte de sm_120); el trabajo extra que impone la 11 está confinado al INT8, que es la
última fase y además descartable sin invalidar el benchmark; y FP32 y FP16 —el núcleo del
pipeline— no necesitan herramientas adicionales.

### 9.2 Contrato del modelo

**Verificado abriendo el checkpoint y el ONNX generado** (`docs/fase-2-export-onnx.md`),
no deducido de la configuración del repo de entrenamiento:

| | |
| --- | --- |
| Registrado como | `sira-yolo11n-General`, versión 17 |
| Parámetros reales | **20 053 779 (~20 M)** |
| Clases | **`nc = 1`** → `{0: "1_Fenestratedb bipolar forceps"}` |
| Tarea | `detect` |
| `imgsz` de entrenamiento | 640 |

⚠️ **El nombre del modelo registrado no corresponde a su tamaño.** 20 M parámetros
coinciden con un YOLO11**m** (~20,1 M), no con un YOLO11**n** (~2,6 M). El registry tiene
además un `sira-yolo11m-General` sin alias. Esto afecta a las expectativas de latencia y
de VRAM: es un modelo casi 8 veces mayor que el asumido. **Pendiente de aclarar con el
usuario**; no cambia el código, pero sí lo que cabe esperar del benchmark.

⚠️ **Es un detector de una sola clase.** El catálogo de `data_general.yaml` lista 133
entradas, pero `keep_classes: [1]` conserva literalmente el índice 1 del catálogo. Por eso
`nc` y `names` **se leen del artefacto y se persisten** en `model.onnx.json`: deducirlos
del YAML habría dado 133 y roto el postproceso en silencio.

**Contrato del ONNX** (`models/model.onnx`, opset 18, 80 428 503 B):

| | Nombre | Forma | Tipo |
| --- | --- | --- | --- |
| Entrada | `images` | `[1, 3, 640, 640]` | FLOAT |
| Salida | `output0` | `[1, 5, 8400]` | FLOAT |

La salida es `[1, 4+nc, 8400]` con `nc=1`. Exportado con `nms=False` y `dynamic=False`.
Canales RGB, valores en `[0, 1]`.

### 9.3 Preproceso

Debe reproducir exactamente lo que hace Ultralytics, o las detecciones divergirán del
baseline y el benchmark medirá el bug en vez del modelo:

1. **Letterbox** conservando la relación de aspecto, relleno **`(114, 114, 114)`**.
2. BGR → RGB.
3. `/255.0` → `float32`.
4. HWC → CHW, añadir dimensión de batch.

**Guarda `scale` y `(pad_x, pad_y)`**: hacen falta para devolver las cajas a las
coordenadas del frame original.

### 9.4 Postproceso

1. Transponer `[1, 5, 8400]` → `[8400, 5]`.
2. `(cx, cy, w, h)` → `(x1, y1, x2, y2)`.
3. Score = máximo sobre las `nc` clases. Con `nc=1` es directamente la columna 4, pero
   **escribe el código genérico sobre `nc`**: se lee de `model.onnx.json`, y un
   reentrenamiento con más clases no debe obligar a tocar el postproceso.
   Umbral de confianza **0.45**.
4. **NMS por clase**, umbral IoU **0.5**.
5. Revertir letterbox con `scale` y padding.

Los umbrales salen de `config/yolo/predict.yaml` del repo de entrenamiento; deben ser
configurables pero con esos valores por defecto, para que la comparación con el baseline
sea justa.

### 9.5 Artefactos y nomenclatura

```
models/best.pt                    # descargado del registry
models/model.onnx
models/model_fp32.engine          # referencia del benchmark
models/model_fp16.engine
models/model_int8.engine
models/model_<precision>.engine.json   # sidecar de metadatos
models/provenance.json
data/<nombre>.mp4                 # copiado por scp
results/bench_<precision>.json
```

**Sidecar de cada engine:** `trt_version`, `cuda_version`, `gpu_name`, `compute_cap`,
`precision`, `onnx_sha256`, `imgsz`, `nc`, `names`, `build_seconds`, `created_at`.

**`provenance.json`:** `registered_model`, `version`, `run_id`, `alias`,
`artifact_sha256`, `fetched_at`.

**Al cargar un engine, valida `trt_version` y `compute_cap` contra el entorno actual y
falla con un error explícito si no coinciden.** Un engine de otra versión o de otra GPU
no debe cargarse a ciegas.

### 9.6 API

| Ruta | Qué hace |
| --- | --- |
| `GET /` | UI estática |
| `GET /health` | `{status, precision, model_version, run_id, nc, gpu, fps}` |
| `GET /stream` | MJPEG (`multipart/x-mixed-replace`), frames ya anotados |
| `GET /detections` | Últimas detecciones en JSON (para clientes que no quieren imagen) |
| `POST /infer` | Una imagen → detecciones. Necesario para tests y para el bench |

Esquema de detección, único en toda la aplicación:

```json
{"bbox": [x1, y1, x2, y2], "score": 0.87, "class_id": 3, "class_name": "..."}
```

Coordenadas en píxeles del frame original, no del tensor de entrada.

### 9.7 Vídeo

- **Cola de tamaño 1 con descarte del frame viejo.** No encolar nunca (§9.9).
- **Cronometrar por etapas y por separado:** `decode_ms`, `preprocess_ms`, `infer_ms`,
  `postprocess_ms`, `encode_ms`. Sin esta separación no se sabe si el techo lo pone
  TensorRT o el decodificador, y es habitual que lo ponga el decodificador.
- Si la decodificación domina, NVDEC por hardware es la salida.

### 9.8 Benchmark

Matriz de §4.5 (`.pt` → TRT FP32 → FP16 → INT8), un proceso por configuración.

**Particionado del vídeo, declarado explícitamente en el informe:** un tramo para
calibración INT8 y otro, disjunto, para evaluación. Calibrar y medir sobre los mismos
frames invalida el resultado.

**Salida** en `results/bench_<precision>.json`: percentiles p50/p95/p99 de cada etapa,
FPS sostenido, frames descartados, VRAM pico, tiempo de construcción del engine, y las
métricas de divergencia frente a FP32 (§5) — nunca etiquetadas como mAP.

---

## 10. Qué no hacer

- Dar por buena una ejecución sin haberla lanzado por ssh en `192.168.1.44`.
- Commitear pesos, `.onnx`, `.engine`, vídeos o credenciales.
- Encolar frames en el servidor cuando la inferencia se retrasa: descartar, no acumular.
- Reportar divergencia entre configuraciones como si fuera mAP. Sin etiquetas no hay
  precisión absoluta (§5).
- Calibrar INT8 y evaluar sobre los mismos frames.
- Mover un `.engine` entre máquinas o entre versiones de TensorRT.
- Copiar ejemplos de TensorRT 8.x **ni 10.x** sin adaptarlos. Es la mayor parte del
  material que hay por internet y la API de la 11 difiere en puntos centrales: sin flags
  de precisión, sin calibradores INT8, sin `setPrecision`/`setDynamicRange`, sin
  `EXPLICIT_BATCH`. Comprueba contra la instalación real antes de dar por buena cualquier
  llamada.
- Delegar en Ultralytics más allá del export: el pre/postproceso y el runtime son
  propios por decisión (§1).
- Reimplementar lo que ya resuelve `../camma-laparoscopy`: entrenamiento, promoción y
  criterios de champion viven allí.
