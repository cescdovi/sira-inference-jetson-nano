# Fase 7 — Contenerización

## Objetivo

Hacer el despliegue **reproducible**: versiones fijadas e imagen que cualquiera pueda
reconstruir, en lugar de un entorno montado a mano siguiendo comandos de un documento.

## Estado

**completada** — 2026-09-06. Imagen construida, engines regenerados dentro y benchmark
repetido.

## Por qué ahora, y por qué no antes

En la fase 0 los contenedores se descartaron porque el usuario de la máquina no tenía
sudo y no había Docker Engine nativo. Se pivotó a un venv, que funcionó y con el que se
midió todo el benchmark.

Con acceso root disponible, el bloqueo desaparece. Pero conviene ser preciso sobre lo que
esto aporta **ahora**: no mejora el rendimiento —es el mismo TensorRT sobre la misma
GPU— ni desbloquea nada pendiente. Lo que aporta es reproducibilidad: hoy el entorno se
reconstruye siguiendo unos `pip install` sin versiones fijadas, lo que es una garantía
pobre para reproducir un benchmark dentro de seis meses.

## Preparación del host

Ejecutada por el usuario como root:

| Componente | Versión |
| --- | --- |
| Docker Engine | 29.8.0 |
| containerd.io | 2.3.4 |
| nvidia-container-toolkit | 1.20.0 |

`nvidia-ctk runtime configure --runtime=docker` registró el runtime `nvidia`, y
`francesc` se añadió al grupo `docker`.

Verificado que un contenedor ve la GPU:

```
$ docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 \
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv
name, compute_cap, driver_version
NVIDIA GeForce RTX 5060, 12.0, 580.159.03
```

Nota: `docker-desktop` seguía instalado y aportaba su propio `docker-ce-cli`. No hubo
conflicto —`dnf` instaló `docker-ce` junto a él— pero el daemon que corre es el nativo,
que es el único que expone la GPU a los contenedores.

## Decisiones de diseño de la imagen

**Base de Python limpia, no una imagen de CUDA.** Es la decisión menos obvia y la más
importante. El stack de CUDA y TensorRT llega por **wheels de pip** (`nvidia-cuda-runtime`,
`nvidia-cudnn-cu13`, `nvidia-cublas`, `tensorrt-cu13-libs`…), que es exactamente cómo se
instaló y se midió el entorno de referencia. Partir de `nvidia/cuda` metería una segunda
copia del runtime en la imagen y abriría la puerta a que se cargue una distinta de la
validada. El driver lo inyecta `nvidia-container-toolkit` en tiempo de ejecución.

**Versiones fijadas en `requirements.lock`.** Las 88 dependencias exactas del venv con el
que se midió el benchmark, obtenidas con `pip freeze`. Incluye la advertencia de que
cambiar `tensorrt` invalida los engines ya construidos, que llevan la versión grabada en
su sidecar y la comprueban al cargar.

**Modelos, datos y resultados como volúmenes, no capas.** Pesan cientos de MB, cambian a
menudo, y los engines son específicos de esta GPU y de esta versión de TensorRT: hornearlos
en la imagen sería empaquetar algo que no es portable.

**El puerto se publica sólo en `127.0.0.1` del host.** Dentro del contenedor se escucha en
`0.0.0.0` porque la red está aislada; la exposición real la decide el mapeo del compose.

## Resultados

### Versiones dentro del contenedor

```
python 3.13.15
tensorrt 11.2.1.2
torch 2.14.0+cu130 | cuda: True
numpy 2.5.2
builder OK
```

Coinciden con el entorno de referencia. El único desajuste es el micro de Python
(3.13.15 frente a 3.13.13 del host), irrelevante.

### Engines regenerados dentro del contenedor

| Engine | Build | Tamaño |
| --- | --- | --- |
| `model_r_fp32.engine` | 5,87 s | 82 565 172 B |
| `model_r_fp16.engine` | 6,56 s | 41 818 236 B |

Los tamaños difieren en unos pocos KB respecto a los construidos en el venv: la selección
de *tactics* del builder no es determinista bit a bit. Los engines del venv se conservaron
como `.engine.venv` por si hiciera falta compararlos.

### Benchmark dentro del contenedor

| Configuración | Infer p50 | e2e p50 | FPS | VRAM (MiB) |
| --- | --- | --- | --- | --- |
| PyTorch `.pt` | 8,53 | 8,53 | 108,0 | 1950 |
| TensorRT FP32 | 3,52 | 4,25 | 223,4 | 1798 |
| TensorRT FP16 | **1,49** | 3,26 | 298,0 | 1744 |

**La latencia de inferencia es idéntica** a la medida en el venv (3,52 y 1,48 ms). El
throughput sostenido queda un 4-5 % por debajo (298 frente a 312,8 fps) y el p99 empeora,
lo cual es coherente con algo de sobrecarga de CPU del contenedor: la inferencia en GPU no
se ve afectada, el trabajo de CPU sí, ligeramente. La VRAM sube unos 200 MiB.

**La divergencia sale cifra por cifra idéntica**, que es el resultado importante de esta
fase:

| | Contenedor | venv |
| --- | --- | --- |
| FP16 vs FP32, IoU medio | 0,99709 | 0,99709 |
| FP16 vs FP32, Δ score medio | 0,00099 | 0,00099 |
| PyTorch vs FP32, IoU medio | 0,99945 | 0,99945 |

Mismo lockfile, mismos resultados numéricos. Eso es exactamente lo que se buscaba.

### Servicio en ejecución

`docker compose up -d`, con el vídeo y los modelos montados como volúmenes:

```
fps 29.98 | procesados 948 | descartados 1
  decode        p50=  1.09  p95=  1.71
  preproceso    p50=  1.81  p95=  1.90
  inferencia    p50=  1.52  p95=  1.54
  postproceso   p50=  0.22  p95=  0.32
  anotado       p50=  0.87  p95=  1.14
  encode        p50=  4.72  p95=  4.88
```

Indistinguible de la ejecución en el venv (1,53 ms de inferencia, 4,70 de encode).

## Incidencia: el servicio arrancaba inalcanzable

La primera ejecución con compose levantó el contenedor correctamente —los logs mostraban
el motor arrancado y procesando— pero `curl` al puerto publicado no obtenía nada.

La pista estaba en el propio log: `Sirviendo ... en http://127.0.0.1:8080`. El servicio
escuchaba en el **loopback del contenedor**, donde el mapeo de puertos no puede alcanzarlo.

La causa: el `Dockerfile` fija `SIRA_SERVE_HOST=0.0.0.0`, pero el `env_file: ../.env` del
compose trae `SIRA_SERVE_HOST=127.0.0.1` —correcto para ejecutar en el host— y **`env_file`
tiene prioridad sobre el `ENV` de la imagen**.

Resuelto añadiendo un bloque `environment:` en el compose, que a su vez gana sobre
`env_file`. La exposición real la sigue decidiendo `ports`, que publica únicamente en el
loopback del host, así que el servicio no queda expuesto a la red.

Es un buen ejemplo de por qué un valor por defecto seguro en el host puede ser el valor
equivocado dentro de un contenedor, y de por qué conviene leer los logs antes de suponer
que el problema es la red.

## Mejora pendiente: la imagen pesa 18,8 GB

Casi toda la masa son `torch`, `torchvision` y `ultralytics` con su stack de CUDA. Pero
**el servicio no los necesita**: `serve` sólo usa `tensorrt`, `cuda-python`, `numpy`,
`opencv`, `fastapi` y `uvicorn`. Torch y Ultralytics hacen falta únicamente para `export`
y para el baseline de PyTorch de `bench`.

Separar el lockfile en runtime y herramientas permitiría una imagen de despliegue de
unos pocos GB, dejando la pesada sólo para las tareas de preparación y medición. No está
hecho: implicaría dos lockfiles y repetir la verificación.
