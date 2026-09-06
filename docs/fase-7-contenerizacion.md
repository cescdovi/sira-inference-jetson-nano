# Fase 7 — Contenerización

## Objetivo

Hacer el despliegue **reproducible**: versiones fijadas e imagen que cualquiera pueda
reconstruir, en lugar de un entorno montado a mano siguiendo comandos de un documento.

## Estado

**en curso** — 2026-09-06.

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

## Pendiente

- Reconstruir los engines dentro del contenedor.
- Repetir el benchmark y confirmar que los 312,8 fps se mantienen.
