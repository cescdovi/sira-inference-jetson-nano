# Imagen de despliegue de SIRA.
#
# Base de Python limpia a proposito, no una imagen de CUDA: el stack de CUDA y TensorRT
# llega por wheels de pip (ver requirements.lock), que es exactamente como se instalo y
# se midio el entorno de referencia. Usar una base con CUDA propio anadiria una segunda
# copia del runtime y abriria la puerta a que se cargue una distinta de la validada.
#
# El acceso a la GPU lo aporta nvidia-container-toolkit al arrancar (`--gpus all`), que
# inyecta el driver del host.
FROM python:3.13-slim-bookworm

# libGL y libglib son las dependencias de sistema de opencv-python.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Las dependencias se copian e instalan antes que el codigo para que editar el codigo no
# invalide la capa de instalacion, que tarda varios minutos y pesa varios GB.
COPY requirements.lock ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.lock

COPY pyproject.toml ./
COPY src/ ./src/
COPY web/ ./web/

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# Dentro del contenedor se escucha en todas las interfaces porque la red esta aislada;
# la exposicion real la decide el mapeo de puertos del compose, que publica solo en
# 127.0.0.1 del host.
ENV SIRA_SERVE_HOST=0.0.0.0 \
    SIRA_SERVE_PORT=8080

EXPOSE 8080

CMD ["python", "-m", "sira.cli.serve"]
