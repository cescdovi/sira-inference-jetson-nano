# Cómo funciona y cómo se ejecuta

Este repositorio coge el modelo de detección que el pipeline de entrenamiento ha
promocionado en MLflow, lo optimiza con TensorRT y lo sirve sobre vídeo en tiempo real.

El recorrido completo es:

```
MLflow Registry  →  best.pt  →  ONNX  →  engine TensorRT  →  servidor de vídeo
                                                          ↘  benchmark
```

Cada paso tiene su propio documento con el detalle, las salidas reales y las incidencias
que aparecieron. Aquí va la versión corta: qué hace cada paso, cómo se ejecuta y qué
conviene tener en cuenta.

---

## Antes de empezar

Hace falta la máquina de inferencia preparada (Docker con acceso a GPU) y un fichero
`.env` con las credenciales de MLflow. Se copia de la plantilla:

```bash
cp .env.example .env      # y se rellenan usuario y contraseña
```

Los comandos de abajo se lanzan todos dentro del contenedor. Para no repetir la misma
parrafada, conviene definir un atajo:

```bash
sira() {
  docker run --rm --gpus all \
    -v "$PWD/models:/app/models" \
    -v "$PWD/data:/app/data" \
    -v "$PWD/results:/app/results" \
    --env-file .env sira-inference:0.1.0 python -m sira.cli."$@"
}
```

A partir de ahí, `sira fetch`, `sira export`, etc.

---

## 1. Preparar la máquina

Se instala Docker Engine y el *NVIDIA Container Toolkit*, que es lo que permite a un
contenedor ver la GPU. Después se construye la imagen del proyecto, que lleva TensorRT,
PyTorch y todas las dependencias con las versiones exactas fijadas en
`requirements.lock`.

```bash
# Una sola vez, como root
dnf -y install docker-ce docker-ce-cli containerd.io docker-compose-plugin
systemctl enable --now docker
usermod -aG docker $USER

curl -sL https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo \
  > /etc/yum.repos.d/nvidia-container-toolkit.repo
dnf -y install nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Comprobar que un contenedor ve la GPU antes de seguir
docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi

# Construir la imagen del proyecto (tarda, son varios GB)
docker build -f docker/Dockerfile -t sira-inference:0.1.0 .
```

**A tener en cuenta.** La imagen pesa unos 19 GB, casi todo PyTorch y Ultralytics. El
servicio en sí no los necesita —sólo hacen falta para exportar el modelo y para el
baseline del benchmark—, así que hay margen para una imagen de despliegue mucho más
ligera si algún día interesa.

Detalle en [fase-0](fase-0-preparacion-host.md) y [fase-7](fase-7-contenerizacion.md).

---

## 2. Descargar el modelo

Se pregunta al registry de MLflow qué versión tiene el alias `champion` para la
especialidad configurada, se descarga su `best.pt` y se deja al lado un
`provenance.json` con el número de versión, el `run_id` y el hash del fichero, para saber
en todo momento qué se está sirviendo.

```bash
sira fetch
```

**A tener en cuenta.** Si se vuelve a lanzar y el champion no ha cambiado, no vuelve a
descargar; `--forzar` lo salta. El servidor de MLflow de este proyecto publica su API
bajo `/ajax-api` en lugar de `/api`, así que el cliente oficial de `mlflow` no sirve: se
habla REST directamente.

Detalle en [fase-1](fase-1-cliente-registry.md).

---

## 3. Exportar a ONNX

Se convierte el checkpoint de PyTorch a ONNX, que es el formato que entiende TensorRT.
Es el único paso que usa Ultralytics, y no hay alternativa: un `.pt` sólo se puede abrir
desde Python.

Se generan dos versiones, una en precisión normal y otra en media precisión:

```bash
sira export --imgsz 256 640 --destino models/model_r.onnx
sira export --imgsz 256 640 --half --destino models/model_r_fp16.onnx
```

**A tener en cuenta.** El tamaño de entrada `256×640` no es arbitrario: el vídeo es muy
apaisado (2560×1024), y con una entrada cuadrada más de la mitad del cómputo se gastaría
en procesar bordes grises. Con la forma rectangular las detecciones son las mismas y va
casi el doble de rápido.

El export desactiva el NMS a propósito, para que el filtrado de detecciones lo haga
nuestro código y se pueda controlar. Y el número de clases se lee del propio modelo, no
de ningún fichero de configuración: en este caso resultó ser **una sola clase**, aunque
el catálogo del repositorio de entrenamiento lista 133.

Detalle en [fase-2](fase-2-export-onnx.md).

---

## 4. Construir los engines

TensorRT compila el ONNX para esta GPU concreta, eligiendo la mejor implementación de
cada operación. El resultado es un `.engine` junto a un fichero de metadatos.

```bash
sira build --onnx models/model_r.onnx      --destino models/model_r_fp32.engine --precision fp32
sira build --onnx models/model_r_fp16.onnx --destino models/model_r_fp16.engine --precision fp16
```

**A tener en cuenta.** Hacen falta dos ONNX distintos porque en TensorRT 11 la precisión
la decide el modelo, no el compilador: ya no existe un flag para pedir FP16.

Un engine **sólo vale para la GPU y la versión de TensorRT con las que se construyó**.
Por eso no se guardan en git y se regeneran en cada máquina; el código lo comprueba al
cargarlos y avisa si no encajan.

Detalle en [fase-3](fase-3-engines-tensorrt.md).

---

## 5. Comprobar que las detecciones son correctas

Antes de fiarse de nada conviene verificar que el pipeline propio —redimensionado,
normalización, decodificación de la salida y filtrado— produce lo mismo que Ultralytics
con el modelo original.

```bash
sira predict --engine models/model_r_fp16.engine \
             --video data/urol_procesado_2min.mp4 --frame 500 --comparar
```

Devuelve las detecciones y una comparación con la referencia. En las pruebas coinciden
con un solapamiento medio del 99,9 %.

**A tener en cuenta.** Este paso importa más de lo que parece: un error en el
preprocesado no da ningún fallo, simplemente devuelve cajas ligeramente desplazadas. Sin
comparar contra la referencia, el benchmark acabaría midiendo el error en vez del modelo.

Detalle en [fase-4](fase-4-runtime-inferencia.md).

---

## 6. Servir el vídeo

Levanta un servidor que lee el vídeo, detecta sobre cada frame, dibuja las cajas y lo
emite al navegador, con una página que muestra el vídeo anotado y las métricas en vivo.

```bash
docker compose -f docker/compose.yaml up -d
docker compose -f docker/compose.yaml logs -f    # para ver cómo va
```

El servicio escucha sólo en local. Para verlo desde otro equipo se abre un túnel:

```bash
ssh -L 8080:localhost:8080 usuario@maquina
# y luego http://localhost:8080 en el navegador
```

**A tener en cuenta.** Si la inferencia se retrasara respecto al vídeo, el servidor
**descarta frames en lugar de acumularlos**. Es deliberado: encolarlos haría que el
directo se fuera quedando atrás sin límite. Con la configuración actual sobra
muchísimo margen —el vídeo va a 30 fps y el sistema aguanta unos 300—, así que en la
práctica no descarta casi nada.

No hay ningún tipo de autenticación, así que no conviene exponerlo en la red tal cual.

Detalle en [fase-5](fase-5-servidor-video.md).

---

## 7. Medir la ganancia

Compara el modelo original de PyTorch contra los engines de TensorRT sobre los mismos
frames, y escribe los resultados en `results/`.

```bash
sira bench --frames 300 --engines model_r_fp32 model_r_fp16
```

Resultados obtenidos:

| | Latencia | Imágenes por segundo |
| --- | --- | --- |
| PyTorch (modelo original) | 7,96 ms | 123 |
| TensorRT, precisión normal | 3,52 ms | 241 |
| **TensorRT, media precisión** | **1,48 ms** | **313** |

Es decir, **más de 5 veces más rápido** en la inferencia, y sin pérdida apreciable de
calidad: las detecciones coinciden con las del modelo original en un 99,7 %.

**A tener en cuenta.** El vídeo no tiene etiquetas, así que no se puede calcular la
precisión real del modelo. Lo que se mide es cuánto se aparta cada versión optimizada de
la de referencia, que es una pregunta distinta y así está etiquetado en los resultados.

Y un matiz importante: la inferencia ya sólo supone el 11 % del tiempo total. Lo que
domina ahora es preparar la imagen y comprimir el vídeo para enviarlo al navegador, ambas
tareas de CPU. Optimizar más el modelo apenas cambiaría nada; si se quisiera ir más
rápido, habría que atacar esas dos.

Detalle en [fase-6](fase-6-benchmark.md).
