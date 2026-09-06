# Fase 0 — Preparación del host

## Objetivo

Dejar la máquina de inferencia capaz de ejecutar TensorRT sobre la RTX 5060, y
**verificar la compatibilidad CUDA/driver** de la que cuelga el resto del stack.

Bloquea todas las fases siguientes.

## Estado

**completada** — 2026-09-05.

El plan original (contenedores NGC) resultó **inviable** por falta de permisos
administrativos. Se ha pivotado a instalación en espacio de usuario mediante venv. Ver
[Decisiones](#decisiones).

## Estado de partida verificado

Inventario por ssh el 2026-09-05:

| Comprobación | Resultado |
| --- | --- |
| GPU | RTX 5060, compute capability 12.0 (sm_120) |
| Driver | 580.159.03, expone CUDA 13.0 |
| CUDA del sistema | 13.0.88 en `/usr/local/cuda-13.0` |
| SELinux | `Disabled` — no será fuente de fricción |
| Grupos del usuario | `francesc docker` |
| **sudo** | **`Sorry, user francesc may not run sudo on fedora`** |
| **Docker engine** | **no instalado** — solo el cliente |
| Docker Desktop | `docker-desktop-4.56.0`, inactivo |
| Podman | presente (`/usr/bin/podman`) |
| Python | 3.11, 3.13 y 3.14; `venv` funcional en 3.11 y 3.13 |
| Acceso a PyPI | HTTP 200 |

## Los dos bloqueos que tumbaron el plan de contenedores

**1. El usuario no tiene sudo.** `sudo -v` responde
`Sorry, user francesc may not run sudo on fedora`. Sin permisos administrativos no se
puede instalar `nvidia-container-toolkit`, que es lo que permite a un contenedor ver la
GPU. No hay rodeo: el toolkit instala componentes de sistema.

**2. No hay Docker Engine, solo Docker Desktop.** Los paquetes instalados son
`docker-ce-cli`, `docker-compose-plugin`, `docker-buildx-plugin` y `docker-desktop`,
pero **no `docker-ce`**. `systemctl is-enabled docker` devuelve `not-found` y el socket
`/var/run/docker.sock` no existe:

```
--- daemon system ---
inactive
not-found
--- socket ---
ls: no se puede acceder a '/var/run/docker.sock': No existe el fichero o el directorio
```

Es decir: el cliente de Docker está, pero no hay demonio nativo contra el que hablar.
Docker Desktop en Linux ejecuta el motor dentro de una máquina virtual, y el paso de GPU
a contenedores no está soportado en esa configuración. Instalar el engine nativo requiere
root, con lo que volvemos al bloqueo 1.

Podman está instalado y funciona sin root, pero no rescata el plan: para exponer la GPU
necesita las especificaciones CDI que genera `nvidia-ctk`, y ese sigue requiriendo
instalar el toolkit como root.

## El pivote: TensorRT en espacio de usuario

Los contenedores servían para dos cosas: evitar que Fedora no esté soportado por NVIDIA,
y no tener que emparejar a mano TensorRT con CUDA y cuDNN. **Los wheels de PyPI resuelven
las dos igual de bien, y sin root.**

Comprobado contra PyPI desde la propia máquina:

```
latest: 11.2.1.2
requires: ['tensorrt_cu13==11.2.1.2']
```

Es **exactamente la misma versión de TensorRT que trae el contenedor NGC 26.08**, con
variante para CUDA 13. Los wheels son manylinux, así que la distribución del host deja de
importar, y arrastran sus propias dependencias de CUDA como wheels de NVIDIA.

Lo único que sigue necesitando root es el driver de la GPU — y ya está instalado.

**Restricción a respetar:** los wheels de TensorRT llegan hasta **Python 3.13**. El
`python3` por defecto de la máquina es 3.14, que no vale. El venv se crea explícitamente
con `python3.13`.

## Runbook

### 1. Crear el venv

```bash
python3.13 -m venv ~/sira-venv
~/sira-venv/bin/python -V
~/sira-venv/bin/python -m pip --version
```

Ejecutado. Resultado:

```
Python 3.13.13
pip 25.1.1 from /home/francesc/sira-venv/lib64/python3.13/site-packages/pip (python 3.13)
```

### 2. Instalar TensorRT

```bash
~/sira-venv/bin/python -m pip install --upgrade pip wheel
~/sira-venv/bin/python -m pip install "tensorrt==11.2.1.2"
~/sira-venv/bin/python -m pip install numpy cuda-python
```

Ejecutado. Resultado:

```
Successfully built tensorrt tensorrt_cu13 tensorrt_cu13_libs
Installing collected packages: tensorrt_cu13_libs, tensorrt_cu13_bindings, tensorrt_cu13, tensorrt
Successfully installed tensorrt-11.2.1.2 tensorrt_cu13-11.2.1.2 \
  tensorrt_cu13_bindings-11.2.1.2 tensorrt_cu13_libs-11.2.1.2
```

### 3. Verificar que TensorRT ve la GPU

Que `import tensorrt` funcione **no prueba nada**: el import no toca la GPU y pasaría
igual con un driver incompatible. Se verificó construyendo y deserializando un engine
real (una red mínima con una capa unaria sobre `[1,3,8,8]`).

Ejecutado. Resultado:

```
TensorRT: 11.2.1.2
builder OK
plan bytes: 12028
engine: OK | ctx: OK
```

Y con tensores de entrada en FP16:

```
plan FP16 bytes: 12036
```

## Resultados

**La fase 0 se da por superada.** Queda demostrado, con salida real, que:

- TensorRT 11.2.1.2 se instala y funciona en espacio de usuario, sin root.
- **CUDA 13.x funciona sobre el driver 580.159.03**, que expone CUDA 13.0. Ésta era la
  suposición pendiente de la que colgaba todo el stack, y queda confirmada: no basta con
  que importe la librería, es que se construye, se serializa y se deserializa un engine,
  y se crea un contexto de ejecución sobre la GPU.
- Se pueden construir engines con tensores en FP32 y en FP16.

Entorno resultante: `~/sira-venv` (Python 3.13.13) con `tensorrt 11.2.1.2`, `numpy` y
`cuda-python`.

## Hallazgo que invalida parte de la especificación

Al introspeccionar la API aparecieron dos cosas que obligan a rehacer el diseño de la
etapa `build` y del benchmark. **TensorRT 11 no se controla como TensorRT 8 ni como 10.**

Salida literal de la introspección:

```
BuilderFlag disponibles: ['DEBUG', 'DIRECT_IO', 'DISABLE_COMPILATION_CACHE',
 'DISABLE_TIMING_CACHE', 'DISTRIBUTIVE_INDEPENDENCE', 'EDITABLE_TIMING_CACHE',
 'ERROR_ON_TIMING_CACHE_MISS', 'EXCLUDE_LEAN_RUNTIME', 'GPU_FALLBACK', 'MONITOR_MEMORY',
 'REFIT', 'REFIT_IDENTICAL', 'REFIT_INDIVIDUAL', 'SAFETY_SCOPE', 'SPARSE_WEIGHTS',
 'STRICT_NANS', 'STRIP_PLAN', 'TF32', 'VERSION_COMPATIBLE', 'WEIGHT_STREAMING']

NetworkDefinitionCreationFlag: ['PREFER_AOT_PYTHON_PLUGINS', 'PREFER_JIT_PYTHON_PLUGINS',
 'STRONGLY_TYPED']

calibradores INT8 legacy:
   IInt8Calibrator False
   IInt8EntropyCalibrator2 False
   IInt8MinMaxCalibrator False

atributos de plataforma (platform_has_fast_fp16, etc.): []
```

**1. No existen flags de precisión.** No hay `BuilderFlag.FP16` ni `BuilderFlag.INT8`. La
documentación de NVIDIA lo confirma: TensorRT 11.0 eliminó *todos* los flags de precisión
(`FP16`, `INT8`, `BF16`, `FP8`, `INT4`, `FP4`) y toda red es **strongly typed** por
defecto. La precisión ya no la decide el builder: la decide el **grafo ONNX** con los
tipos de sus tensores.

**2. La cuantización implícita ha desaparecido.** `IInt8Calibrator` y todas sus subclases
han sido eliminadas. INT8 exige ahora **cuantización explícita**: nodos QuantizeLinear /
DequantizeLinear insertados en el ONNX *antes* de llegar a TensorRT.

También desaparecieron `ITensor.setType`, `ITensor.setDynamicRange`, `ILayer.setPrecision`
y `ILayer.setOutputType`, y `EXPLICIT_BATCH` ya no es un flag porque el batch explícito es
el único modo.

### Consecuencia práctica

El plan de "construir tres engines desde el mismo ONNX cambiando un flag" **no funciona**.
El camino real pasa por preparar un ONNX distinto para cada precisión:

| Precisión | Cómo se obtiene ahora |
| --- | --- |
| FP32 | ONNX tal cual → build directo |
| FP16 | Convertir el ONNX a precisión mixta (ModelOpt AutoCast) → build |
| INT8 | Cuantizar el ONNX insertando nodos Q/DQ, con calibración → build |

La calibración se traslada del builder de TensorRT a la herramienta de cuantización. Lo
que **no** cambia es de dónde salen los datos de calibración: siguen siendo frames del
`.mp4`, y sigue siendo obligatorio usar tramos disjuntos para calibrar y para evaluar.

Esto añade una dependencia nueva (`nvidia-modelopt`) y trabajo real a las fases 4 y 7.
**Hay una decisión abierta**: quedarse en TensorRT 11 con cuantización explícita, o bajar
a TensorRT 10.x, donde siguen existiendo los flags de precisión y los calibradores
clásicos, con mucho más material disponible. Está sin resolver.

## Decisiones

- **Espacio de usuario en vez de contenedores.** Forzado por la ausencia de sudo, pero el
  resultado no es peor: se obtiene la misma versión de TensorRT (11.2.1.2), los wheels
  manylinux hacen irrelevante que Fedora no esté soportado, y desaparecen el toolkit de
  contenedores y la limitación de GPU de Docker Desktop. También simplifica el ciclo de
  desarrollo: no hay imagen que reconstruir en cada cambio.

  **Reversible:** si en algún momento hay acceso root, el camino de contenedores sigue
  siendo válido y la versión de TensorRT sería la misma, así que nada del código de arriba
  cambiaría.

- **Python 3.13, no 3.14.** Impuesto por los wheels de TensorRT.

- **Verificar con un `trt.Builder`, no con un `import`.** El import no toca la GPU y daría
  una falsa confirmación justo de lo que esta fase existe para comprobar. Se fue más allá
  y se construyó un engine completo, que es lo que reveló los cambios de API de arriba:
  una verificación superficial los habría dejado para la fase 4, cuando ya hubiera código
  escrito sobre una API inexistente.
