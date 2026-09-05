# Fase 2 — Export a ONNX

## Objetivo

Convertir `models/best.pt` a ONNX, y de paso resolver las dos incógnitas que arrastraban
las fases anteriores: cuántas clases tiene el modelo y por qué el checkpoint pesa 40 MB.

## Estado

**completada** — 2026-09-05. Ejecutada en la máquina de inferencia.

## Por qué este paso es Python

Es el único punto del proyecto donde interviene Ultralytics, y no hay alternativa: un
`.pt` es un pickle que serializa el objeto `DetectionModel` con referencias a clases
Python, así que deserializarlo exige el intérprete y el paquete. Además `torch.onnx.export`
no tiene equivalente fuera de Python. A partir del ONNX, todo lo demás es propio.

## Instalación

`ultralytics onnx onnxslim` sobre `~/sira-venv`. Arrastra torch y su stack de CUDA:

```
Successfully installed ... cuda-toolkit-13.0.3.0 ... torch ... torchvision ... ultralytics
```

El venv pasó de ~2 GB a **~9 GB**. Torch llegó con los paquetes `nvidia-*-cu13`, es decir
compilado contra CUDA 13, coherente con el driver 580 de la máquina. Que torch funcione
sobre la GPU (sm_120) **no está verificado todavía**: para exportar se usa `device="cpu"`
a propósito, y la comprobación de GPU se hará cuando haga falta para el baseline del
benchmark.

## Hallazgos: dos supuestos del proyecto eran falsos

### 1. El modelo no es un YOLO11n

```json
{
  "nc": 1,
  "task": "detect",
  "parametros": 20053779,
  "imgsz_entrenamiento": 640,
  "bytes_checkpoint": 40480620
}
```

**20 053 779 parámetros.** Un YOLO11n tiene ~2,6 M; esta cifra coincide con un
YOLO11**m** (~20,1 M). El modelo está registrado como `sira-yolo11n-General` pero contiene
algo casi 8 veces mayor. En el registry existe además un `sira-yolo11m-General` sin ningún
alias.

Encaja con el tamaño del checkpoint: 40 MB para 20 M parámetros son ~2 bytes por
parámetro, es decir pesos guardados en FP16. El ONNX exportado en FP32 ocupa 80 MB, justo
el doble, lo que confirma la cuenta.

**No es un problema de este repositorio** —el código lee todo del artefacto y funciona
igual—, pero cambia lo que cabe esperar en latencia y VRAM, y conviene aclarar con el
equipo de entrenamiento si el nombre registrado está mal o si se cambió el tamaño del
modelo sin actualizar la plantilla.

### 2. Es un detector de una sola clase

```json
"names": { "0": "1_Fenestratedb bipolar forceps" }
```

El catálogo de `data_general.yaml` lista 133 entradas, y la conjetura era que
`keep_classes: [1]` conservaba el grupo `1_` (instrumental, ~21 clases). **Es más
literal:** conserva el índice 1 del catálogo, una única clase.

Esto valida la decisión de §9.2 de leer `nc` y `names` del artefacto en vez de deducirlos
del YAML. Deducirlos habría dado 133 canales esperados frente a los 5 reales, y el
postproceso habría interpretado mal el tensor **sin lanzar ningún error**: cajas
plausibles con la clase equivocada.

## Resultado del export

```json
{
  "onnx": "model.onnx",
  "onnx_sha256": "032c538235d0d3f35814a9912aa8a7e81d1b328b394483e3ac78d712ff30b65c",
  "onnx_bytes": 80428503,
  "imgsz": 640,
  "opset": 18,
  "nms_incrustado": false,
  "dinamico": false,
  "nc": 1,
  "entrada": { "nombre": "images", "forma": [1, 3, 640, 640], "dtype": "FLOAT" },
  "salida":  { "nombre": "output0", "forma": [1, 5, 8400], "dtype": "FLOAT" },
  "origen_pt_sha256": "fe77008ec059d9fd8a64f2365eb12561324008b484aa637e93a771ce8866e09c"
}
```

La salida `[1, 5, 8400]` es exactamente `[1, 4+nc, 8400]` con `nc=1`: el grafo confirma
de forma independiente lo que decía el checkpoint.

## Decisiones

- **`nms=False`.** Si el NMS va incrustado en el grafo, el postproceso deja de ser
  nuestro: se pierde el control sobre los umbrales y la comparación con el baseline deja
  de ser limpia. El NMS se implementa en la fase 4.
- **`dynamic=False`.** Forma de entrada fija. Simplifica el engine y el runtime, y el
  servidor siempre alimenta frames del mismo tamaño tras el letterbox.
- **`device="cpu"` para exportar.** El export no necesita GPU, y así no se mezcla un
  posible problema de soporte sm_120 en torch con un problema de export.
- **Opset por defecto (18).** Al contrario que con la Jetson y TensorRT 8.2, aquí no hace
  falta forzar un opset antiguo.
- **Persistir `nc` y `names` en `model.onnx.json`.** El runtime los lee de ahí en vez de
  llevarlos escritos, para que un reentrenamiento con otro número de clases no obligue a
  tocar código.

## Salvaguarda añadida

El exportador compara los canales de salida del ONNX con `4 + nc` del checkpoint y avisa
si no cuadran. Es justo el fallo que no da error y produce detecciones con la clase
cambiada, así que merece una comprobación explícita en vez de confiar en que coincidan.

## Pendiente

- Aclarar con el equipo de entrenamiento la discrepancia entre el nombre registrado
  (`yolo11n`) y el tamaño real del modelo (~20 M, propio de un `11m`).
- Verificar que torch ve la GPU (sm_120). No hace falta hasta el benchmark, pero conviene
  saberlo antes de depender de ello.
