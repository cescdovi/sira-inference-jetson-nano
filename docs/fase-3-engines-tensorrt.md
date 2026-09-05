# Fase 3 — Construcción de engines TensorRT

## Objetivo

Compilar el ONNX a planes de TensorRT en FP32 y FP16, cacheados y con metadatos que
permitan validar que un engine corresponde al entorno donde se va a cargar.

## Estado

**completada para FP32 y FP16** — 2026-09-05. **INT8 pendiente**, ver [Pendiente](#pendiente).

## El condicionante: TensorRT 11 no tiene flags de precisión

Descubierto en la fase 0 e incorporado aquí. No existen `BuilderFlag.FP16` ni
`BuilderFlag.INT8`, toda red es *strongly typed*, y `IInt8Calibrator` fue eliminado.
**La precisión la determinan los tipos del grafo ONNX, no el builder.**

Consecuencia directa sobre el diseño: no se construyen varias precisiones desde un mismo
ONNX. Cada precisión parte de su propio ONNX, y el parámetro `precision` del builder es
sólo una **etiqueta** para nombrar el artefacto y para que el benchmark sepa qué compara.

## API verificada sobre la instalación

Sondeada antes de escribir código, para no programar contra ejemplos de TensorRT 8.x
o 10.x que no aplican:

```
OnnxParser: True
MemoryPoolType: ['DLA_GLOBAL_DRAM', 'DLA_LOCAL_DRAM', 'DLA_MANAGED_SRAM',
                 'TACTIC_DRAM', 'TACTIC_SHARED_MEMORY', 'WORKSPACE']
build_serialized_network: True
BuilderConfig timing: ['avg_timing_iterations', 'create_timing_cache',
                       'get_timing_cache', 'progress_monitor', 'set_timing_cache']
```

Y la GPU, vía `cuda.bindings.runtime` (no `cuda.cudart`, que no existe en esta versión):

```
gpu: NVIDIA GeForce RTX 5060   cc: 12.0   vram: 7705 MiB
```

## Resultados

| Precisión | ONNX de partida | Engine | Tamaño | Build |
| --- | --- | --- | --- | --- |
| FP32 | `model.onnx` (80,4 MB) | `model_fp32.engine` | **82 588 868 B** | 17,8 s |
| FP16 | `model_fp16.onnx` (40,3 MB) | `model_fp16.engine` | **42 471 596 B** | — |

El engine FP16 ocupa la mitad justa que el FP32, como cabía esperar.

El ONNX en FP16 se generó con `export --half`. Requiere GPU: Ultralytics no puede castear
a FP16 en CPU. Por eso el exportador tiene el dispositivo como parámetro, manteniendo CPU
por defecto para que un problema de soporte de la GPU no se confunda con uno de
exportación.

## torch sí ve la GPU

Comprobado de paso, porque el baseline del benchmark depende de ello y sm_120 es reciente:

```
torch 2.14.0+cu130
cuda disponible: True
arquitecturas compiladas: ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
gpu: NVIDIA GeForce RTX 5060 (12, 0)
matmul en GPU ok: True
```

Riesgo eliminado: el baseline de PyTorch para la fase 6 es viable.

## Decisiones

- **Sidecar `.engine.json` validado al cargar.** Registra versión de TensorRT, GPU y
  compute capability, y `cargar_engine` los compara con el entorno actual antes de
  deserializar. Un plan construido para otra versión u otra GPU falla de formas poco
  claras o, peor, no falla.
- **`ILogger` propio.** Los avisos del parser van al log del proyecto; son la principal
  pista cuando un ONNX no encaja, y descartarlos deja a ciegas.
- **Timing cache compartido** entre construcciones, en `models/timing.cache`.
- **Workspace de 2 GB por defecto**, holgado en una GPU de 8 GB.

## Pendiente

**INT8.** Requiere cuantización explícita: insertar nodos Q/DQ calibrados en el ONNX
*antes* de TensorRT, con `nvidia-modelopt`. No está hecho. Es una dependencia nueva y
trabajo real, y el benchmark sigue siendo válido sin esa fila.
