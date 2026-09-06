# Fase 6 — Benchmark

## Objetivo

Cuantificar la ganancia de TensorRT frente al modelo tal como salió de entrenamiento, y
**poder atribuirla**: cuánto viene de la compilación del grafo y cuánto de la precisión
reducida.

## Estado

**completada** — 2026-09-06. INT8 fuera de alcance por decisión del usuario.

## Metodología

- **300 frames precargados en RAM.** Decodificar dentro del bucle habría medido el
  decodificador, que en este vídeo cuesta más que la inferencia.
- **50 iteraciones de calentamiento descartadas**: las primeras incluyen compilación de
  kernels y subida de pesos.
- **Percentiles, no medias.**
- **Latencia de inferencia pura separada de la de extremo a extremo.**
- **Al baseline de PyTorch se le pasa la misma forma de entrada y `rect=False`.** Sin eso
  se compararían dos preprocesos distintos en vez de dos runtimes.
- **Sin etiquetas no hay mAP.** Lo que se reporta es divergencia respecto al engine FP32,
  y está etiquetado como tal en el JSON de salida.

Vídeo: `urol_procesado_2min.mp4`, 2560×1024, 30 fps. Umbrales conf 0,45 / IoU 0,5.

## Resultados

| Configuración | Entrada | Infer p50 | Infer p99 | e2e p50 | FPS | VRAM (MiB) | det/frame |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch `.pt` | 256×640 | 7,96 | 8,25 | 7,96 | 122,7 | 1737 | 1,763 |
| TensorRT FP32 | 256×640 | 3,52 | 3,54 | 4,11 | 241,2 | 1585 | 1,760 |
| **TensorRT FP16** | 256×640 | **1,48** | **1,50** | **3,18** | **312,8** | 1531 | 1,760 |
| TensorRT FP32 | 640×640 | 6,79 | 6,89 | 7,72 | 129,2 | 1655 | 1,633 |
| TensorRT FP16 | 640×640 | 2,50 | 2,54 | 6,17 | 161,8 | 1567 | 1,627 |

*(PyTorch no expone la inferencia pura por separado: su `predict` incluye pre y
postproceso, así que ambas columnas coinciden. La comparación honesta con TensorRT es la
de extremo a extremo.)*

### Atribución de la ganancia

Sobre entrada rectangular, que es la configuración de despliegue:

| Tramo | Factor | Qué aporta |
| --- | --- | --- |
| PyTorch → TensorRT FP32 | **2,26×** | Fusión de capas y autotuning de kernels |
| TensorRT FP32 → FP16 | **2,38×** | Precisión reducida |
| **Total, inferencia** | **5,38×** | |
| **Total, extremo a extremo** | **2,55×** | 122,7 → 312,8 fps |

El salto extremo a extremo es menor que el de inferencia pura porque el pre y el
postproceso, que corren en CPU, no se benefician de nada de esto. Es la misma conclusión
de la fase 5 vista desde otro ángulo.

Como eje adicional, la **forma de entrada** aporta 1,93× por sí sola (FP32 cuadrado 6,79 ms
→ rectangular 3,52 ms) sin tocar el modelo.

### Divergencia

| Configuración | IoU medio | IoU mínimo | Δ score medio | Sin pareja |
| --- | --- | --- | --- | --- |
| PyTorch vs TRT FP32 | **0,99945** | 0,98262 | 0,00047 | 0 / 1 |
| TRT FP16 vs TRT FP32 | **0,99709** | 0,88715 | 0,00099 | 0 / 0 |
| TRT cuadrado vs rectangular | 0,95308 | 0,65863 | 0,04728 | 40 / 2 |

Dos lecturas:

1. **El pipeline propio es fiel.** PyTorch y el engine FP32 coinciden con IoU medio
   0,99945 sobre 300 frames. El preproceso, el decodificado del tensor y el NMS
   reimplementados reproducen la referencia.
2. **FP16 es prácticamente gratis en calidad.** IoU medio 0,997, ninguna detección
   aparece o desaparece, y el score se desplaza una milésima. A cambio de 2,38× de
   velocidad.

La tercera fila compara entrada cuadrada contra rectangular y **no mide degradación de
precisión**: son dos preprocesos distintos, con distinta rejilla de anclajes y distinto
contexto en los bordes. Se incluye para dejar constancia de que la forma de entrada
cambia las detecciones, no sólo la velocidad.

## Incidencia: el benchmark encontró un bug de rendimiento

La primera ejecución dio un resultado imposible: **FP16 infería 2,4× más rápido que FP32
pero daba menos fps**.

```
model_r_fp32   infer p50 3.53   e2e p50 4.13   FPS 239.7
model_r_fp16   infer p50 1.49   e2e p50 5.29   FPS 187.6
```

El preproceso pasaba de 0,6 ms a 3,8 ms al cambiar de precisión. La causa: `numpy` no
tiene SIMD nativo para `float16`, así que dividir entre 255 en media precisión se emula
elemento a elemento.

Corregido haciendo la aritmética **siempre en float32** y casteando sólo al final. Tras el
arreglo, FP16 pasó de 187,6 a **312,8 fps**, y el engine cuadrado FP16 de 88,3 a 161,8.

Es exactamente el tipo de fallo que motiva medir las etapas por separado: mirando sólo la
latencia de inferencia, el pipeline parecía correcto.

## Conclusiones

- **TensorRT FP16 sobre entrada rectangular es la configuración de despliegue**: 1,48 ms
  de inferencia, 312,8 fps extremo a extremo, y divergencia despreciable frente a FP32.
- Con un vídeo a 30 fps, el margen es de **más de 10×** sobre lo necesario para tiempo
  real.
- El cuello de botella ya no es el modelo, sino el pre y el postproceso en CPU (fase 5).
  Cualquier optimización adicional debería ir ahí, no a cuantizar más.
- INT8 se descartó por decisión del usuario. A la vista de estos números habría aportado
  poco: la inferencia ya es una fracción menor del tiempo total.
