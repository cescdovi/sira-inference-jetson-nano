# Fase 4 — Runtime de inferencia

## Objetivo

Preproceso, ejecución del engine y postproceso propios, **verificados contra las
detecciones de Ultralytics** sobre el `.pt` original.

## Estado

**completada** — 2026-09-06. Verificada sobre frames reales del vídeo.

## Por qué la verificación es el núcleo de esta fase

El pipeline es manual por decisión (CLAUDE.md §1): letterbox, normalización, decodificado
del tensor y NMS son código propio. Un error sutil aquí —media anchura de padding, un eje
transpuesto, un umbral aplicado antes de tiempo— **no lanza ninguna excepción**: produce
cajas de aspecto plausible. Y entonces el benchmark mediría el bug en vez del modelo.

Por eso el criterio de aceptación no es "funciona", sino "coincide con la referencia".

## Resultados

Frame 500 de `urol_procesado_2min.mp4`, comparado contra Ultralytics sobre el `.pt`:

| Engine | Detecciones | Emparejadas | IoU mínimo | Δ score máx | Clases |
| --- | --- | --- | --- | --- | --- |
| **FP32** | 2 vs 2 | 2 | **0,9996** | **0,0000** | coinciden |
| **FP16** | 2 vs 2 | 2 | **0,9965** | **0,0009** | coinciden |

FP32 reproduce la referencia prácticamente de forma exacta. **FP16 degrada de forma
despreciable**: IoU 0,9965 y menos de una milésima de score.

## Incidencia: una comparación injusta que parecía un bug

La primera comparación dio **IoU 0,95** y delta de score 0,006. Suficiente para
sospechar, insuficiente para saber si era un fallo del postproceso o una diferencia
legítima.

La hipótesis se comprobó ejecutando Ultralytics de las dos maneras sobre el mismo frame:

```
ultralytics (por defecto)
   caja [11.8, 164.0, 643.8, 614.8]   score 0.7785
   caja [1285.0, 193.0, 1929.8, 613.3] score 0.7773

ultralytics {'imgsz': 640, 'rect': False}
   caja [14.0, 178.9, 639.6, 611.7]   score 0.7845
   caja [1286.0, 199.4, 1925.0, 604.2] score 0.7779
```

Mi pipeline daba `[1285.96, 199.31, 1925.06, 604.22]` con score `0.7780`: coincide con la
segunda hasta la **décima de píxel**.

La causa: **Ultralytics hace inferencia rectangular por defecto**, ajustando la entrada a
la relación de aspecto del frame, mientras que el ONNX exportado tiene entrada cuadrada
fija de 640×640. No era un fallo del postproceso, sino dos preprocesos distintos y ambos
correctos.

Corregido pasando `rect=False` en la comparación, con el motivo documentado en el código:
sin esa nota, la próxima persona que vea IoU 0,95 volverá a sospechar del postproceso.

## Detalles de implementación que importan

- **Relleno del letterbox repartido a ambos lados.** Si se pusiera todo a un lado, las
  cajas saldrían desplazadas exactamente la mitad del padding.
- **El dtype de entrada se lee del engine.** Un engine FP16 alimentado con `float32`
  produce basura sin lanzar ningún error. `trt.nptype()` sobre el tipo declarado evita
  tener que acordarse.
- **Búferes de device reservados una vez y reutilizados.** En vídeo se ejecuta decenas de
  veces por segundo; reservar y liberar por frame domina el tiempo y fragmenta memoria.
- **NMS por clase, no global.** Dos objetos de clases distintas pueden solaparse
  legítimamente. Hoy `nc=1` y da igual, pero el código es genérico sobre `nc`, que se lee
  de los metadatos.
- **`IExecutionContext` no es seguro para uso concurrente.** La clase no se protege sola;
  está documentado que quien la use desde varios hilos debe serializar o instanciar por
  hilo. Importa para la fase 5.

## Hallazgo con impacto en el rendimiento

El vídeo es **2560×1024**, relación de aspecto 2,5:1.

Con entrada cuadrada de 640×640, la escala es `min(640/1024, 640/2560) = 0,25`, así que el
contenido ocupa 640×256 y el resto —**el 60 % del tensor**— es relleno gris. Se está
gastando más de la mitad del cómputo en procesar padding.

Un ONNX exportado a **640×256** daría exactamente la misma escala (0,25), las mismas
detecciones, y 2,5 veces menos píxeles que procesar. Está planteado al usuario en
[Pendiente](#pendiente).
