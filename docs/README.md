# Documentación por fases

Un documento por fase. Cada uno registra **lo que se decidió, lo que se ejecutó y lo que
se obtuvo** — no solo el plan. El objetivo es que cualquiera (o yo mismo dentro de tres
semanas) pueda reconstruir por qué el sistema es como es sin releer el código.

## Convención

Cada documento de fase tiene esta estructura:

- **Objetivo** — qué desbloquea esta fase.
- **Estado** — `pendiente` / `en curso` / `completada` / `bloqueada`, con fecha.
- **Runbook** — los comandos exactos, en orden, para reproducirlo desde cero.
- **Resultados** — salidas reales obtenidas, pegadas literalmente. Se rellena **al
  ejecutar**, nunca antes.
- **Incidencias** — lo que falló y cómo se resolvió. Es la parte más valiosa.
- **Decisiones** — lo que se eligió y por qué, si hubo alternativas.

Regla: **los resultados se pegan tal cual salen.** Si un comando falla, se documenta el
fallo, no se limpia. Nada se marca como completado sin salida real que lo respalde.

## Índice

| Fase | Documento | Estado |
| --- | --- | --- |
| 0 | [Preparación del host](fase-0-preparacion-host.md) | **completada** (2026-09-05) |
| 1 | [Cliente del registry MLflow](fase-1-cliente-registry.md) | **completada** (2026-09-05) |
| 2 | [Export a ONNX](fase-2-export-onnx.md) | **completada** (2026-09-05) |
| 3 | [Builder de engines TensorRT](fase-3-engines-tensorrt.md) | **completada** (FP32/FP16); INT8 pendiente |
| 4 | [Runtime de inferencia](fase-4-runtime-inferencia.md) | **completada** (2026-09-06) |
| 5 | [Servidor de vídeo](fase-5-servidor-video.md) | **completada** (2026-09-06) |
| 6 | [Benchmark](fase-6-benchmark.md) | **completada** (2026-09-06) |
| 7 | [Contenerización](fase-7-contenerizacion.md) | **completada** (2026-09-06) |

Las fases y su contenido están definidas en [`../CLAUDE.md`](../CLAUDE.md) §8. Las
especificaciones técnicas (versiones, contratos, esquemas) están en §9 de ese mismo
fichero: **este directorio documenta la ejecución, no la especificación.**
