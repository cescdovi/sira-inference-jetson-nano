# Fase 5 — Servidor de vídeo

## Objetivo

Servir el vídeo anotado en tiempo real por HTTP, con una interfaz web mínima y métricas
por etapa.

## Estado

**completada** — 2026-09-06. Verificada en la máquina de inferencia.

## Diseño

**Un hilo de captura e inferencia, una ranura de tamaño uno.** El motor lee frames,
infiere y publica el último resultado; los clientes HTTP leen esa ranura.

**Se descartan frames, nunca se encolan.** Es la decisión central del servidor. Si la
inferencia va más lenta que el vídeo y se encolara, la latencia crecería sin techo hasta
que el stream fuera minutos por detrás de la realidad. Con una ranura de uno, un cliente
lento sólo pierde frames intermedios y siempre ve lo más reciente.

**Contador de frame publicado.** El generador MJPEG bloquea hasta que hay un frame más
nuevo que el último que envió. Sin eso, un cliente rápido reenviaría imágenes idénticas
quemando CPU.

**Transporte MJPEG** (`multipart/x-mixed-replace`): funciona en un `<img>` sin
JavaScript, no necesita negociación y se depura con `curl`. WebRTC daría menos latencia,
pero para un prototipo la simplicidad gana.

**UI en HTML/CSS/JS estático**, sin build ni npm. Consulta `/health` cada segundo y pinta
KPIs, tiempos por etapa y contadores de frames.

## Resultados

Servidor sobre `model_r_fp16.engine` y `urol_procesado_2min.mp4` (2560×1024, 30 fps):

```json
{
  "corriendo": true,
  "fps_video": 30.0,
  "fps": 30.03,
  "frames_leidos": 340,
  "frames_procesados": 339,
  "frames_descartados": 1
}
```

Mantiene los 30 fps del vídeo con un único frame descartado. Endpoints verificados:
`/` `200`, `/static/*` `200`, `/health`, `/metrics`, `/detections`, y `/stream` emitiendo
multipart válido:

```
--sirastream
Content-Type: image/jpeg
Content-Length: 286152
```

## El hallazgo: TensorRT no es el cuello de botella

Tiempos por etapa, p50 sobre ventana deslizante:

| Etapa | p50 (ms) | p95 (ms) | % del total |
| --- | --- | --- | --- |
| decode | 1,668 | 1,857 | 12 % |
| **preproceso** | **4,525** | 5,035 | **33 %** |
| inferencia | 1,528 | 1,553 | 11 % |
| postproceso | 0,257 | 0,325 | 2 % |
| anotado | 0,919 | 1,104 | 7 % |
| **encode JPEG** | **4,715** | 5,020 | **35 %** |

Total ≈ 13,6 ms por frame, techo teórico ~73 fps.

**La inferencia con TensorRT es el 11 % del pipeline.** Los dos tercios los ocupan el
preproceso y la codificación JPEG, ambos trabajo de CPU sobre frames de 2560×1024.

Esto es exactamente lo que la especificación advertía que ocurriría si no se medían las
etapas por separado: se habría atribuido a TensorRT un techo que impone la CPU. Y tiene
una consecuencia incómoda para el proyecto: **seguir optimizando el modelo apenas movería
la aguja**. Bajar la inferencia de 1,53 ms a, digamos, 1,0 ms con INT8 mejoraría el total
un 4 %.

Las palancas reales, por orden de impacto:

1. **Codificar el JPEG a menor resolución.** El navegador no necesita 2560×1024; a 1280×512
   el coste caería a la cuarta parte. Es la mejora más barata que existe aquí.
2. **Preproceso en GPU.** El `resize`, la conversión de color y la normalización se hacen
   hoy en CPU con OpenCV sobre el frame completo.
3. **Decodificación por hardware (NVDEC)** en lugar de `cv2.VideoCapture`.

Ninguna se ha implementado: quedan documentadas como el siguiente paso natural si el
objetivo pasa a ser el rendimiento del servicio y no el del modelo.

## Nota sobre concurrencia

El `IExecutionContext` de TensorRT no es seguro para uso concurrente. El diseño lo evita
por construcción: hay **un solo hilo** que infiere, y los clientes HTTP nunca tocan el
contexto, sólo leen la ranura de JPEG. Si en el futuro se quisieran varios streams
simultáneos con engines distintos, haría falta un contexto por hilo.

## Seguridad

Escucha en `127.0.0.1` por defecto; se accede desde el Mac por túnel ssh:

```bash
ssh -L 8080:localhost:8080 francesc@192.168.1.44
```

Si se le pasa otra interfaz, el servidor lo avisa en el log: no hay autenticación.
