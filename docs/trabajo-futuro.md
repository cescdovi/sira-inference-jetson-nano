# Trabajo futuro

Este documento no propone mejoras para el prototipo de inferencia, sino para
**`camma-laparoscopy`**, el repositorio que entrena, evalúa y promociona los modelos. Ese
es el sitio donde las mejoras tienen efecto duradero: aquí se optimiza un artefacto, allí
se decide qué artefacto existe.

Las propuestas están ancladas a sus propias especificaciones (`src/*/*.SPEC.md`) y, en
varios casos, cierran o desbloquean decisiones que esas specs dejan explícitamente
abiertas.

---

## 1. Dónde encaja este prototipo en su arquitectura

Conviene situarlo, porque cambia lo que hay que hacer después.

Según `model_serving.SPEC` §1 y §3, el servicio de inferencia **no optimiza**: carga el
*artefacto servido* que la promoción ya publicó. Y `model_optimization.SPEC` §2 define el
pipeline que lo produce: cuantización → ONNX → backend (TensorRT en pruebas, OpenVINO en
producción). Coherente con eso, `src/serving/champion.py` ya resuelve el champion buscando
por prioridad `*.engine`, `*_openvino_model`, `*.onnx` y sólo al final `best.pt`.

**Hoy el registry sólo contiene `best.pt`.** El propio docstring de `champion.py` lo
reconoce. Es decir: el prototipo de inferencia ha tenido que hacer la exportación y la
compilación por su cuenta, que es trabajo que la arquitectura asigna a
`model_optimization`.

Casi todo lo que sigue consiste en devolver ese trabajo a su sitio, ahora que existe una
implementación validada de la que partir.

---

## 2. Publicar el artefacto servido, no sólo el checkpoint

**Qué falta.** Que el paso de promoción exporte a ONNX y registre ese artefacto —junto a
la *receta* de exportación— como parte del modelo registrado.

`model_optimization.SPEC` §8 ya resuelve la parte difícil de esta decisión: el engine de
TensorRT está atado a GPU, driver y versión, **no es portable**, así que lo que se versiona
es la receta y no el binario. El ONNX sí es portable y sí debería viajar en el registry.

**Qué aporta este prototipo.** La receta concreta, ya validada: `imgsz` explícito,
`nms=False`, `dynamic=False`, opset por defecto, y los metadatos que hacen falta
después (`nc`, `names`, formas de entrada y salida, hash del checkpoint de origen). Con
eso, el nodo de inferencia dejaría de necesitar PyTorch ni Ultralytics: bastaría descargar
el ONNX y compilarlo.

Esto tiene un efecto colateral grande: la imagen de despliegue pasaría de ~19 GB a unos
pocos, porque `torch` y `ultralytics` sólo hacen falta para exportar.

---

## 3. Cerrar la decisión de cuantización, que ya no está bloqueada

`model_optimization.SPEC` §4 deja abierto el nivel de cuantización con este razonamiento:
INT8 «requiere set de calibración (que no existe)», luego FP16 es lo viable.

**El bloqueo, tal como está redactado, ya no se sostiene del todo.** La calibración INT8
**no necesita etiquetas**, sólo datos representativos del dominio, y hoy existe vídeo
quirúrgico real. Lo que sigue faltando no es material de calibración, sino datos
**etiquetados** para *verificar* que la cuantización no degrada — que es un requisito
distinto y conviene enunciarlo así.

**Pero la recomendación práctica es la misma, y ahora con evidencia:** quedarse en FP16.
Los motivos, medidos:

- FP16 sale esencialmente gratis en calidad: solapamiento medio 0,997 frente a FP32, sin
  perder ni añadir una sola detección en 300 frames.
- INT8 aportaría poco al sistema real: **la inferencia ya es el 15 % del tiempo total**
  (ver [optimizacion-inferencia](optimizacion-inferencia.md) §5). Aunque fuese instantánea,
  el rendimiento subiría un 17 %.
- Y añadiría trabajo real: TensorRT 11 eliminó la cuantización implícita, así que INT8
  exige insertar nodos de cuantización en el ONNX con una herramienta aparte, y el
  procedimiento **difiere entre TensorRT y OpenVINO**, como la propia spec advierte.

La propuesta es **cerrar el TODO con una decisión motivada** en lugar de dejarlo abierto:
FP16, con los números que lo justifican y la condición que obligaría a revisarlo (que la
inferencia vuelva a dominar el tiempo, por ejemplo con modelos mayores o hardware más
modesto).

---

## 4. Fijar `tol_map` y adoptar el acuerdo de detecciones como gate técnico

`model_optimization.SPEC` §5 define dos métricas de fidelidad: `ΔmAP` como primaria —con
`tol_map` marcado como TODO— y, como secundaria, «fracción de cajas con IoU ≥ umbral y
clase coincidente entre base y servido», justificada porque «detecta exports rotos que el
mAP agregado podría enmascarar».

**Esa métrica secundaria está implementada y validada** en el prototipo
(`src/sira/bench/runner.py`, `comparar_detecciones`). Y da una referencia empírica de qué
aspecto tiene una conversión sana:

| Comparación | IoU medio | Δ score medio | Detecciones perdidas/añadidas |
| --- | ---: | ---: | ---: |
| PyTorch → TensorRT FP32 | 0,99945 | 0,00047 | 0 / 1 |
| FP32 → FP16 | 0,99709 | 0,00099 | 0 / 0 |

Propuesta concreta: **hacer del acuerdo de detecciones el gate técnico de fidelidad**, con
un umbral del tipo «≥ 99 % de las cajas emparejadas con IoU ≥ 0,5 y misma clase», y dejar
`ΔmAP` como métrica informativa hasta que haya datos suficientes para anclar `tol_map` al
ruido. Tiene dos ventajas sobre esperar a `tol_map`: no necesita un conjunto grande para
ser estable, y detecta el fallo que realmente ocurre —un export roto— en lugar de una
degradación difusa.

---

## 5. Quitar la arquitectura del nombre del modelo registrado

`mlflow_params.yaml` usa `registered_model_name_template: "sira-yolo11n-{specialty}"`.
Pero el champion actual de `General` tiene **20 millones de parámetros**, propios de un
YOLO11**m**, no de un 11n. El nombre miente.

No es sólo cosmético. Meter la arquitectura en el nombre del modelo registrado obliga a
elegir entre dos males cuando se cambia de tamaño de modelo: dejar el nombre mintiendo, o
registrar bajo otro nombre y **perder el historial de alias** (`champion`, `previous`,
`challenger`) y con él la trazabilidad de qué se sirvió cuándo.

Además, su propia `model_serving.SPEC` §2 escribe el identificador **sin** la
arquitectura: `models:/sira-<especialidad>@champion`. Hay una incoherencia entre la spec
y la configuración.

Propuesta: alinear la plantilla con la spec (`sira-{specialty}`) y registrar la
arquitectura como *tag* o parámetro del run, que es donde la información de "con qué se
entrenó" pertenece y donde no rompe nada al cambiar.

---

## 6. Llevar los datos etiquetados al nodo donde se construye el artefacto

`promotion_gates.SPEC` §2 establece el principio de que **«se evalúa lo que se sirve»**: los
gates se aplican sobre el artefacto post-optimización, no sobre el checkpoint FP32.

Eso tiene una implicación logística que conviene hacer explícita: como el engine **no es
portable**, sólo se puede evaluar en el nodo donde se construye. Por tanto **`frozen_test` y
`rolling_gate` tienen que ser accesibles desde ese nodo**.

En el prototipo esto se notó como una limitación real: sin datos etiquetados en la máquina
de inferencia no se pudo calcular ningún mAP, y toda la validación de calidad tuvo que
reducirse a divergencia respecto a una referencia. Es suficiente para verificar que una
conversión no está rota, pero **no** para ejecutar Gate 1, que exige `mAP50 ≥ 0,5`.

Propuesta: replicar en el nodo de optimización el acceso al remoto DVC/MinIO que ya usa el
pipeline de entrenamiento, aunque sea sólo para los dos conjuntos de gate.

---

## 7. Incluir la forma de entrada en la receta, y evaluar con ella

Esta es, probablemente, la propuesta con más recorrido.

La mayor ganancia unitaria del prototipo —**1,93×**, más que la que aporta la media
precisión— no vino del compilador, sino de ajustar la forma de entrada a la geometría real
del vídeo: 256×640 en lugar de 640×640, porque el material es 2560×1024 y en un cuadrado
el 60 % del tensor era relleno gris.

Eso significa que **la forma de entrada es una decisión de despliegue con impacto de primer
orden**, y hoy no aparece en ninguna parte de la receta de exportación.

Y va más allá, por coherencia con su propio principio de que se evalúa lo que se sirve: si
el artefacto se va a servir a 256×640, **los gates deberían puntuarlo a 256×640**. Evaluar
a 640×640 un modelo que se servirá con otra geometría mide una configuración que nadie
usa. En las mediciones, cambiar la forma de entrada altera las detecciones de forma
apreciable (solapamiento 0,95 entre ambas), así que no es un detalle despreciable.

Propuesta: que `imgsz` sea un parámetro explícito de la receta, ligado a la geometría del
flujo de destino, y que la evaluación de promoción use ese mismo valor.

---

## 8. Preparar el cutover a OpenVINO con una advertencia

`model_optimization.SPEC` §2 y `model_serving.SPEC` §3 dejan claro que **producción es
OpenVINO sobre Intel**, y que TensorRT es el entorno de prueba. Todo el trabajo del
prototipo vive, por tanto, en la rama de pruebas.

**Qué se puede reutilizar tal cual**, porque es independiente del backend: la receta de
exportación a ONNX, el pre y el postproceso, la comparación de divergencia, la metodología
de medición y el desglose de tiempos por etapa. Lo único específico de TensorRT es el
compilador de engines.

Propuesta técnica: poner el backend detrás de una interfaz, de modo que el mismo
pre/postproceso y el mismo benchmark sirvan para ambos. Así la comparación TensorRT vs
OpenVINO sería directa, sobre el mismo grafo y el mismo código de entrada y salida, en vez
de comparar dos implementaciones distintas.

**Y una advertencia que merece llegar antes del cutover.** El desglose por etapas
([optimizacion-inferencia](optimizacion-inferencia.md) §5) muestra que el 85 % del tiempo
por frame es trabajo de CPU: decodificar el vídeo, preparar la imagen, dibujar y comprimir.
En un nodo Intel, esas etapas **no mejoran, y previsiblemente empeoran**, mientras que la
inferencia pierde el acelerador. El margen extremo a extremo puede encogerse mucho más de
lo que sugiere comparar sólo latencias de inferencia entre backends.

La recomendación es medir el nodo Intel **con el pipeline completo desde el principio**, no
sólo la pasada de red, y contemplar desde ya las mitigaciones de la sección siguiente.

---

## 9. Optimizaciones de servicio, que es donde queda margen

Para el prototipo estas quedaron fuera de alcance, pero son las que de verdad moverían la
aguja, y ganan importancia si producción es CPU:

1. **Comprimir a menor resolución para el envío.** Es el 46 % del tiempo por frame. El
   cliente no necesita 2560×1024. Reducirlo a la mitad de lado abarataría esa etapa cuatro
   veces: del orden de un 53 % más de rendimiento, tres veces más que hacer la inferencia
   instantánea.
2. **Preproceso en el acelerador.** Redimensionado, conversión de color y normalización se
   hacen hoy en CPU sobre el frame completo.
3. **Decodificación por hardware** del vídeo en lugar de la ruta genérica de OpenCV.

---

## 10. Aspectos ya abiertos en sus specs que el prototipo confirma

Tres TODOs de `model_serving.SPEC` §11–12 sobre los que el prototipo aporta algo:

- **Multi-especialidad en un proceso.** El prototipo carga un engine al arrancar y lo
  mantiene. Como el contexto de ejecución de TensorRT no admite uso concurrente, servir
  varias especialidades en un proceso exigiría un contexto por especialidad y serializar el
  acceso, o un proceso por especialidad. Con 8 GB de VRAM y un modelo de 42 MB, la memoria
  no sería el límite.
- **Hot-reload frente a redespliegue.** Con contenedores el redespliegue es barato y
  evita todo el estado a medio camino; parece la opción sensata frente a un endpoint de
  recarga.
- **Autenticación y política de red.** El prototipo escucha sólo en loopback y se accede
  por túnel. Para un uso real hace falta decidirlo de verdad: hoy no hay ningún control.

Y uno de `promotion_gates.SPEC` §3: el gate por clase, que hoy coincide con el global
porque sólo existe la clase *forceps bipolar*. El postproceso del prototipo ya es genérico
sobre el número de clases, y la métrica de divergencia debería reportarse **por clase**
cuando lleguen más, por el mismo motivo que ellos exigen el gate por clase: una clase que
se hunde puede quedar oculta en la media.
