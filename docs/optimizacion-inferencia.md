# Optimización de la inferencia

Este documento recoge qué se hizo para acelerar la inferencia del modelo de detección de
SIRA, cuánto se ganó, y —quizá lo más útil— qué enseña el desglose de esos números sobre
dónde tiene sentido seguir invirtiendo esfuerzo.

Todas las medidas están tomadas sobre la misma máquina (NVIDIA RTX 5060, 8 GB, driver
580.159.03) y el mismo vídeo (`urol_procesado_2min.mp4`, 2560×1024, 30 fps).

---

## 1. El punto de partida

El pipeline de entrenamiento publica en MLflow un checkpoint de PyTorch. Servirlo tal
cual, a través de Ultralytics, funciona: unos **8 ms por imagen**, alrededor de 123
imágenes por segundo. Para un vídeo a 30 fps eso ya sería suficiente.

La pregunta que motiva este trabajo no es por tanto "¿podemos llegar a tiempo real?",
sino **cuánto margen se gana optimizando**, y a costa de qué en calidad de detección. Ese
margen es lo que permite, más adelante, procesar varios flujos a la vez, subir la
resolución, o mover el sistema a hardware más modesto.

---

## 2. Qué se hizo

### 2.1 Del checkpoint a un formato compilable

Un `.pt` de PyTorch no es un formato de intercambio: es un objeto de Python serializado,
que sólo se puede abrir desde Python y con la librería que lo creó. El primer paso es
convertirlo a **ONNX**, una representación del grafo de la red independiente del
framework.

Dos decisiones de esta conversión condicionan todo lo demás:

- **Se desactiva el NMS interno.** El *non-maximum suppression* —el filtrado que elimina
  detecciones duplicadas— puede incrustarse dentro del grafo exportado. Se optó por
  dejarlo fuera y reimplementarlo, para mantener el control sobre los umbrales y para
  que la comparación contra el modelo original sea limpia.
- **Se fija la forma de entrada.** Sin formas dinámicas el compilador puede especializar
  mucho más agresivamente.

### 2.2 La forma de entrada: la optimización que no cuesta nada

El modelo se entrenó con imágenes cuadradas de 640×640, y ese fue el primer export. Pero
el vídeo real es **2560×1024**, una relación de aspecto de 2,5:1, mucho más apaisada que
lo habitual.

Al encajar una imagen tan apaisada en un cuadrado, la imagen ocupa 640×256 y **el 60 %
restante del tensor es relleno gris**. Es decir: más de la mitad del cómputo se dedicaba
a procesar bordes vacíos.

Exportando a **256×640** el factor de escala es exactamente el mismo (0,25), así que la
imagen se ve con idéntico detalle, pero desaparece el relleno. El resultado son las
mismas detecciones con **1,93× menos tiempo**.

Merece subrayarse porque es una optimización *gratis*: no toca el modelo, no degrada
nada, y sólo requiere haber mirado la resolución real de los datos. Es también un buen
recordatorio de que las suposiciones sobre los datos —"un vídeo será 16:9"— salen caras.

### 2.3 La compilación con TensorRT

TensorRT toma el ONNX y lo compila para una GPU concreta. A diferencia de un intérprete
de grafos, no ejecuta las operaciones una a una tal como están escritas: las **fusiona**
(una convolución, su normalización y su activación pasan a ser un único núcleo de
cómputo), elimina operaciones redundantes, reordena la memoria al formato que mejor va
para esa arquitectura, y para cada capa **prueba varias implementaciones posibles y se
queda con la más rápida** en ese hardware.

El resultado es un *engine*, un artefacto binario ya especializado. Tiene una
contrapartida importante: **sólo sirve para la GPU y la versión de TensorRT con las que
se construyó**. No es portable, no se guarda en el control de versiones, y se regenera en
cada máquina. El código lo comprueba al cargarlo y se niega a usar un engine que no
corresponda al entorno.

Se generaron dos engines, uno en precisión simple (FP32) y otro en media precisión
(FP16), para poder separar la ganancia de la compilación de la ganancia de reducir la
precisión numérica.

> **Particularidad de la versión utilizada.** En TensorRT 11 desaparecieron los flags de
> precisión: ya no se le pide al compilador "hazlo en FP16". Toda red es *fuertemente
> tipada* y la precisión la determinan los tipos del propio grafo ONNX. En la práctica
> esto significa que **cada precisión necesita su propio ONNX**, no basta con recompilar
> el mismo con otra opción. Es un cambio de fondo respecto a casi toda la documentación y
> los ejemplos que circulan, escritos para versiones anteriores.

### 2.4 El pre y el postproceso, y por qué hubo que validarlos

Al dejar el NMS fuera del grafo, hay que implementar por nuestra cuenta todo lo que rodea
a la red: redimensionar la imagen conservando la proporción y rellenar hasta la forma de
entrada, convertir el espacio de color, normalizar, y luego decodificar el tensor de
salida, filtrar por confianza, aplicar NMS y devolver las cajas a las coordenadas del
frame original.

Nada de esto es difícil, pero **todo esto falla en silencio**. Un relleno mal repartido,
un eje transpuesto o un umbral aplicado en el orden equivocado no producen ninguna
excepción: producen cajas de aspecto perfectamente plausible, ligeramente desplazadas. Y
entonces el benchmark mediría el error de implementación en lugar de medir el modelo.

Por eso el criterio de aceptación no fue "funciona" sino "coincide con la referencia". Se
comparó cada detección contra la que produce Ultralytics con el modelo original sobre el
mismo frame. El resultado, sobre 300 frames: **solapamiento medio del 99,9 %** y
diferencias de puntuación por debajo de la milésima.

---

## 3. Cómo se mide, y por qué la metodología no es un detalle

Antes de dar cifras conviene explicar cómo se obtienen, porque en medición de rendimiento
es fácil producir números que parecen buenos y no significan nada.

**Se descartan las primeras iteraciones.** Las primeras ejecuciones incluyen la
compilación de núcleos de cómputo y la subida de los pesos a la GPU. Incluirlas
contamina la mediana con un coste que sólo se paga una vez.

**Los frames se precargan en memoria.** Si se decodifica el vídeo dentro del bucle de
medición, se acaba midiendo el decodificador. En este caso el decodificador cuesta
*menos* que la inferencia, pero podría fácilmente ser al revés.

**Se reportan percentiles, no medias.** La media esconde la variabilidad; en una GPU que
ajusta su frecuencia según la temperatura, el p99 dice cosas que la media no.

**Se separa la inferencia pura del extremo a extremo.** Es la distinción central de todo
este documento y se desarrolla en la sección 5.

---

## 4. Resultados: la inferencia aislada

Latencia de una sola pasada de la red, sin contar nada de lo que la rodea. 300 frames,
entrada 256×640:

| Configuración | Latencia p50 | Latencia p99 | Imágenes/s | VRAM |
| --- | ---: | ---: | ---: | ---: |
| PyTorch (modelo original) | 7,96 ms | 8,25 ms | 123 | 1737 MiB |
| TensorRT FP32 | 3,52 ms | 3,54 ms | 241 | 1585 MiB |
| **TensorRT FP16** | **1,48 ms** | **1,50 ms** | **313** | **1531 MiB** |

### La ganancia, descompuesta

Tener el punto intermedio en FP32 permite atribuir la mejora en lugar de dar un número
agregado que no se sabe interpretar:

| Tramo | Factor | Qué lo produce |
| --- | ---: | --- |
| PyTorch → TensorRT FP32 | **2,26×** | Fusión de operaciones, selección de núcleos, eliminación de sobrecarga del framework |
| TensorRT FP32 → FP16 | **2,38×** | Media precisión: la mitad de ancho de banda de memoria y unidades de cómputo específicas |
| **Total** | **5,38×** | |

Aparte, y previa a todo lo anterior, la forma de entrada aportó **1,93×** por sí sola.

Vale la pena notar que **el p99 casi coincide con el p50** en los engines de TensorRT
(1,48 frente a 1,50 ms), mientras que en PyTorch se separa bastante más. Un engine
compilado no sólo es más rápido: es mucho más **predecible**, porque no hay planificación
dinámica ni lanzamiento de núcleos desde Python en cada pasada. Para un sistema que tiene
que sostener un ritmo de vídeo, la previsibilidad importa tanto como la media.

---

## 5. Resultados: el sistema completo

Aquí es donde los números cambian de significado, y es la parte que más fácilmente se
malinterpreta.

La inferencia no es lo único que ocurre por cada frame. En el servicio real hay que
decodificar el vídeo, preparar la imagen, inferir, decodificar la salida, dibujar las
cajas y comprimir el resultado para enviarlo al navegador. Desglose medido sobre el
servicio en funcionamiento:

| Etapa | Tiempo | Porcentaje |
| --- | ---: | ---: |
| Compresión JPEG | 4,72 ms | **46,1 %** |
| Preproceso | 1,81 ms | 17,7 % |
| **Inferencia** | **1,52 ms** | **14,9 %** |
| Decodificación del vídeo | 1,09 ms | 10,7 % |
| Dibujado de cajas | 0,87 ms | 8,5 % |
| Postproceso | 0,22 ms | 2,2 % |
| **Total** | **10,23 ms** | ~98 imágenes/s |

**La inferencia, después de haberla acelerado 5,38×, es el 15 % del tiempo.** Casi la
mitad se va en comprimir la imagen para mandarla al navegador, y otro tercio largo en
preparar la imagen y decodificar el vídeo. Todo eso es trabajo de CPU sobre frames de
2560×1024, y no se beneficia en absoluto de TensorRT.

De ahí que la mejora *extremo a extremo* sea **2,55×** (de 123 a 313 imágenes por segundo
midiendo preproceso + inferencia + postproceso) y no 5,38×. Y si se cuenta el servicio
completo, con decodificación y compresión incluidas, el techo baja a unas 98 imágenes por
segundo.

### La consecuencia incómoda

Es aritmética elemental, pero conviene decirla explícitamente:

- Si la inferencia pasara a costar **cero**, el sistema pasaría de 98 a 115 imágenes por
  segundo. Una mejora del **17 %**.
- Si la compresión JPEG se hiciera a media resolución —cuatro veces más barata—, el
  sistema pasaría de 98 a 149. Una mejora del **53 %**.

Es decir: **hacer la inferencia infinitamente rápida rinde tres veces menos que un cambio
trivial en cómo se envía la imagen al navegador.** Cualquier esfuerzo adicional en
cuantizar el modelo (INT8, por ejemplo) tendría un retorno marginal en el sistema real.

Esta es exactamente la razón de medir por etapas y no sólo el total. Mirando únicamente
la latencia de inferencia, la conclusión natural habría sido "sigamos optimizando el
modelo", y habría sido la decisión equivocada.

### Un ejemplo de por qué esto no es teórico

Durante el benchmark apareció un resultado imposible: el engine FP16, que infería 2,4
veces más rápido que el FP32, **daba menos imágenes por segundo**. Mirando sólo la
latencia de inferencia, todo parecía correcto.

El desglose por etapas señaló al preproceso, que pasaba de 0,6 a 3,8 ms al cambiar de
precisión. La causa: la biblioteca de cálculo numérico no tiene soporte vectorial nativo
para media precisión, así que dividir entre 255 en FP16 se emulaba elemento a elemento.
Haciendo la aritmética en precisión simple y convirtiendo sólo al final, el rendimiento
pasó de 188 a 313 imágenes por segundo.

Un cuello de botella creado por la propia optimización, invisible en la métrica que se
estaba optimizando.

---

## 6. Y la calidad, ¿qué se pierde?

Un modelo más rápido que detecta peor no es una mejora. La comparación se hace tomando el
engine FP32 como referencia y midiendo cuánto se aparta cada configuración:

| Comparación | Solapamiento medio | Δ puntuación | Detecciones perdidas o añadidas |
| --- | ---: | ---: | ---: |
| PyTorch vs TensorRT FP32 | 0,99945 | 0,00047 | 0 / 1 |
| TensorRT FP16 vs FP32 | 0,99709 | 0,00099 | 0 / 0 |

Dos lecturas:

1. **La compilación no altera el modelo.** PyTorch y el engine FP32 producen
   prácticamente las mismas cajas. Confirma además que el pre y el postproceso
   reimplementados son fieles.
2. **La media precisión sale esencialmente gratis.** No aparece ni desaparece ninguna
   detección en 300 frames, las cajas se desplazan de forma imperceptible y la
   puntuación varía en la tercera cifra decimal. A cambio de 2,38× de velocidad.

> **Limitación importante.** El vídeo no tiene anotaciones, así que **no se puede calcular
> la precisión real del modelo** (mAP u otra métrica absoluta). Lo que se mide es
> *divergencia* respecto a una referencia, que es una pregunta distinta: dice que las
> configuraciones son equivalentes entre sí, no dice si el modelo acierta. Presentar estas
> cifras como si fueran precisión sería incorrecto.

---

## 7. Conclusiones

**La compilación con TensorRT aporta una mejora sustancial y sin coste en calidad.** 5,38×
en la inferencia, con detecciones indistinguibles de las del modelo original. Repartida
casi a partes iguales entre la compilación del grafo (2,26×) y la media precisión (2,38×).

**La mayor ganancia unitaria no vino del compilador.** Vino de ajustar la forma de entrada
a la geometría real del vídeo (1,93×), algo que no requiere ninguna herramienta y sí
mirar los datos.

**El trabajo de optimización del modelo está terminado.** Con la inferencia en el 15 % del
tiempo total, el retorno de seguir por ahí es marginal. Si se quisiera más rendimiento,
por orden de impacto: comprimir a menor resolución para el envío al navegador,
llevar el preproceso a la GPU, y usar decodificación por hardware del vídeo.

**Para el caso de uso actual sobra margen.** El vídeo va a 30 imágenes por segundo y el
sistema sostiene unas 98 con todo incluido, más de tres veces lo necesario. Ese margen es
lo que permitiría procesar varios flujos simultáneos o subir la resolución de trabajo.

**La metodología fue parte del resultado.** Medir por etapas separadas no fue una
formalidad: destapó un cuello de botella introducido por la propia optimización y cambió
la conclusión sobre dónde merece la pena seguir trabajando.
