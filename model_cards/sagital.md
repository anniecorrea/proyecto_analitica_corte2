# Model card: modelo sagital

> **Uso exclusivamente académico.** Este modelo no es un dispositivo médico, no está validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

## Resumen

| | |
|---|---|
| Archivo | `resultados/modelos/entrenamiento_multitarea_sagital/modelo_multitarea_sagital.pt` (+ `resultados/modelos/nms_pertenencia/cabeza_pertenencia_sagital.pt`) |
| Tarea | En cada corte sagital de una tomografía de pelvis: detectar las regiones óseas, segmentar cada fragmento y asignarlo a su hueso (sacro, coxal izquierdo o coxal derecho) |
| Entrada | 3 cortes sagitales consecutivos, ventana ósea en HU con realce Otsu + CLAHE, píxeles cuadrados y 256 × 256 px, más **un cuarto canal con la posición lateral del corte** |
| Salida | Cajas por región, máscara de instancia por fragmento y hueso de cada fragmento. La app une los cortes en fragmentos 3D y mide la distancia de cada uno a su fragmento principal |
| Uso en el sistema | Uno de los tres modelos de la app (axial, coronal y sagital); el usuario elige qué plano ver |
| Autores | Equipo del proyecto PENGWIN, Corte 2, 2026-2 |

## Arquitectura

Igual a la del modelo axial, con una diferencia en la entrada:
- **Backbone:** FundidoraPC "plus", desde cero.
- **Atención:** CBAM con compuerta residual.
- **Cabeza de detección:** grilla 16 × 16 con NMS por clase propio.
- **Decoder de instancias:** con conexiones desde el backbone (región, centros y desplazamientos).
- **Cabeza de pertenencia:** entrenada aparte, con el tronco congelado.

**Canal de posición:** en un corte sagital, el coxal izquierdo y el derecho se ven casi iguales. Para distinguirlos, el modelo recibe un canal constante con la distancia del corte a la línea media del paciente (en mm, con signo). En la app, la línea media se estima como el centro del campo de visión de la tomografía, sin usar la anotación.

Unos 2.5 millones de parámetros. No usa YOLO, Detectron2 ni Mask R-CNN.

## Entrenamiento

- **Datos:** dataset PENGWIN (MICCAI 2024). Partición fija por paciente: 75 / 5 / 20 (semilla 42).
- **Cortes:** 25.882 cortes sagitales con pelvis en entrenamiento, más cortes sin pelvis en proporción de 1 por cada 3.
- **Canal de posición:** durante el entrenamiento, la línea media lleva ruido y a veces se reemplaza por el centro del campo de visión, para que el modelo tolere la estimación imprecisa que usa la app.
- **Etapas:** A (detección, 10 épocas), B (decoder con el tronco congelado, 10 épocas) y C (todo junto, 10 épocas). El modelo entregado es de la **etapa C, época 8**.
- **Pérdida y configuración:** las mismas del modelo axial (`w_det = w_seg = 1`, λ de detección 0.46 / 2.90 / 2.03, AdamW, lote de 16, precisión mixta, semilla 42).

## Métricas (test: 20 pacientes no usados para entrenar ni para elegir nada)

| Métrica | Objetivo | Resultado | ¿Cumple? |
|---|---|---|---|
| Clasificación por fragmento (F1 macro) | ≥ 0.85 | 0.888 | Sí |
| Clasificación por fragmento (AUC) | ≥ 0.85 | 1.000 | Sí |
| Detección: IoU promedio | ≥ 0.65 | 0.852 | Sí |
| Detección: mAP@0.50 | ≥ 0.65 | 0.918 | Sí |
| Detección: mAP@[0.50:0.95] | ≥ 0.40 | 0.711 | Sí |
| Segmentación: Dice por fragmento | ≥ 0.85 | 0.790 | **No** |
| Segmentación: IoU por fragmento | ≥ 0.70 | 0.715 | Sí |

- **F1 por hueso:** sacro 0.903, coxal izquierdo 0.867, coxal derecho 0.895.
- **Por región anatómica**, Dice 0.954 e IoU 0.912. Con las mismas cajas como prompt, SAM (ViT-B, sin entrenar) obtiene Dice 0.806 e IoU 0.674.
- **Latencia del modelo por corte:** 20.5 ms en GPU (NVIDIA RTX 5060), con lote de 1.

La evaluación se hizo sobre 6.658 cortes con pelvis y 11.721 fragmentos reales.

En este plano, la etapa C mejora levemente a la B (F1 +0.009 en test).

## Alcances

- Tomografías de pelvis de adultos con fractura, adquiridas de forma similar a las de PENGWIN.
- Es el plano con mejor F1 de clasificación y mejor mAP@[0.50:0.95].
- Un control propio descarta los estudios sin pelvis completa: en validación aceptó los 5 estudios con pelvis y rechazó los 10 recortes sin ella.

## Limitaciones conocidas

- **Fragmentos que se tocan se fusionan.** Cuando un fragmento está en contacto con su fragmento principal en la superficie de fractura, el modelo suele segmentarlos como uno solo. En test, ninguno de los 54 fragmentos conminutos quedó separado: 43 se fusionaron y 11 no se encontraron. Cambiar el armado 3D no lo corrige; el problema viene de la segmentación 2D.
- **La distancia de separación no es confiable para fragmentos desplazados.** Un fragmento separado de su hueso puede reportarse con distancia 0 mm. En test, de los 6 fragmentos realmente separados, ninguno se midió bien; el mayor error fue de 23.3 mm.
- **Fragmentos pequeños.** Se encuentra solo el 43 % de los fragmentos de menos de 100 px, frente al 90 % de los de 400 a 1599 px. Es el plano con más dificultad en este punto.
- **Depende de la estimación de la línea media.** Si la tomografía no está centrada en el paciente, el canal de posición pierde exactitud y puede confundir el lado izquierdo con el derecho.
- **Modelo 2D corte a corte** y **datos limitados:** solo PENGWIN, 100 pacientes, 20 de ellos en test. No se probó con otros equipos, protocolos ni pacientes pediátricos.

## Advertencia

Este modelo y la app que lo usa tienen fines exclusivamente académicos. No son un dispositivo médico, no están validados clínicamente y no deben usarse para diagnosticar, planear cirugías ni tomar ninguna decisión sobre pacientes reales.
