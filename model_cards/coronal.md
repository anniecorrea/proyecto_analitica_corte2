# Model card: modelo coronal

> **Uso exclusivamente académico.** Este modelo no es un dispositivo médico, no está validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

## Resumen

| | |
|---|---|
| Archivo | `resultados/modelos/entrenamiento_multitarea_coronal/modelo_multitarea_coronal.pt` (+ `resultados/modelos/nms_pertenencia/cabeza_pertenencia_coronal.pt`) |
| Tarea | En cada corte coronal de una tomografía de pelvis: detectar las regiones óseas, segmentar cada fragmento y asignarlo a su hueso (sacro, coxal izquierdo o coxal derecho) |
| Entrada | 3 cortes coronales consecutivos, ventana ósea en HU con realce Otsu + CLAHE, píxeles cuadrados y 256 × 256 px |
| Salida | Cajas por región, máscara de instancia por fragmento y hueso de cada fragmento. La app une los cortes en fragmentos 3D y mide la distancia de cada uno a su fragmento principal |
| Uso en el sistema | Uno de los tres modelos de la app (axial, coronal y sagital); el usuario elige qué plano ver |
| Autores | Equipo del proyecto PENGWIN, Corte 2, 2026-2 |

## Arquitectura

Igual a la del modelo axial:
- **Backbone:** FundidoraPC "plus", desde cero.
- **Atención:** CBAM con compuerta residual.
- **Cabeza de detección:** grilla 16 × 16 con NMS por clase propio.
- **Decoder de instancias:** con conexiones desde el backbone (región, centros y desplazamientos).
- **Cabeza de pertenencia:** entrenada aparte, con el tronco congelado.

Unos 2.5 millones de parámetros. No usa YOLO, Detectron2 ni Mask R-CNN.

## Entrenamiento

- **Datos:** dataset PENGWIN (MICCAI 2024). Partición fija por paciente: 75 / 5 / 20 (semilla 42).
- **Cortes:** 14.545 cortes coronales con pelvis en entrenamiento, más cortes sin pelvis en proporción de 1 por cada 3.
- **Etapas:** A (detección, 10 épocas), B (decoder con el tronco congelado, 10 épocas) y C (todo junto, 10 épocas). El modelo entregado es de la **etapa C, época 1**: el criterio de validación eligió la primera época de esa etapa.
- **Pérdida y configuración:** las mismas del modelo axial (`w_det = w_seg = 1`, λ de detección 0.46 / 2.90 / 2.03, AdamW, lote de 16, precisión mixta, semilla 42).

## Métricas (test: 20 pacientes no usados para entrenar ni para elegir nada)

| Métrica | Objetivo | Resultado | ¿Cumple? |
|---|---|---|---|
| Clasificación por fragmento (F1 macro) | ≥ 0.85 | 0.819 | **No** |
| Clasificación por fragmento (AUC) | ≥ 0.85 | 1.000 | Sí |
| Detección: IoU promedio | ≥ 0.65 | 0.787 | Sí |
| Detección: mAP@0.50 | ≥ 0.65 | 0.866 | Sí |
| Detección: mAP@[0.50:0.95] | ≥ 0.40 | 0.555 | Sí |
| Segmentación: Dice por fragmento | ≥ 0.85 | 0.739 | **No** |
| Segmentación: IoU por fragmento | ≥ 0.70 | 0.636 | **No** |

- **F1 por hueso:** sacro 0.765, coxal izquierdo 0.812, coxal derecho 0.880.
- **Por región anatómica**, Dice 0.953 e IoU 0.910. Con las mismas cajas como prompt, SAM (ViT-B, sin entrenar) obtiene Dice 0.756 e IoU 0.608.
- **Latencia del modelo por corte:** 20.8 ms en GPU (NVIDIA RTX 5060), con lote de 1.

La evaluación se hizo sobre 3.817 cortes con pelvis y 12.843 fragmentos reales.

**Por qué es el plano más débil:**
- **Encuentra menos fragmentos.** El 71 % de los fragmentos reales queda emparejado (IoU ≥ 0.5), frente al 81 % del axial y el sagital. Un fragmento que no se encuentra cuenta como error de clasificación: con las máscaras reales, el F1 de clasificación sube a 0.984.
- **La etapa C lo empeoró.** En test, el modelo al final de la etapa B obtiene F1 0.853 y Dice 0.757, frente a 0.819 y 0.739 del modelo entregado; la diferencia es significativa con un intervalo de confianza del 95 %. **No se cambió el modelo por este resultado**, porque elegirlo con test invalidaría la evaluación; se reporta como hallazgo.

## Alcances

- Tomografías de pelvis de adultos con fractura, adquiridas de forma similar a las de PENGWIN.
- Detección de las tres regiones y segmentación del hueso completo con buena precisión.
- Un control propio descarta los estudios sin pelvis completa: en validación aceptó los 5 estudios con pelvis y rechazó los 10 recortes sin ella.

## Limitaciones conocidas

- **Fragmentos que se tocan se fusionan.** Cuando un fragmento está en contacto con su fragmento principal en la superficie de fractura, el modelo suele segmentarlos como uno solo. En test, ninguno de los 54 fragmentos conminutos quedó separado: 50 se fusionaron y 4 no se encontraron. Cambiar el armado 3D no lo corrige; el problema viene de la segmentación 2D.
- **La distancia de separación no es confiable para fragmentos desplazados.** Un fragmento separado de su hueso puede reportarse con distancia 0 mm. En test, de los 6 fragmentos realmente separados, ninguno se midió bien; el mayor error fue de 23.3 mm.
- **Fragmentos pequeños.** Se encuentra el 51 % de los fragmentos de menos de 100 px, frente al 84 % de los de 400 a 1599 px. Influye la resolución de trabajo de 256 × 256 px.
- **El sacro es el hueso más difícil en este plano** (F1 0.765).
- **Modelo 2D corte a corte** y **datos limitados:** solo PENGWIN, 100 pacientes, 20 de ellos en test. No se probó con otros equipos, protocolos ni pacientes pediátricos.

## Advertencia

Este modelo y la app que lo usa tienen fines exclusivamente académicos. No son un dispositivo médico, no están validados clínicamente y no deben usarse para diagnosticar, planear cirugías ni tomar ninguna decisión sobre pacientes reales.
