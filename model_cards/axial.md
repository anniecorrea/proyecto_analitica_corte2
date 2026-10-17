# Model card: modelo axial

> **Uso exclusivamente académico.** Este modelo no es un dispositivo médico, no está validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

## Resumen

| | |
|---|---|
| Archivo | `resultados/modelos/entrenamiento_multitarea_axial/modelo_multitarea_axial.pt` (+ `resultados/modelos/nms_pertenencia/cabeza_pertenencia_axial.pt`) |
| Tarea | En cada corte axial de una tomografía de pelvis: detectar las regiones óseas, segmentar cada fragmento y asignarlo a su hueso (sacro, coxal izquierdo o coxal derecho) |
| Entrada | 3 cortes axiales consecutivos (z−1, z, z+1), ventana ósea en HU con realce Otsu + CLAHE, 256 × 256 px |
| Salida | Cajas por región, máscara de instancia por fragmento y hueso de cada fragmento. La app une los cortes en fragmentos 3D y mide la distancia de cada uno a su fragmento principal |
| Uso en el sistema | Uno de los tres modelos de la app (axial, coronal y sagital); el usuario elige qué plano ver |
| Autores | Equipo del proyecto PENGWIN, Corte 2, 2026-2 |

## Arquitectura

- **Backbone:** FundidoraPC "plus", propio del curso: 4 bloques convolucionales más un bloque de contexto con convoluciones dilatadas. Entrenado desde cero.
- **Atención:** CBAM (canal y espacial) sobre el mapa de características final, con compuerta residual, antes de bifurcarse a las cabezas.
- **Cabeza de detección:** grilla propia de 16 × 16 con NMS por clase implementado por el equipo.
- **Decoder de instancias:** con conexiones desde el backbone; predice la región de cada píxel, el centro de cada fragmento y el desplazamiento de cada píxel hacia su centro.
- **Cabeza de pertenencia:** clasifica cada fragmento en sacro, coxal izquierdo o coxal derecho. Se entrena aparte, con el tronco congelado.
- Unos 2.5 millones de parámetros. No usa YOLO, Detectron2 ni Mask R-CNN.

## Entrenamiento

- **Datos:** dataset PENGWIN (MICCAI 2024), 100 tomografías con máscaras de fragmentos validadas por expertos. Partición fija por paciente: 75 de entrenamiento, 5 de validación y 20 de prueba (semilla 42).
- **Cortes:** 18.072 cortes axiales con pelvis en entrenamiento, más cortes sin pelvis en proporción de 1 por cada 3 con pelvis, para que el modelo no invente fragmentos fuera de ella.
- **Etapas:**
  - A: detección (10 épocas).
  - B: decoder, con el tronco congelado (10 épocas).
  - C: todo junto (10 épocas, tasa de aprendizaje menor).

  Se guardó el mejor checkpoint de cada etapa según validación; el modelo entregado es de la **etapa C, época 5**.
- **Pérdida:** suma ponderada de detección y segmentación (`w_det = w_seg = 1`). Pesos internos de la detección: obj 0.46, box 2.90, cls 2.03, elegidos comparando 4 configuraciones con 3 semillas.
- **Configuración:** AdamW, lote de 16, precisión mixta (`torch.amp`), semilla 42.

## Métricas (test: 20 pacientes no usados para entrenar ni para elegir nada)

| Métrica | Objetivo | Resultado | ¿Cumple? |
|---|---|---|---|
| Clasificación por fragmento (F1 macro) | ≥ 0.85 | 0.886 | Sí |
| Clasificación por fragmento (AUC) | ≥ 0.85 | 1.000 | Sí |
| Detección: IoU promedio | ≥ 0.65 | 0.855 | Sí |
| Detección: mAP@0.50 | ≥ 0.65 | 0.933 | Sí |
| Detección: mAP@[0.50:0.95] | ≥ 0.40 | 0.692 | Sí |
| Segmentación: Dice por fragmento | ≥ 0.85 | 0.793 | **No** |
| Segmentación: IoU por fragmento | ≥ 0.70 | 0.704 | Sí |

- **F1 por hueso:** sacro 0.837, coxal izquierdo 0.916, coxal derecho 0.904.
- **Por región anatómica**, Dice 0.962 e IoU 0.927. Con las mismas cajas como prompt, SAM (ViT-B, sin entrenar) obtiene Dice 0.803 e IoU 0.672.
- **Latencia por imagen:** 28.9 ms en GPU (NVIDIA RTX 5060, precisión mixta) y 102.5 ms en CPU (AMD Ryzen 5 5600XT).

La evaluación se hizo sobre 4.897 cortes con pelvis y 16.041 fragmentos reales.

**Ablación (en este plano, al final de la etapa B):**
- Quitar CBAM empeora todas las métricas en cerca de 0.008.
- Arrancar el backbone con pesos preentrenados en Oxford-IIIT Pet no cambia los resultados de forma significativa.

Los detalles están en `resultados/metricas/comparaciones/`.

## Alcances

- Tomografías de pelvis de adultos con fractura, adquiridas de forma similar a las de PENGWIN.
- Detecta las tres regiones óseas con alta precisión y segmenta bien el hueso completo.
- Un control propio descarta los estudios sin pelvis completa: en validación aceptó los 5 estudios con pelvis y rechazó los 10 recortes sin ella.

## Limitaciones conocidas

- **Fragmentos que se tocan se fusionan.** Cuando un fragmento está en contacto con su fragmento principal en la superficie de fractura, el modelo suele segmentarlos como uno solo. En test, ninguno de los 54 fragmentos conminutos quedó separado de su principal: 52 se fusionaron y 2 no se encontraron. Cambiar el armado 3D no lo corrige; el problema viene de la segmentación 2D.
- **La distancia de separación no es confiable para fragmentos desplazados.** Como consecuencia de lo anterior, un fragmento separado de su hueso puede reportarse con distancia 0 mm. En test, de los 6 fragmentos realmente separados, ninguno se midió bien; el mayor error fue de 23.3 mm.
- **Fragmentos pequeños.** Se encuentra el 57 % de los fragmentos de menos de 100 px en el corte, frente al 90 % de los de 400 a 1599 px. Es la principal razón por la que el Dice por fragmento no llega a 0.85. Influye la resolución de trabajo de 256 × 256 px.
- **Modelo 2D corte a corte.** No usa contexto 3D al segmentar; los fragmentos 3D se reconstruyen uniendo cortes.
- **Datos limitados.** Se entrenó y evaluó solo con PENGWIN: 100 pacientes, 20 de ellos en test. No se probó con otros equipos de tomografía, otros protocolos ni pacientes pediátricos.

## Advertencia

Este modelo y la app que lo usa tienen fines exclusivamente académicos. No son un dispositivo médico, no están validados clínicamente y no deben usarse para diagnosticar, planear cirugías ni tomar ninguna decisión sobre pacientes reales.
