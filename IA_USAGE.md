# Uso de IA

## Integración y evaluación PENGWIN

- **Prompt:** Auditar `evaluacion.py` contra las interfaces reales de los notebooks y `deteccion_grid.py`; verificar clases, checkpoint y dependencias; corregir únicamente el evaluador y probarlo, sin inventar artefactos ni umbrales.
- **Sugerencias de IA:** Permitir la carga segura del objeto `TorchVersion`, exigir que los umbrales estén guardados en el checkpoint, validar el mapeo de clases y calcular una máscara de scores por detección para filtrar las cajas de segmentación.
- **Cambios aplicados:** Se verificaron las firmas y convenciones en el código fuente, se corrigió el filtro de cajas que fallaba al indexar tensores y se añadieron validaciones de rutas, clases y metadatos obligatorios del checkpoint. El NMS usado en pruebas fue `deteccion_grid.nms_por_clase`.
- **Análisis crítico:** Las pruebas sintéticas confirmaron el recorrido de evaluación y la escritura de métricas, tabla, ejemplos y latencias en CPU/CUDA; esos valores no describen el modelo real. El checkpoint C disponible carga y ejecuta el pipeline, pero carece de `inference_defaults`, por lo que no se deben inferir resultados reales hasta regenerar/exportar el checkpoint con esos metadatos. No se encontró checkpoint ni paquete SAM; esa comparación queda pendiente.
