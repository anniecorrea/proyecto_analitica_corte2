# Datos y targets de deteccion - Semana 9

## Que resuelve esta parte

El modulo transforma cada volumen CT y su mascara experta `.mha` en ejemplos
2D para la cabeza de deteccion. Cada ejemplo contiene:

- corte CT con ventaneo oseo y normalizacion entre 0 y 1;
- tamano `256 x 256` mediante letterbox, sin deformar la anatomia;
- una bounding box por region anatomica presente;
- clase numerica y nombre de cada region;
- mascara semantica de tres clases;
- mascara e IDs de instancia para la etapa de segmentacion;
- `case_id`, numero de corte y spacing fisico del volumen.

## Taxonomia usada

| IDs de la mascara | Target | Clase |
|---|---|---:|
| 1-10 | Sacro | 1 |
| 11-20 | Coxal izquierdo | 2 |
| 21-30 | Coxal derecho | 3 |

Los fragmentos de una misma region se unen para construir una sola caja. Esto
cumple el requisito de Ferro de detectar las regiones anatomicas en cada corte;
los IDs individuales se conservan para la segmentacion de instancias posterior.

Las cajas usan el formato `xyxy`: `[x_min, y_min, x_max, y_max)`. Los limites
maximos son exclusivos, que es la convencion habitual en codigo de deteccion.

## Generar la evidencia

Desde la raiz del repositorio:

```bash
python3 src/data/generar_targets_semana9.py
```

Se crean:

```text
evidence/entrega2/datos_targets/
├── manifest_cortes_positivos.csv
└── targets_bounding_boxes_caso_001.png
```

El manifiesto respeta los splits fijos `75/5/20`: ningun corte de un paciente
puede aparecer en dos particiones diferentes.

## Uso desde el modelo

```python
from src.data.targets_detection import PengwinDetectionDataset, detection_collate

train_dataset = PengwinDetectionDataset(
    split="train",
    image_size=256,
    channels=1,
    as_torch=True,
)

image, target = train_dataset[0]
print(image.shape)       # (1, 256, 256)
print(target["boxes"])  # N x 4
print(target["labels"]) # N, valores 1-3
```

Para un backbone preentrenado que exija RGB, usar `channels=3`; el corte se
replica en los tres canales sin alterar la informacion HU.

## Pruebas

```bash
python3 -m unittest tests/test_targets_detection.py -v
```

Las pruebas verifican taxonomia, cajas consolidadas, ventaneo HU y letterbox.

## Evidencia minima para la asesoria

1. Mostrar el PNG con varias cajas sobre cortes reales.
2. Explicar que la separacion se hace por paciente, no por corte.
3. Mostrar que una region con varios fragmentos produce una caja de deteccion.
4. Entregar a la cabeza de deteccion `boxes` y `labels` con formas variables.
5. Usar un subconjunto pequeno de este dataset para el overfit intencional.
