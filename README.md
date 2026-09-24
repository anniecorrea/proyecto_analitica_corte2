# Proyecto Analitica Corte 2 PENGWIN

Sistema academico para deteccion, segmentacion, clasificacion y medicion de fracturas pelvicas a partir de tomografia computarizada usando el dataset PENGWIN.

## Entrega 1

Este avance cubre:

- dataset organizado localmente;
- splits fijos de entrenamiento, validacion y prueba;
- carga de archivos medicos `.mha`;
- lectura de spacing fisico del volumen;
- ventaneo y umbral HU para resaltar hueso;
- EDA de fragmentos por caso;
- visualizador 1 raw con evidencia MIP y visualizacion 3D interactiva.

## Estructura del repositorio

```text
.
├── entrega1.ipynb
├── requirements.txt
├── data/
│   └── README.md
├── docs/
│   └── reproducibilidad.md
├── evidence/
│   └── entrega1/
├── splits/
└── src/
    ├── data/
    ├── utils/
    └── visualization/
```

## Datos locales

Los datos pesados no se suben a GitHub. Cada integrante debe tener esta estructura local:

```text
data_mha/
├── PENGWIN_CT_train_images_part1/
├── PENGWIN_CT_train_images_part2/
└── PENGWIN_CT_train_labels/
```

Verificar datos:

```bash
python3 src/data/verificar_dataset_local.py
```

## Visualizador 1

Generar visualizador 3D raw:

```bash
python3 src/visualization/generar_visualizador1_3d_raw.py
```

Generar evidencia MIP complementaria:

```bash
python3 src/visualization/generar_visualizador1_mip.py
```

Las salidas quedan en `evidence/entrega1/`.
