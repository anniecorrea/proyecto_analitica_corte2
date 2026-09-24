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

Para preparar la asesoria con el profesor, revisar:

```text
docs/guia_asesoria_entrega1.md
```

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

## Donde clonar el repositorio

Para evitar problemas de rutas, se recomienda clonar el repositorio en una carpeta de trabajo estable, no en `Downloads` ni dentro de carpetas temporales.

Ejemplos recomendados:

```text
~/Documents/proyectos/proyecto_analitica_corte2
~/Desktop/proyectos/proyecto_analitica_corte2
C:\Users\TU_USUARIO\Documents\proyectos\proyecto_analitica_corte2
```

Despues de clonar, la carpeta `data_mha/` debe quedar dentro de la raiz del repositorio, al mismo nivel de `README.md` y `entrega1.ipynb`.

## Instalacion de dependencias

Desde la raiz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
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
