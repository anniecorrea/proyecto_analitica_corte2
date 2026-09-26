# Proyecto Analitica Corte 2 PENGWIN

Sistema academico para deteccion, segmentacion, clasificacion y medicion de fracturas pelvicas a partir de tomografia computarizada usando el dataset PENGWIN.

## Entrega 1

Este avance cubre:

- dataset organizado localmente;
- splits fijos de entrenamiento, validacion y prueba con proporcion 75/5/20;
- carga de archivos medicos `.mha`;
- lectura de spacing fisico del volumen;
- ventaneo, histogramas HU y umbral para resaltar hueso;
- EDA de fragmentos por caso;
- visualizador 1 raw con evidencia MIP, visualizacion 3D por puntos y visualizacion 3D mejorada por malla.

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

Regenerar splits fijos recomendados en asesoria:

```bash
python3 src/data/generar_splits_75_5_20.py
```

Los splits quedan en:

```text
train: 75 casos
val: 5 casos
test: 20 casos
```

La particion usa semilla fija y estratifica de forma aproximada por complejidad del caso, usando el numero de fragmentos de la mascara.

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
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Si aparece un error como `numpy.dtype size changed`, el ambiente quedo mezclado o corrupto. En Windows, cerrar VS Code, abrir una terminal en la raiz del repo y recrear el ambiente:

```bash
rmdir /s /q .venv
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt
```

## Visualizador 1

Generar visualizador limpio por malla usando labels de referencia:

```bash
python3 src/visualization/generar_visualizador1_mesh_labels.py
```

Generar visualizador raw por malla usando CT y umbral HU dentro de la ROI de pelvis:

```bash
python3 src/visualization/generar_visualizador1_mesh_raw.py
```

Generar visualizador 3D raw:

```bash
python3 src/visualization/generar_visualizador1_3d_raw.py
```

Generar evidencia MIP complementaria:

```bash
python3 src/visualization/generar_visualizador1_mip.py
```

Las salidas quedan en `evidence/entrega1/`.

## Ventaneo HU

Generar evidencia para justificar el ventaneo con histogramas y comparacion visual:

```bash
python3 src/data/generar_evidencia_ventaneo_hu.py
```

Las salidas quedan en:

```text
evidence/entrega1/ventaneo_hu/
```

## EDA

Generar tablas, graficas y resumen del analisis exploratorio:

```bash
python3 src/data/generar_eda_entrega1.py
```

Las salidas quedan en:

```text
evidence/entrega1/eda/
```
