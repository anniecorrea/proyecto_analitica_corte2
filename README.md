# Proyecto PENGWIN

Sistema academico para deteccion, segmentacion, clasificacion y medicion de
fracturas pelvicas a partir de tomografia computarizada.

## Regla de organizacion

Todo el proyecto se construye de forma acumulativa en un unico notebook:

```text
proyecto_pengwin.ipynb
```

Cada avance agrega nuevas secciones al mismo archivo. No se crean notebooks ni
carpetas separadas por semana o por entrega.

## Estructura

```text
.
|-- proyecto_pengwin.ipynb
|-- README.md
|-- requirements.txt
|-- IA_USAGE.md
|-- Proyecto_Corte2_PENGWIN.docx
|-- data_mha/
|-- splits/
`-- evidencias/
```

- `data_mha/`: datos pesados locales; no se suben a GitHub.
- `splits/`: particion fija por paciente `75/5/20`, semilla 42.
- `evidencias/`: archivos generados por el notebook, organizados por funcion.

## Datos locales

La carpeta debe estar junto al notebook:

```text
data_mha/
|-- PENGWIN_CT_train_images_part1/
|-- PENGWIN_CT_train_images_part2/
`-- PENGWIN_CT_train_labels/
```

El flujo trabaja directamente con `.mha` mediante SimpleITK. No utiliza NIfTI
ni requiere conversiones intermedias.

## Instalacion

macOS o Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt
```

## Ejecucion

1. Abrir la carpeta del repositorio en Visual Studio Code.
2. Abrir `proyecto_pengwin.ipynb`.
3. Seleccionar el kernel del ambiente `.venv`.
4. Pulsar **Run All**.

El notebook comprueba los 100 pares CT-label, genera los splits, ejecuta el EDA,
aplica ventaneo HU, crea el visualizador MIP y la malla 3D, y prepara los
targets de deteccion con bounding boxes.

## Resultados actuales

- 100 imagenes y 100 labels `.mha` correctamente emparejados.
- Splits por paciente: 75 entrenamiento, 5 validacion y 20 prueba.
- Semilla fija: 42.
- 575 fragmentos anotados.
- Visualizador MIP y malla 3D interactiva.
- 24.234 cortes positivos con targets de deteccion.
- Cero pacientes compartidos entre splits.
