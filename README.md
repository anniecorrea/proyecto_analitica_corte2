# Proyecto PENGWIN

Sistema academico para deteccion, segmentacion, clasificacion y medicion de
fracturas pelvicas a partir de tomografia computarizada.

## Regla de organizacion

El pipeline se divide en notebooks dentro de `notebooks/`, uno por etapa. Cada
notebook se ejecuta por separado: lee los archivos que genera el anterior
(`splits/dataset.csv`, el manifiesto de cortes positivos, etc.) en lugar de
depender de lo que haya en memoria, y carga por nombre las clases y funciones
compartidas desde el notebook donde viven
(`src.definiciones_notebooks.cargar_definiciones`).

El codigo reutilizable y los scripts que se ejecutan fuera de los notebooks
viven en `src/`. Todo lo que se genera va a `resultados/`, ordenado por tipo de
archivo; `src/rutas.py` define esas carpetas en un solo lugar.

Orden de ejecucion (la tabla va de arriba hacia abajo):

| Notebook | Contenido | Genera en `resultados/` |
|---|---|---|
| `Dataset_splits` | Verificacion CT-label, dataset curado, splits 75/5/20 | `splits/` (fuera de `resultados/`) |
| `Ventaneo_hu` | Ventaneo HU y realce Otsu + CLAHE | — |
| `EDA` | Fragmentos por region, volumenes, fragmento principal | `datos/fragmentos/` |
| `Distancias_referencia` | Distancia 3D de cada fragmento al principal | `datos/distancias_referencia.csv` |
| `Visualizacion_3d` | MIP en tres vistas y malla 3D interactiva | `html/` |
| `Targets_deteccion` | Cajas por region, manifiesto, `PengwinDetectionDataset` | `datos/targets/`, `figuras/targets/`, `modelos/batch_fijo.pt` |
| `Backbone` | Backbone FundidoraPC (`base` y `plus`) y sus pruebas | — |
| `cbam` | CBAM y compuerta residual | — |
| `Cabeza_grid` | Cabeza de deteccion grid y perdidas | — |
| `Segmentacion_instancias` | Decoder con skips, centros/offsets, metricas, SAM | — |
| `Entrenamiento_overfiting` | Prueba de memorizacion (semana 9) | `modelos/` y `metricas/entrenamiento_overfiting/` |
| `Entrenamiento_multitarea` | Entrenamiento por etapas A/B/C | `modelos/`, `metricas/` y `figuras/entrenamiento_multitarea/` |
| `nms_pertenencia_pipeline` | NMS por clase, cabeza de pertenencia, pipeline completo | `modelos/`, `metricas/` y `figuras/nms_pertenencia/` |

## Estructura

```text
.
|-- notebooks/            # un notebook por etapa
|-- src/                  # modulos del pipeline, evaluacion.py y rutas.py
|-- tests/
|-- resultados/           # todo lo generado, por tipo de archivo
|   |-- datos/            # CSV que leen otros notebooks
|   |-- metricas/         # historiales de entrenamiento y tablas de resultados
|   |-- figuras/          # PNG, una subcarpeta por etapa
|   |-- html/             # visualizadores interactivos
|   |-- modelos/          # pesos .pth/.pt (se suben con Git LFS)
|   `-- cache/            # cache del entrenamiento (no se sube a GitHub)
|-- splits/
|-- data_mha/
|-- README.md
|-- requirements.txt
|-- IA_USAGE.md
`-- Proyecto_Corte2_PENGWIN.docx
```

- `data_mha/`: datos pesados locales; no se suben a GitHub.
- `splits/`: particion fija por paciente `75/5/20`, semilla 42.
- Dentro de `metricas/`, `figuras/` y `modelos/` cada etapa usa el mismo nombre
  de subcarpeta: lo de `nms_pertenencia` esta en las tres bajo ese nombre.

## Datos locales

La carpeta debe estar en la raiz del repositorio:

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
2. Abrir los notebooks de `notebooks/` en el orden de la tabla de arriba.
3. Seleccionar el kernel del ambiente `.venv`.
4. Pulsar **Run All** en cada uno.

En un repositorio recien clonado, `splits/` y los CSV de `resultados/` ya
vienen versionados, asi que cualquier notebook puede ejecutarse sin correr los
anteriores. Los pesos de `resultados/modelos/` tambien vienen en el repositorio,
pero por Git LFS (son archivos grandes): instalar Git LFS una vez y descargarlos.

```bash
git lfs install
git lfs pull
```

Sin ese paso, los `.pth`/`.pt` quedan como punteros de texto de unos pocos bytes y
`torch.load` falla. No se suben los `last_*.pth` (solo sirven para reanudar un
entrenamiento) ni la cache de `resultados/cache/`.

Para regenerar las capturas de revisión de distancia desde la raíz del
repositorio, después de tener los datos locales instalados:

```bash
python -m src.generar_evidencias_distancias
```

Las capturas son proyecciones 2D para inspección; el valor de distancia se
calcula en 3D con el espaciado físico del archivo MHA.

## Frontend (Streamlit)

Desde la raiz del repositorio, con el `.venv` activo:

```bash
streamlit run app/streamlit_app.py
```

- **Prediccion de fragmentos**: se sube una tomografia `.mha` (o se elige un caso de `data_mha/`) y el modelo
  `modelo_multitarea_final.pt` marca los fragmentos con la configuracion 3D congelada en
  `resultados/metricas/salida_3d/config_3d.json`. Muestra cortes axial/coronal, la proyeccion MIP, la tabla de
  fragmentos y permite descargar la mascara `.mha` y el CSV.
- **Dashboards**: en construccion.

Las funciones del pipeline no se copian: `app/inferencia.py` las carga por nombre desde los cuadernos, igual que
`salida_3d.ipynb`. El limite de carga se sube a 1 GB en `.streamlit/config.toml`.

## Evaluacion integrada

Cuando la etapa `C_conjunto` haya terminado, exporta `modelo_multitarea_final.pt`
desde `notebooks/Entrenamiento_multitarea.ipynb` y ejecuta desde la raiz:

```bash
python -m src.evaluacion --nms-function src.deteccion_grid:nms_por_clase --sam-checkpoint ruta\sam_vit_b.pth --device auto
```

La comparacion con SAM corre solo sobre `test`, usando las cajas predichas como
prompts. El checkpoint SAM es opcional; sin el, se evalua el modelo propio y el
reporte deja constancia de que SAM no se ejecuto. Las metricas y los resultados
por corte quedan en `resultados/metricas/evaluacion/`, y los ejemplos de
acierto/error en `resultados/figuras/evaluacion/`.
Usa `--device cpu` o `--device cuda` para medir cada escenario por separado.
El manifiesto actual contiene cortes positivos, por lo que los resultados se
interpretan sobre esos cortes y no sobre cortes vacios del volumen.
F1/AUC miden presencia de cada region por corte; Dice/IoU del modelo se calculan
por fragmento con emparejamiento uno-a-uno dentro de su region. La comparacion
SAM se reporta por region, usando las mismas cajas predichas como prompts.

## Resultados actuales

- 100 imagenes y 100 labels `.mha` correctamente emparejados.
- Splits por paciente: 75 entrenamiento, 5 validacion y 20 prueba.
- Semilla fija: 42.
- 575 fragmentos anotados.
- Distancias de referencia para 275 fragmentos conminutos, comparados con el
  fragmento principal de su región; los 300 principales se incluyen en la tabla
  con distancia cero por definición.
- Visualizador MIP y malla 3D interactiva.
- 24.234 cortes positivos con targets de deteccion.
- Cero pacientes compartidos entre splits.
