# Proyecto PENGWIN: fracturas pélvicas en tomografia

Este repositorio contiene un sistema académico que analiza tomografías computarizadas de pelvis con fractura, tomadas del dataset PENGWIN. Para cada estudio, el sistema:

1. **Detección:** De tres regiones oseas de la pelvis (sacro, coxal izquierdo y coxal derecho) con una cabeza de detección propiaa basada en una grid y un NMS implementado por nosotros.

2. **Segmentación:** Cada fragmento oseo se clasifica según el hueso al que pertenece con un color caracteristico para cada una de las regiones.

3. **Muestras:** los resultados se presentan en un dashboard de Streamlit con 3 visualizadores: el volumen crudo, laa inferencia entre los cortes y la reconstrucción 3D.

**Aviso:** Proyecto con fines exclusivamente académicos. No es un dispositivo médico, no está validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

# Dataset

Se usa la parte del CT del challenge de PENGWIN (MICCAI 2024). Son 100 tomografias de pelvis con fracturas y sus mascaras de fragmentos, anotada y validadas por ortopedistas. Los datos no se encuentran en el repositorio, ya que estos pesan varias GB, deberas de descargar el dataset en: [PENGWIN](https://zenodo.org/records/10927452) y crear la carpeta data_mha/ en la rama main del repositorio.

# Instalación

Requisitos:

- Python
- Git y Git LFS

**Recomendación:** Tener GPU disponible para entrenar el modelo.

1. **Clonar y descargar los pesos**

´´
git clone https://github.com/anniecorrea/proyecto_analitica_corte2.git
cd proyecto_analitica_corte2
git lfs install
git lfs pull

2. **Crear el ambiente virtual**

**Windows**
´´
python -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
´´

**MacOs / Linux**
´´
python3.13 -m venv .venv
source .venv/bin/activate
´´

3. **Instalar dependencias**

´´
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt
´´


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
| `Negativos_ajuste` | Ajuste fino con cortes sin pelvis (negativos), elegido con `val` | `datos/targets/manifest_cortes_negativos.csv`, `modelos/`, `metricas/` y `figuras/entrenamiento_negativos/` |
| `Entrenamiento_negativos_completo` | Entrenamiento A, B y C desde cero con los negativos | `modelos/`, `metricas/` y `figuras/entrenamiento_negativos_completo/` |
| `Comparacion_final_negativos` | Original vs. ajustado vs. desde cero en 3D y en estudios sin pelvis; aplica la regla | `metricas/comparacion_final_negativos/` |

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

- **Inicio**: titulo, introduccion del proyecto, la advertencia de uso academico y el boton para subir una tomografia
  (`.mha`, `.nii` o `.nii.gz`). La prediccion arranca sola con los tres modelos por plano
  (`modelo_multitarea_axial.pt`, `modelo_multitarea_coronal.pt` y `modelo_multitarea_sagital.pt`) y la configuracion
  3D congelada en `resultados/metricas/salida_3d/config_3d.json`. La imagen y los resultados se conservan al ir al
  dashboard y volver; se borran al quitar el archivo del cargador.
  - **Fragmentos** (al inicio): cuantos fragmentos tiene la imagen (etiqueta de referencia) y cuantos predice el
    modelo del plano elegido. La etiqueta se toma de `data_mha/PENGWIN_CT_train_labels/` si el archivo se llama como
    un caso del dataset.
  - Visualizadores de la consigna, en el plano elegido (axial, coronal o sagital; cada plano usa su modelo):
    1. MIP coronal del volumen crudo con umbral de 300 HU (sin modelo).
    2. Inferencia corte por corte con un slider: cajas del detector, mascaras coloreadas por hueso y la distancia de
       separacion junto a la etiqueta de cada fragmento (o la referencia, si hay etiqueta).
    3. Reconstruccion 3D: marching cubes (skimage) + Plotly Mesh3d, con la distancia en la etiqueta de cada fragmento.

- **Dashboard de resultados**: metricas de val frente a los objetivos, entrenamiento de los tres modelos por plano,
  modelo frente a SAM, fragmentos 3D por caso y latencia en CPU y GPU. Solo lee archivos de `resultados/`.

Las funciones del pipeline no se copian: `app/inferencia.py` las carga por nombre desde los cuadernos, igual que
`salida_3d.ipynb`. El limite de carga se sube a 1 GB en `.streamlit/config.toml`.

Latencia por imagen y por tomografia en GPU y CPU (la lee el dashboard), con los datos en `data_mha/`:

```bash
python -m src.medir_latencia            # escribe resultados/metricas/latencia/latencia.json
```

Despliegue: `deploy/instalar.sh` deja el frontend como servicio en un servidor Ubuntu y lo publica con un tunel de
Cloudflare (`deploy/pengwin-tunel.service`); la direccion `https://*.trycloudflare.com` sale en
`sudo journalctl -u pengwin-tunel | grep trycloudflare`. El tunel de Cloudflare no deja subir archivos de mas de
100 MB; las tomografias mas grandes se suben por la direccion directa del servidor (`http://<IP>:8501`).

## Cortes sin pelvis (negativos)

El entrenamiento original solo uso cortes con pelvis (`manifest_cortes_positivos.csv`). Al recorrer un volumen
completo, los cortes de arriba y de abajo (femures, columna) reciben cajas: con el modelo 2.5D, en `val` 229 de 394
cortes sin pelvis tienen alguna caja y 212 pasan `umbral_corte = 0.08` (el del modelo de 1 canal dejaba pasar 190 con 0.45). La mAP y el Dice no lo muestran porque se
calculan solo sobre cortes positivos.

`notebooks/Negativos_ajuste.ipynb` ajusta el modelo actual mezclando cortes sin un solo voxel de pelvis, compara dos
proporciones y elige con `val` segun una regla escrita antes de entrenar (`src/cortes_vacios.CRITERIO`): menos
cortes vacios que pasan, sin bajar la mAP@0.50 ni el Dice mas de 0.01 y conservando al menos el 99 % de los cortes
con pelvis. Primero se corre con `SMOKE_TEST = True`; el modelo solo se adopta a mano (`ADOPTAR = True`) y despues
hay que repetir `nms_pertenencia_pipeline` y `salida_3d`.

Resultado del ajuste (val): los cortes vacios que pasan bajan de 212 a 172 sin cambio en mAP (0.864) ni Dice
(0.944 -> 0.943); en 3D, con la configuracion congelada, el F1 por fragmento paso de 0.667 a 0.638 (20 fragmentos
predichos frente a 18; reales 27).

Como el sistema se va a probar con estudios que pueden no tener pelvis, el equipo decidio usar negativos y comparar
dos candidatos con una regla fijada antes de entrenar (`src/cortes_vacios.regla_modelo_final`):

1. `notebooks/Entrenamiento_negativos_completo.ipynb`: repite el entrenamiento original (etapas A, B y C, mismos
   hiperparametros) con los 5.698 negativos de train desde la primera epoca. Revisa los topes en 2D.
2. `notebooks/Comparacion_final_negativos.ipynb`: compara original, ajustado y desde cero en 3D, reajustando la
   salida 3D para cada uno, y en los tramos sin pelvis de val tratados como estudios separados. Aplica la regla.

Resultado del entrenamiento desde cero (val, volumenes completos): los cortes vacios que pasan bajan de 212 a 51 (13 %),
con mAP@0.50 0.854 (original 0.864) y Dice 0.941. La tolerancia de mAP de la regla se amplio de 0.01 a 0.02 antes de
la comparacion final (ver `IA_USAGE.md`).

Control de "pelvis presente" (`src/control_pelvis.py`, usado por el frontend): si los cortes con una caja de score
>= 0.2 no forman un bloque continuo de al menos 80 mm, el estudio no tiene una pelvis completa y no se entregan
fragmentos. En val, con pelvis el bloque mide 200 mm o mas; en los tramos sin pelvis, 32 mm como mucho.

Pruebas sin GPU: `python -m unittest tests.test_cortes_vacios tests.test_control_pelvis -v`.

## Evaluacion integrada

Evalua los tres modelos por plano que exporta `notebooks/Entrenamiento_multitarea_planos.ipynb`
(`modelo_multitarea_{axial,coronal,sagital}.pt`). Desde la raiz:

```bash
python -m src.evaluacion --plano todos --split val --device auto
```

La evaluacion corre el pipeline real del sistema (`src.pipeline_pengwin.inferir_cortes`): NMS por clase con
`NMS_CONFIG`, decoder de instancias y cabeza de pertenencia. Cada modelo necesita su cabeza en
`resultados/modelos/nms_pertenencia/cabeza_pertenencia_{plano}.pt` (`cabeza_pertenencia.pt` para `--plano anterior`),
con su `_meta.json`, entrenada sobre ESE checkpoint: si el tronco no coincide, la evaluacion se detiene y pide
reentrenar la etapa D. Las cabezas por plano se entrenan con `python -m src.entrenar_pertenencia_planos --plano todos
--device cuda` (tronco congelado, ~1 min por plano en GPU). La clasificacion por pertenencia se mide por fragmento real (los no emparejados con IoU >= 0.5
cuentan como error) y se reporta junto con la cobertura, el modo con mascaras reales y las reglas base.

Cada modelo se evalua con los mismos datos y el mismo codigo de su entrenamiento: las clases de dataset, la cache de
cortes y `MultiTaskNet` se cargan del cuaderno de planos, y en `val` se reutiliza la cache que construyo el
entrenamiento (orientacion canonica, pixeles cuadrados, entrada 2.5D y, en el sagital, el canal de posicion con la
linea media del centro del campo de vision). El script comprueba que `dataset.csv`, `planos.py`, `orientacion.py` y
los manifiestos sean los mismos que uso el entrenamiento (`datos_iguales_al_entrenamiento` en las metricas); si
cambiaron, reconstruye la cache con la misma funcion. Las metricas de cada plano quedan en
`resultados/metricas/evaluacion/{plano}/`, el resumen en `resumen_planos_{split}.csv` y los ejemplos de
acierto/error en `resultados/figuras/evaluacion/{plano}/`. `--plano axial|coronal|sagital` evalua uno solo y
`--plano anterior` el modelo axial anterior (`modelo_multitarea_final.pt`).

La comparacion con SAM (`--sam-checkpoint ruta\sam_vit_b.pth`) corre solo sobre `test`, usando las cajas predichas
como prompts. Usa `--device cpu` o `--device cuda` para medir cada escenario por separado.
El manifiesto actual contiene cortes positivos, por lo que los resultados se
interpretan sobre esos cortes y no sobre cortes vacios del volumen.
La pertenencia (F1/AUC) se mide por fragmento real; la presencia de region por corte queda como diagnostico.
Dice/IoU del modelo se calculan por fragmento con emparejamiento uno-a-uno dentro de su region (clase de la cabeza). La comparacion
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
