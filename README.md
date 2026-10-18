# Proyecto PENGWIN: fracturas pélvicas en tomografía

Este repositorio contiene un sistema académico que analiza tomografías computarizadas de pelvis con fractura, tomadas del dataset PENGWIN. Para cada estudio, el sistema:

1. **Detección:** localiza las tres regiones óseas de la pelvis (sacro, coxal izquierdo y coxal derecho) con una cabeza de detección propia, basada en una grilla, y un NMS implementado por nosotros.

2. **Segmentación y clasificación:** separa cada fragmento óseo y lo clasifica según el hueso al que pertenece, con un color característico para cada región.

3. **Medición:** calcula en milímetros la distancia de cada fragmento suelto (conminuto) al fragmento principal de su hueso.

4. **Visualización:** muestra los resultados en una app de Streamlit con tres visualizadores: el volumen crudo, la inferencia corte por corte y la reconstrucción 3D.

**Aviso:** proyecto con fines exclusivamente académicos. No es un dispositivo médico, no está validado clínicamente y no debe usarse para apoyar decisiones quirúrgicas reales.

# Dataset

Se usa la parte de CT del challenge PENGWIN (MICCAI 2024). Son 100 tomografías de pelvis con fractura y sus máscaras de fragmentos, anotadas y validadas por ortopedistas. Los datos no están en el repositorio porque pesan varios GB: descárgalos de [PENGWIN en Zenodo](https://zenodo.org/records/10927452) y déjalos en una carpeta `data_mha/` en la raíz del repositorio, así:

```text
data_mha/
|-- PENGWIN_CT_train_images_part1/   # 001.mha ... 050.mha
|-- PENGWIN_CT_train_images_part2/   # 051.mha ... 100.mha
`-- PENGWIN_CT_train_labels/         # 001.mha ... 100.mha (máscaras)
```

En las máscaras, cada vóxel guarda el número de su fragmento: 1–10 sacro, 11–20 coxal izquierdo, 21–30 coxal derecho y 0 fondo.

> Sin `data_mha/` igual puedes correr las pruebas, ver el dashboard de resultados y analizar en la app una tomografía propia. Los datos hacen falta para correr los notebooks y los scripts.

# Instalación

Requisitos:

- Python 3.12 o 3.13
- Git y Git LFS

**Recomendación:** tener GPU disponible para entrenar el modelo. Para usar la app y revisar los resultados basta con la CPU.

1. **Clonar y descargar los pesos**

```bash
git clone https://github.com/anniecorrea/proyecto_analitica_corte2.git
cd proyecto_analitica_corte2
git lfs install
git lfs pull
```

Sin `git lfs pull`, los modelos quedan como archivos de texto de unos pocos bytes y la app no los puede cargar.

2. **Crear el ambiente virtual**

**Windows**
```
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
```

Si PowerShell no deja activar el ambiente, ejecuta una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**macOS / Linux**
```
python3.13 -m venv .venv
source .venv/bin/activate
```

3. **Instalar dependencias**

```
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt
```

4. **Comprobar la instalación**

```
python -c "import torch, SimpleITK; print('torch', torch.__version__, '| GPU:', torch.cuda.is_available())"
```

`GPU: False` es normal en un computador sin tarjeta NVIDIA: la app y las pruebas funcionan igual.

# Paso a paso

Todos los comandos se ejecutan desde la raíz del repositorio, con el ambiente activo.

## Paso 1 Ver los resultados

En la misma app, abre **Dashboard de resultados**. Lee lo que ya está en `resultados/` y no calcula nada:

| Pestaña | Qué muestra |
|---|---|
| Resumen | Métricas en test frente a los objetivos del proyecto y matriz de confusión |
| Entrenamiento | Pérdida de entrenamiento de los tres modelos |
| Modelo vs SAM | Nuestro modelo frente a SAM usando las mismas cajas |
| Ablación | Efecto de quitar CBAM y de usar transfer learning, con intervalos de confianza |
| Latencia | Tiempo por imagen y por tomografía en GPU y CPU |

# Cómo se construyó el modelo (notebooks)

Los notebooks de `notebooks/` cuentan, etapa por etapa, cómo se construyó el sistema. **Ya están ejecutados**: al abrirlos se ven el código, las salidas y las figuras, así que para entender el trabajo basta con leerlos en este orden. Volver a ejecutarlos es **opcional**.

Si quieres ejecutar alguno: ábrelo en VS Code, elige el kernel del `.venv` y pulsa **Run All**. Necesitan `data_mha/`. Cada notebook lee los archivos que generó el anterior, y esos archivos ya vienen en el repositorio, así que cualquiera se puede ejecutar sin correr los anteriores.

**CPU** = funciona en cualquier computador. **GPU** = entrena o procesa volúmenes completos con el modelo; hace falta una GPU NVIDIA.

### 1. Datos y preprocesamiento

| Notebook | Qué hace | Requiere |
|---|---|---|
| `Dataset_splits` | Verifica los 100 pares imagen-máscara y crea la partición fija por paciente 75/5/20 (semilla 42) | CPU |
| `Ventaneo_hu` | Ventana ósea en unidades Hounsfield y realce de contraste (Otsu + CLAHE) | CPU |
| `EDA` | Análisis exploratorio: fragmentos por caso y por hueso, volúmenes y fragmento principal | CPU |
| `Distancias_referencia` | Distancia real de cada fragmento a su principal, con la transformada de distancia y el espaciado físico | CPU |
| `Visualizacion_3d` | Visualizador 1: proyecciones MIP y malla 3D del volumen crudo | CPU |
| `Targets_deteccion` | Cajas de cada región y lista de cortes para entrenar | CPU |

### 2. Piezas del modelo

| Notebook | Qué hace | Requiere |
|---|---|---|
| `Backbone` | Backbone FundidoraPC (versiones `base` y `plus`) | CPU |
| `cbam` | Bloque de atención CBAM (canal y espacial) | CPU |
| `Cabeza_grid` | Cabeza de detección por grilla, pérdidas y NMS | CPU |
| `Segmentacion_instancias` | Decoder que separa los fragmentos con centros y desplazamientos | GPU |
| `Entrenamiento_overfiting` | Prueba de memorización de un lote: comprueba que el modelo es capaz de aprender | GPU |

### 3. Entrenamiento y sistema completo

| Notebook | Qué hace | Requiere |
|---|---|---|
| `Entrenamiento_multitarea` | Primera versión del entrenamiento (solo axial), reemplazada por la siguiente | GPU |
| `Entrenamiento_multitarea_planos` | **Entrenamiento de los tres modelos que usa la app** (axial, coronal y sagital), por etapas | GPU |
| `prueba_ida_y_vuelta_planos` | Comprueba que pasar un volumen a cada plano y devolverlo no pierde información | CPU |
| `Negativos_ajuste` | Ajuste con cortes sin pelvis, para no inventar fragmentos fuera de ella | GPU |
| `nms_pertenencia_pipeline` | NMS por clase y cabeza de pertenencia (a qué hueso pertenece cada fragmento) | GPU |
| `salida_3d` | Une los cortes en fragmentos 3D y ajusta esa etapa con validación | GPU |
| `sam_baseline` | Comparación con SAM (modelo general, sin entrenar en este dataset) | GPU |
| `Test_final_fusion_B` | **Fusión 3D de los tres planos**, la reconstrucción que usa la app: promedia las probabilidades de los modelos de la etapa B y la evalúa en test | GPU |

# Scripts (`src/`)

La evaluación final, la ablación y las mediciones se hicieron con scripts de Python, que se ejecutan uno por uno desde la raíz del repositorio. Sus resultados ya están en `resultados/` (es lo que muestra el dashboard), así que **ejecutarlos también es opcional**. Todos necesitan `data_mha/`, salvo los dos que solo leen resultados.

Los que tienen opciones las muestran con `--help`; `analisis_pesos_cabezas` y `generar_evidencias_distancias` no tienen opciones y se ejecutan directamente.

| Script | Qué hace | Requiere |
|---|---|---|
| `python -m src.evaluacion --plano todos --split test` | Métricas de la consigna en test para los tres modelos. Con `--sam-checkpoint resultados/modelos/modelos_sam/sam_vit_b_01ec64.pth` agrega la comparación con SAM | GPU (o `--device cpu`) |
| `python -m src.medir_latencia --split test` | Latencia por imagen y por tomografía en GPU y CPU (con `--solo-cpu` si no hay GPU) | CPU y GPU |
| `python -m src.distancias_pred_vs_gt --split test` | Distancia de cada fragmento en la predicción frente a la anotación real | GPU recomendada |
| `python -m src.entrenar_pertenencia_planos --plano todos` | Entrena la cabeza de pertenencia de cada modelo | GPU recomendada |
| `python -m src.verificar_planos_val` | Verifica en validación el control de "pelvis presente" y el armado 3D de los fragmentos | GPU recomendada |
| `python -m src.preentrenar_backbone_pets --datos <carpeta oxford-iiit-pet>` | Preentrena el backbone en Oxford-IIIT Pet para el transfer learning (necesita ese dataset, no `data_mha/`) | GPU |
| `python -m src.ablacion entrenar --variante sin_cbam` | Entrena una variante de la ablación (`sin_cbam`, `con_tl` o `sin_cbam_con_tl`) | GPU |
| `python -m src.ablacion exportar-b --plano axial` | Exporta el modelo al final de la etapa B, usado como base de la ablación | CPU |
| `python -m src.resumen_ablacion --split test` | Tablas con intervalos de confianza: ablación y etapa C frente a B. Solo lee resultados | CPU |
| `python -m src.analisis_pesos_cabezas` | Análisis de los pesos entre la detección y la segmentación. Solo lee resultados | CPU |
| `python -m src.generar_evidencias_distancias` | Figuras de revisión de las distancias de referencia | CPU |
| `python -m src.evaluar_fusion_3d` | Dice e IoU por fragmento en 3D de la reconstrucción de la app (fusión de los tres planos), en test. Usa las probabilidades que guarda `Test_final_fusion_B` en `resultados/resultados_test_fusion.zip` | CPU |
| `python -m src.plausibilidad_pelvis referencia` | Rangos de tamaño y proporciones de la pelvis en los pacientes de entrenamiento. La app los usa para advertir cuando una tomografía se sale de lo esperado para una pelvis humana adulta | CPU |
| `python -m src.plausibilidad_pelvis verificar --predicciones` | Comprueba que la advertencia no salte en los pacientes de validación y prueba, y que sí detecte casos simulados (otro tamaño, lados invertidos, hueso faltante). Sin `--predicciones` usa solo las anotaciones | CPU (con `--predicciones`, GPU recomendada) |

# Model cards

Cada modelo tiene su ficha técnica, con arquitectura, datos de entrenamiento, métricas en test, alcances, limitaciones conocidas:

- [Modelo axial](model_cards/axial.md)
- [Modelo coronal](model_cards/coronal.md)
- [Modelo sagital](model_cards/sagital.md)

# Despliegue

La app se puede publicar desde este computador de dos formas. Con el ambiente instalado y los pesos descargados, desde la raíz del repositorio en PowerShell:

**Recomendado**
**Con túnel de Cloudflare** (se puede abrir desde cualquier red, en computador o celular):

```
powershell -ExecutionPolicy Bypass -File deploy\local.ps1
```

**Solo en la red local** (este computador y celulares conectados a la misma Wi-Fi):

```
powershell -ExecutionPolicy Bypass -File deploy\local.ps1 -SinTunel
```

- El script imprime las direcciones: `http://localhost:8501` para este computador, `http://<IP-local>:8501` para un celular en la misma Wi-Fi y `https://<algo>.trycloudflare.com` para el túnel.
- La primera vez instala `cloudflared` con `winget`, si no está instalado.
- El túnel gratuito de Cloudflare no deja subir archivos de más de 100 MB. Para tomografías más grandes, usa la red local (`-SinTunel`).
- `Ctrl+C` detiene la app y el túnel.

**Como usar el despliegue**

1. En **Inicio**, sube una tomografía (`.mha`, `.nii` o `.nii.gz`). Puede ser del dataset PENGWIN o una que no pertenezca a él.
2. Espera el análisis: corre los tres modelos (axial, coronal y sagital) y **fusiona sus resultados en 3D**. Usa la GPU si la hay y, si no, la CPU.
3. Revisa los tres visualizadores:
   - **Volumen crudo:** proyección de máxima intensidad (MIP) con umbral de 300 HU.
   - **Corte por corte:** elige el plano y mueve el slider para ver las cajas del detector de ese plano, las máscaras coloreadas por hueso y la distancia de cada fragmento.
   - **Reconstrucción 3D:** malla interactiva de la fusión de los tres planos, con la distancia en la etiqueta de cada fragmento.

**AWS:** el proyecto también está desplegado en un servidor de AWS EC2 (`deploy/instalar.sh`). Para acceder a esa versión, contacta al dueño de la máquina: [Juan Hoyos](https://github.com/JuanHoyos329).


# Uso de IA

El registro del uso de herramientas de IA generativa (prompts, qué se tomó de ellas y qué se corrigió a mano) está en [IA_USAGE.md](IA_USAGE.md).
