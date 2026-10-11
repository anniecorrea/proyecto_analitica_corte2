"""Donde vive cada archivo del proyecto. Cambiar una carpeta aqui la cambia en notebooks y scripts.

resultados/            todo lo que generan los notebooks y scripts, ordenado por tipo de archivo
|-- datos/             CSV que leen otros notebooks (fragmentos, manifiesto de cortes, distancias)
|-- metricas/          historiales de entrenamiento, configuraciones y tablas de resultados
|-- figuras/           imagenes PNG, una subcarpeta por etapa
|-- html/              visualizadores interactivos
|-- modelos/           pesos .pth / .pt y el batch fijo (se suben con Git LFS)
`-- cache/             cache de cortes del entrenamiento (no se sube a GitHub)

Dentro de metricas/, figuras/ y modelos/ cada etapa usa el mismo nombre de subcarpeta
(entrenamiento_multitarea, entrenamiento_overfiting, nms_pertenencia...).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_MHA_DIR = ROOT / "data_mha"
SPLITS_DIR = ROOT / "splits"
NOTEBOOKS_DIR = ROOT / "notebooks"

RESULTADOS_DIR = ROOT / "resultados"
DATOS_DIR = RESULTADOS_DIR / "datos"
METRICAS_DIR = RESULTADOS_DIR / "metricas"
FIGURAS_DIR = RESULTADOS_DIR / "figuras"
HTML_DIR = RESULTADOS_DIR / "html"
MODELOS_DIR = RESULTADOS_DIR / "modelos"
CACHE_DIR = RESULTADOS_DIR / "cache"

# Archivos que comparten varios notebooks y scripts
DATASET_CSV = SPLITS_DIR / "dataset.csv"
FRAGMENTOS_DIR = DATOS_DIR / "fragmentos"                      # metadata_fragments.csv y main_fragments.csv (EDA)
MANIFEST_CSV = DATOS_DIR / "targets" / "manifest_cortes_positivos.csv"
DISTANCIAS_REFERENCIA_CSV = DATOS_DIR / "distancias_referencia.csv"
BATCH_FIJO_PT = MODELOS_DIR / "batch_fijo.pt"
