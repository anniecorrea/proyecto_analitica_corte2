# proyecto_analitica_corte2
Sistema de detección, segmentación y clasificación de fracturas pélvicas a partir de tomografía computarizada (dataset PENGWIN)

## Datos
Las carpetas `data/` y `data_mha/` no están incluidas en este repositorio por su tamaño.
Cada integrante debe tener localmente:
- `data_mha/PENGWIN_CT_train_images_part1/`
- `data_mha/PENGWIN_CT_train_images_part2/`
- `data_mha/PENGWIN_CT_train_labels/`

Ajustar `BASE_DIR` en las primeras celdas de `entrega1.ipynb` según la ruta local de cada quien.
Los archivos `dataset.csv`, `dataset_nifti.csv`, `splits/*.txt`, `metadata_fragments.csv` y 
`main_fragments.csv` ya están versionados y no necesitan regenerarse (aunque el notebook 
lo permite si hace falta).