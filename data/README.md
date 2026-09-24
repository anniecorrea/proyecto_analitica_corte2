# Datos locales

Los datos PENGWIN son pesados y no deben subirse a GitHub.

Cada integrante debe crear o copiar esta carpeta en la raiz del repo:

```text
data_mha/
├── PENGWIN_CT_train_images_part1/
├── PENGWIN_CT_train_images_part2/
└── PENGWIN_CT_train_labels/
```

Conteo esperado:

- `PENGWIN_CT_train_images_part1`: 50 archivos `.mha`
- `PENGWIN_CT_train_images_part2`: 50 archivos `.mha`
- `PENGWIN_CT_train_labels`: 100 archivos `.mha`

Validar:

```bash
python3 src/data/verificar_dataset_local.py
```
