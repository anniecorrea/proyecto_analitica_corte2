# Reproducibilidad

## Ambiente

Primero clonar el repositorio en una carpeta estable, por ejemplo:

```bash
mkdir -p ~/Documents/proyectos
cd ~/Documents/proyectos
git clone https://github.com/anniecorrea/proyecto_analitica_corte2.git
cd proyecto_analitica_corte2
```

Evitar clonar en `Downloads`, carpetas temporales o rutas que luego se vayan a mover.

Desde la raiz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Rutas

El codigo usa rutas relativas al repositorio. No se deben escribir rutas absolutas como:

```python
C:\Users\annie\...
/ruta/absoluta/Downloads/...
```

Las rutas compartidas estan en:

```text
src/utils/project_paths.py
```

## Verificar datos

La carpeta de datos debe quedar asi:

```text
proyecto_analitica_corte2/
├── README.md
├── entrega1.ipynb
├── src/
└── data_mha/
    ├── PENGWIN_CT_train_images_part1/
    ├── PENGWIN_CT_train_images_part2/
    └── PENGWIN_CT_train_labels/
```

```bash
python3 src/data/verificar_dataset_local.py
```

Salida esperada:

```text
Imagenes parte 1: 50
Imagenes parte 2: 50
Imagenes totales unicas: 100
Labels totales: 100
Dataset local verificado correctamente.
```

## Generar evidencias

```bash
python3 src/visualization/generar_visualizador1_3d_raw.py
python3 src/visualization/generar_visualizador1_mip.py
```

Las salidas quedan en:

```text
evidence/entrega1/
```
