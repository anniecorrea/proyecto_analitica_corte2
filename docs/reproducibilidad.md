# Reproducibilidad

## Ambiente

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
