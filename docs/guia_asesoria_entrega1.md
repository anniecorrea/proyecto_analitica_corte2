# Guia para asesoria con el profesor Ferro

## 1. Resumen del proyecto

El proyecto consiste en desarrollar un sistema academico para trabajar con tomografias computarizadas de pelvis fracturada usando el dataset PENGWIN.

El sistema final debe integrar cuatro tareas principales:

- detectar regiones anatomicas de la pelvis mediante bounding boxes;
- segmentar fragmentos oseos individuales;
- clasificar cada fragmento segun su region anatomica: sacro, coxal izquierdo o coxal derecho;
- medir en milimetros la separacion entre fragmentos conminutos y el fragmento principal del mismo hueso.

Ademas del modelo, el proyecto pide un dashboard con visualizadores. Por eso no se trata solo de entrenar una red neuronal, sino de construir un flujo completo: datos, preprocesamiento, modelo, metricas, medicion, visualizacion, documentacion y despliegue.

## 2. Que significa el primer avance

El primer avance corresponde a la base del proyecto. Antes de entrenar modelos, el equipo debe demostrar que ya puede trabajar correctamente con los datos medicos.

Para esta primera entrega se busca cumplir:

- tener el dataset organizado;
- tener splits fijos de entrenamiento, validacion y prueba;
- poder cargar los archivos medicos `.mha`;
- leer informacion fisica del volumen, especialmente el spacing en milimetros;
- aplicar ventaneo o umbral HU para resaltar hueso;
- hacer EDA de fragmentos por caso;
- generar una primera visualizacion volumetrica del CT crudo.

En palabras simples, el mensaje para el profesor es:

> Ya tenemos los datos organizados, podemos leerlos, sabemos que imagen y label corresponden, tenemos particiones fijas, entendemos la distribucion de fragmentos y generamos una primera visualizacion 3D del volumen crudo antes de entrenar el modelo.

## 3. Estructura actual del repositorio

El repositorio se organizo de forma simple para que no se vea sobrecargado y sea reproducible:

```text
proyecto_analitica_corte2/
├── README.md
├── requirements.txt
├── entrega1.ipynb
├── data/
│   └── README.md
├── docs/
│   ├── reproducibilidad.md
│   └── guia_asesoria_entrega1.md
├── evidence/
│   └── entrega1/
├── splits/
└── src/
    ├── data/
    ├── utils/
    └── visualization/
```

Los datos pesados no se suben a GitHub. Cada integrante debe tenerlos localmente en:

```text
data_mha/
├── PENGWIN_CT_train_images_part1/
├── PENGWIN_CT_train_images_part2/
└── PENGWIN_CT_train_labels/
```

## 4. Reproducibilidad

Se corrigio el problema de rutas absolutas. Antes, el notebook dependia de rutas como `C:\Users\annie\...` o rutas especificas de un computador. Ahora la idea es que el proyecto use rutas relativas a la raiz del repositorio.

Para reproducir el avance:

```bash
git clone https://github.com/anniecorrea/proyecto_analitica_corte2.git
cd proyecto_analitica_corte2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Luego cada integrante debe ubicar los datos descomprimidos en `data_mha/`.

Para verificar que los datos estan bien ubicados:

```bash
python3 src/data/verificar_dataset_local.py
```

La salida esperada es:

```text
Imagenes parte 1: 50
Imagenes parte 2: 50
Imagenes totales unicas: 100
Labels totales: 100
Duplicados entre partes: 0
Imagenes sin label: []
Labels sin imagen: []
Dataset local verificado correctamente.
```

## 5. Que se ha realizado en el primer avance

### 5.1 Organizacion del dataset

Se verifico que existen:

- 50 imagenes en `PENGWIN_CT_train_images_part1`;
- 50 imagenes en `PENGWIN_CT_train_images_part2`;
- 100 labels en `PENGWIN_CT_train_labels`;
- 100 imagenes unicas;
- 100 labels;
- ningun duplicado entre partes;
- ninguna imagen sin label;
- ningun label sin imagen.

Esto muestra que la base imagen-label esta completa.

### 5.2 Splits fijos

El repositorio contiene:

```text
splits/train.txt
splits/val.txt
splits/test.txt
splits/dataset.csv
```

Los splits permiten que el equipo trabaje con la misma particion de datos y que los resultados sean comparables.

Punto a validar con el profesor:

- si la proporcion actual de splits es adecuada;
- si espera validacion por paciente/caso de alguna forma especifica;
- si quiere que los splits se reporten explicitamente en el informe.

### 5.3 Carga de archivos medicos

El avance trabaja con archivos `.mha`, que son los archivos descargados del dataset.

Tambien se contempla la conversion a NIfTI en el notebook, pero para la primera entrega lo importante es demostrar que podemos leer el formato medico original, extraer el volumen y acceder al spacing fisico.

### 5.4 Ventaneo y umbral HU

Se usa un umbral HU para resaltar estructuras oseas. En las evidencias generadas se usa:

```text
HU >= 300
```

Esto no es una segmentacion final. Es un preprocesamiento clasico sobre el volumen crudo para visualizar principalmente hueso.

Punto a validar con el profesor:

- si `HU >= 300` es aceptable para el visualizador raw inicial;
- si prefiere otro rango de ventana para hueso;
- si quiere que se explique como ventaneo HU o como umbral HU.

### 5.5 EDA de fragmentos

El notebook realiza un analisis exploratorio de fragmentos. Entre los resultados trabajados estan:

- conteo total de casos;
- validacion de imagen-label;
- dimensiones de imagenes y labels;
- spacing de imagenes y labels;
- conteo de fragmentos por caso;
- distribucion por region anatomica;
- volumen aproximado de fragmentos en mm3;
- identificacion de fragmento principal por region;
- conteo de huesos sin fractura y fracturados por region.

Ademas, se agrego una evidencia EDA reproducible en:

```text
evidence/entrega1/eda/
```

Esta carpeta incluye:

- metadata por caso;
- metadata de fragmentos;
- distribucion de cortes por volumen;
- distribucion de spacing fisico;
- distribucion de fragmentos por caso;
- conteo de fragmentos por region anatomica;
- distribucion de volumen de fragmentos;
- volumen de fragmentos por region;
- ejemplo visual CT con mascara superpuesta.

Resultados importantes ya observados:

- total de casos: 100;
- total de fragmentos individuales: 575;
- fragmentos por caso: minimo 3, maximo 9, media 5.75;
- fragmentos por region:
  - coxal izquierdo: 215;
  - coxal derecho: 207;
  - sacro: 153.

Punto a validar con el profesor:

- si esta taxonomia de labels esta correctamente interpretada;
- si sacro, coxal izquierdo y coxal derecho son las tres clases esperadas;
- si debemos consolidar ilion, isquion y pubis como un solo coxal por lateralidad, como dice la guia.

## 6. Visualizador 1

La guia pide un visualizador inicial del volumen crudo mediante MIP y mostrando hueso. Para cubrir esto de forma clara, se generaron dos evidencias:

### 6.1 Visualizador 3D raw

Archivo:

```text
evidence/entrega1/visualizador1_3d_raw_caso_001.html
```

Este archivo muestra una vista 3D interactiva del volumen CT crudo. Los puntos representan voxeles cuya intensidad supera el umbral HU definido para hueso.

Explicacion sugerida:

> Los puntos corresponden a voxeles del CT con intensidad HU mayor o igual a 300. Es una visualizacion raw inicial, no una segmentacion del modelo. Por eso puede incluir ruido, camilla u otras estructuras de alta intensidad.

### 6.2 Evidencia MIP complementaria

Archivo:

```text
evidence/entrega1/visualizador1_mip_raw_caso_001.png
```

Esta imagen muestra proyecciones MIP del volumen en tres vistas: axial, coronal y sagital.

Explicacion sugerida:

> El MIP proyecta el volumen CT y conserva los voxeles de mayor intensidad en cada direccion, permitiendo resaltar estructuras oseas del volumen raw.

## 7. Dudas importantes para la asesoria

### Datos y primer avance

1. ¿La organizacion actual del dataset en imagenes parte 1, parte 2 y labels es adecuada para el proyecto?
2. ¿Le parece bien trabajar directamente con `.mha` o prefiere que convirtamos todo a NIfTI desde el inicio?
3. ¿El split train/val/test actual es suficiente o recomienda una proporcion especifica?
4. ¿Debemos reportar los splits como IDs de caso en el informe?
5. ¿Es necesario conservar `dataset.csv` con rutas relativas o basta con `train.txt`, `val.txt` y `test.txt`?

### Visualizador 1

6. Para el visualizador 1, ¿espera una nube de puntos 3D, una superficie tipo malla o basta con MIP en tres vistas?
7. ¿El umbral `HU >= 300` es aceptable para resaltar hueso en el visualizador raw?
8. ¿Prefiere que el visualizador 1 sea interactivo desde el dashboard o por ahora puede presentarse como evidencia HTML/PNG?
9. ¿Debemos eliminar estructuras externas como camilla o ruido desde este primer avance, o eso se acepta por ser volumen crudo?

### EDA y taxonomia

10. ¿La taxonomia correcta es sacro, coxal izquierdo y coxal derecho?
11. ¿Confirmamos que no debemos separar ilion, isquion y pubis?
12. ¿El analisis de volumen de fragmentos en mm3 es pertinente para el primer avance?
13. ¿Debemos identificar desde ya el fragmento principal por region para preparar la medicion de distancia?

### Siguiente etapa tecnica

14. Para la deteccion, ¿espera bounding boxes por corte 2D o por volumen/caso 3D?
15. ¿La arquitectura debe trabajar corte a corte desde el inicio o podemos usar informacion 3D para preprocesamiento?
16. ¿La cabeza de deteccion debe ser grid propio tipo YOLO conceptual, pero sin usar YOLO?
17. ¿Que nivel de detalle espera en la implementacion de NMS propio?
18. ¿CBAM debe estar en el backbone antes de todas las cabezas o se puede probar como ablacion?

### Entrega y evaluacion

19. ¿Para la primera entrega espera solo notebook o tambien scripts reproducibles?
20. ¿Las evidencias en `evidence/entrega1/` son suficientes para mostrar el avance?
21. ¿Quiere capturas en el informe o acepta archivos HTML/PNG reproducibles?
22. ¿La advertencia de uso no clinico debe aparecer desde el primer dashboard o solo en la entrega final?

## 8. Como presentar el avance en la asesoria

Orden recomendado:

1. Mostrar README del repo y explicar estructura.
2. Ejecutar o mostrar salida de:

```bash
python3 src/data/verificar_dataset_local.py
```

3. Mostrar splits:

```text
splits/train.txt
splits/val.txt
splits/test.txt
```

4. Abrir `entrega1.ipynb` y explicar:
   - carga de datos;
   - validacion imagen-label;
   - spacing;
   - conteo de fragmentos;
   - EDA.

5. Abrir visualizador 3D:

```text
evidence/entrega1/visualizador1_3d_raw_caso_001.html
```

6. Mostrar MIP complementario:

```text
evidence/entrega1/visualizador1_mip_raw_caso_001.png
```

7. Cerrar con las dudas tecnicas al profesor.

## 9. Riesgos identificados

- El visualizador 3D actual es una nube de voxeles, no una malla. Puede ser suficiente para evidencia raw, pero conviene preguntarle si espera una superficie.
- El notebook debe mantenerse sin rutas absolutas para que no dependa del computador de una persona.
- Los datos pesados no deben subirse a GitHub.
- Hay que confirmar que la taxonomia de labels se esta interpretando correctamente.
- El siguiente avance debe pasar de exploracion de datos a arquitectura: backbone, CBAM y cabeza de deteccion.

## 10. Frase corta para explicar el avance

> En este primer avance dejamos lista la base reproducible del proyecto: organizamos el dataset PENGWIN, verificamos correspondencia entre imagenes y labels, definimos splits fijos, cargamos volumenes medicos `.mha`, extrajimos spacing fisico, realizamos EDA de fragmentos y generamos el primer visualizador raw del volumen con umbral HU y vista 3D interactiva.
