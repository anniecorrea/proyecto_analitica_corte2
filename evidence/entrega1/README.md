# Evidencias Entrega 1

Esta carpeta contiene evidencias ligeras para el primer avance.

## Archivos

- `visualizador1_3d_raw_caso_001.html`: visualizador 3D interactivo del CT crudo usando umbral HU para hueso.
- `visualizador1_3d_raw_caso_001_resumen.txt`: resumen tecnico del visualizador 3D.
- `visualizador1_mesh_raw_caso_001.html`: visualizador 3D mejorado por malla, generado con umbral HU, limpieza basica y Marching Cubes.
- `visualizador1_mesh_raw_caso_001_resumen.txt`: resumen tecnico del visualizador por malla.
- `visualizador1_mip_raw_caso_001.png`: proyecciones MIP complementarias en tres vistas.
- `visualizador1_mip_raw_caso_001_resumen.txt`: resumen tecnico de la evidencia MIP.
- `ventaneo_hu/`: histogramas y comparacion visual para justificar el ventaneo y el umbral HU.
- `eda/`: graficas, tablas y resumen del analisis exploratorio del dataset.

## Explicacion corta

El visualizador 1 corresponde a una reconstruccion inicial del volumen CT crudo. Se aplica un umbral HU para resaltar estructuras oseas. La version inicial por puntos se conserva como evidencia, pero la version recomendada para presentar despues de la asesoria es la malla 3D, porque reduce el ruido visual y muestra una superficie mas interpretable.
