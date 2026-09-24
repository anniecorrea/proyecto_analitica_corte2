# Resumen EDA Entrega 1

## Dataset

- Total de casos con imagen y label: 100
- CT y label con mismo tamano: 100 de 100
- CT y label con mismo spacing: 100 de 100
- Cortes por volumen: minimo 193, maximo 414, promedio 321.06
- Spacing z promedio: 0.8908 mm

## Fragmentos

- Total de fragmentos individuales: 575
- Fragmentos por caso: minimo 3, maximo 9, promedio 5.75
- Fragmentos por region:
  - Sacro: 153
  - Coxal izquierdo: 215
  - Coxal derecho: 207
- Percentil 5 de volumen: 3145.00 mm3
- Fragmentos pequenos en percentil 5: 29

## Casos con mas fragmentos

case_id
039    9
024    9
087    8
085    8
049    8

## Conclusiones

- El dataset local esta completo para el primer avance: hay 100 casos con imagen y label correspondiente.
- Todos los casos verificados mantienen consistencia de tamano y spacing entre CT y label.
- El numero de fragmentos por caso varia, lo cual anticipa dificultad desigual entre casos simples y casos complejos.
- Existen fragmentos de volumen muy pequeno; estos pueden ser dificiles de segmentar y afectar metricas como Dice o IoU.
- Las medidas de volumen y futuras distancias deben calcularse en milimetros usando el spacing fisico, no en pixeles.

## Archivos generados

- `metadata_casos.csv`
- `metadata_fragmentos.csv`
- `01_distribucion_cortes_z.png`
- `02_spacing_fisico.png`
- `03_fragmentos_por_caso.png`
- `04_fragmentos_por_region.png`
- `05_volumen_fragmentos_log10.png`
- `06_volumen_por_region.png`
- `07_caso_001_ct_label_overlay.png`
