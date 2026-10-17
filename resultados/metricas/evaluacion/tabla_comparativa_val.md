| Metrica | Objetivo | Modelo | SAM | Cumple |
|---|---|---|---|---|
| Pertenencia F1 macro (sistema completo) | >= 0.85 | 0.835 | - | no |
| Pertenencia AUC macro OvR (fragmentos emparejados) | >= 0.85 | 0.997 | - | si |
| Cobertura de fragmentos (IoU >= 0.5) |  | 0.783 | - |  |
| Pertenencia F1 con mascaras reales (diagnostico) |  | 0.945 | - |  |
| Regla semantica F1 (baseline, sistema completo) |  | 0.870 | - |  |
| Presencia de region F1 (detector, diagnostico) |  | 0.927 | - |  |
| Deteccion IoU promedio | >= 0.65 | 0.802 | - | si |
| Deteccion mAP@0.50 | >= 0.65 | 0.882 | - | si |
| Deteccion mAP@[0.50:0.95] | >= 0.40 | 0.615 | - | si |
| Segmentacion Dice por fragmento | >= 0.85 | 0.746 | - | no |
| Segmentacion IoU por fragmento | >= 0.70 | 0.662 | - | no |
| Dice semantico macro (modelo vs SAM) |  | 0.914 | n/a |  |
| IoU semantico macro (modelo vs SAM) |  | 0.842 | n/a |  |
| Dice semantico macro del decoder (diagnostico) |  | 0.944 | - |  |
