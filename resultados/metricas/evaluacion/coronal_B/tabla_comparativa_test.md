| Metrica | Objetivo | Modelo | SAM | Cumple |
|---|---|---|---|---|
| Pertenencia F1 macro (sistema completo) | >= 0.85 | 0.853 | - | si |
| Pertenencia AUC macro OvR (fragmentos emparejados) | >= 0.85 | 1.000 | - | si |
| Cobertura de fragmentos (IoU >= 0.5) |  | 0.760 | - |  |
| Pertenencia F1 con mascaras reales (diagnostico) |  | 0.985 | - |  |
| Regla semantica F1 (baseline, sistema completo) |  | 0.859 | - |  |
| Presencia de region F1 (detector, diagnostico) |  | 0.973 | - |  |
| Deteccion IoU promedio | >= 0.65 | 0.791 | - | si |
| Deteccion mAP@0.50 | >= 0.65 | 0.877 | - | si |
| Deteccion mAP@[0.50:0.95] | >= 0.40 | 0.569 | - | si |
| Segmentacion Dice por fragmento | >= 0.85 | 0.757 | - | no |
| Segmentacion IoU por fragmento | >= 0.70 | 0.661 | - | no |
| Dice semantico macro (modelo vs SAM) |  | 0.957 | n/a |  |
| IoU semantico macro (modelo vs SAM) |  | 0.917 | n/a |  |
| Dice semantico macro del decoder (diagnostico) |  | 0.961 | - |  |
