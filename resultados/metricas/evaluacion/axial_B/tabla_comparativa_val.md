| Metrica | Objetivo | Modelo | SAM | Cumple |
|---|---|---|---|---|
| Pertenencia F1 macro (sistema completo) | >= 0.85 | 0.876 | - | si |
| Pertenencia AUC macro OvR (fragmentos emparejados) | >= 0.85 | 1.000 | - | si |
| Cobertura de fragmentos (IoU >= 0.5) |  | 0.798 | - |  |
| Pertenencia F1 con mascaras reales (diagnostico) |  | 0.985 | - |  |
| Regla semantica F1 (baseline, sistema completo) |  | 0.878 | - |  |
| Presencia de region F1 (detector, diagnostico) |  | 0.975 | - |  |
| Deteccion IoU promedio | >= 0.65 | 0.844 | - | si |
| Deteccion mAP@0.50 | >= 0.65 | 0.928 | - | si |
| Deteccion mAP@[0.50:0.95] | >= 0.40 | 0.675 | - | si |
| Segmentacion Dice por fragmento | >= 0.85 | 0.777 | - | no |
| Segmentacion IoU por fragmento | >= 0.70 | 0.685 | - | no |
| Dice semantico macro (modelo vs SAM) |  | 0.953 | n/a |  |
| IoU semantico macro (modelo vs SAM) |  | 0.910 | n/a |  |
| Dice semantico macro del decoder (diagnostico) |  | 0.958 | - |  |
