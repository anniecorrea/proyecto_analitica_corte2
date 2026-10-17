| Metrica | Objetivo | Modelo | SAM | Cumple |
|---|---|---|---|---|
| Pertenencia F1 macro (sistema completo) | >= 0.85 | 0.819 | - | no |
| Pertenencia AUC macro OvR (fragmentos emparejados) | >= 0.85 | 1.000 | - | si |
| Cobertura de fragmentos (IoU >= 0.5) |  | 0.712 | - |  |
| Pertenencia F1 con mascaras reales (diagnostico) |  | 0.984 | - |  |
| Regla semantica F1 (baseline, sistema completo) |  | 0.823 | - |  |
| Presencia de region F1 (detector, diagnostico) |  | 0.972 | - |  |
| Deteccion IoU promedio | >= 0.65 | 0.787 | - | si |
| Deteccion mAP@0.50 | >= 0.65 | 0.866 | - | si |
| Deteccion mAP@[0.50:0.95] | >= 0.40 | 0.555 | - | si |
| Segmentacion Dice por fragmento | >= 0.85 | 0.739 | - | no |
| Segmentacion IoU por fragmento | >= 0.70 | 0.636 | - | no |
| Dice semantico macro (modelo vs SAM) |  | 0.953 | 0.756 |  |
| IoU semantico macro (modelo vs SAM) |  | 0.910 | 0.608 |  |
| Dice semantico macro del decoder (diagnostico) |  | 0.957 | - |  |
