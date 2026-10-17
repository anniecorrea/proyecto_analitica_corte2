| Metrica | Objetivo | Modelo | SAM | Cumple |
|---|---|---|---|---|
| Pertenencia F1 macro (sistema completo) | >= 0.85 | 0.886 | - | si |
| Pertenencia AUC macro OvR (fragmentos emparejados) | >= 0.85 | 1.000 | - | si |
| Cobertura de fragmentos (IoU >= 0.5) |  | 0.815 | - |  |
| Pertenencia F1 con mascaras reales (diagnostico) |  | 0.988 | - |  |
| Regla semantica F1 (baseline, sistema completo) |  | 0.888 | - |  |
| Presencia de region F1 (detector, diagnostico) |  | 0.973 | - |  |
| Deteccion IoU promedio | >= 0.65 | 0.855 | - | si |
| Deteccion mAP@0.50 | >= 0.65 | 0.933 | - | si |
| Deteccion mAP@[0.50:0.95] | >= 0.40 | 0.692 | - | si |
| Segmentacion Dice por fragmento | >= 0.85 | 0.793 | - | no |
| Segmentacion IoU por fragmento | >= 0.70 | 0.704 | - | si |
| Dice semantico macro (modelo vs SAM) |  | 0.962 | 0.803 |  |
| IoU semantico macro (modelo vs SAM) |  | 0.927 | 0.672 |  |
| Dice semantico macro del decoder (diagnostico) |  | 0.966 | - |  |
