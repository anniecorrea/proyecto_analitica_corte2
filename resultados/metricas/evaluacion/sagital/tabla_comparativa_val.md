| Metrica | Objetivo | Modelo | SAM | Cumple |
|---|---|---|---|---|
| Pertenencia F1 macro (sistema completo) | >= 0.85 | 0.879 | - | si |
| Pertenencia AUC macro OvR (fragmentos emparejados) | >= 0.85 | 1.000 | - | si |
| Cobertura de fragmentos (IoU >= 0.5) |  | 0.788 | - |  |
| Pertenencia F1 con mascaras reales (diagnostico) |  | 0.971 | - |  |
| Regla semantica F1 (baseline, sistema completo) |  | 0.881 | - |  |
| Presencia de region F1 (detector, diagnostico) |  | 0.968 | - |  |
| Deteccion IoU promedio | >= 0.65 | 0.829 | - | si |
| Deteccion mAP@0.50 | >= 0.65 | 0.900 | - | si |
| Deteccion mAP@[0.50:0.95] | >= 0.40 | 0.650 | - | si |
| Segmentacion Dice por fragmento | >= 0.85 | 0.781 | - | no |
| Segmentacion IoU por fragmento | >= 0.70 | 0.703 | - | si |
| Dice semantico macro (modelo vs SAM) |  | 0.947 | n/a |  |
| IoU semantico macro (modelo vs SAM) |  | 0.899 | n/a |  |
| Dice semantico macro del decoder (diagnostico) |  | 0.954 | - |  |
