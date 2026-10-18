"""Dice e IoU por fragmento en 3D de la reconstrucción del frontend en test.

Misma definición que la métrica 2D de src/evaluacion.py (_fragment_metrics), pero con fragmentos 3D completos:
  - cada fragmento real se empareja uno a uno (asignación húngara sobre el IoU, sin umbral) con un fragmento predicho
    del mismo hueso; un fragmento real sin pareja cuenta como Dice = IoU = 0;
  - Dice = 2·IoU / (1 + IoU) de cada pareja; se promedia sobre todos los fragmentos reales del split.
También guarda el SQ del cuaderno (IoU medio de las parejas con IoU >= 0.5), la precisión y el recall a IoU >= 0.5.

La reconstrucción es la de la app (app/inferencia.py, predecir_fusion): probabilidades de los modelos de la etapa B
sumadas con pesos iguales, argmax y fragmentos = componentes conexos 3D de cada hueso de al menos 500 mm³. No infiere:
usa las probabilidades que guardó notebooks/Test_final_fusion_B.ipynb en resultados/resultados_test_fusion.zip (probs/),
así que corre en CPU sin GPU.

    python -m src.evaluar_fusion_3d

Guarda resultados/metricas/fusion_3d/metricas_test.json (lo lee el dashboard) y por_caso_test.csv.
"""
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
for _d in (ROOT, ROOT / "app", ROOT / "src"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import inferencia  # noqa: E402   (lectura canónica y fragmentos_desde_clases; no carga ningún modelo)
from src.rutas import DATASET_CSV, METRICAS_DIR, RESULTADOS_DIR  # noqa: E402

ZIP_FUSION = RESULTADOS_DIR / "resultados_test_fusion.zip"
SALIDA = METRICAS_DIR / "fusion_3d"
REGION = {"sacro": 1, "coxal_izquierdo": 2, "coxal_derecho": 3}


def clase_de_id_real(i):
    return 1 if i <= 10 else 2 if i <= 20 else 3


def fusionar(probs_por_plano):
    """{plano: [3, z, y, x] uint8} -> clases 0-3 (argmax de [fondo, sacro, coxal izq., coxal der.]), igual que la app."""
    acumulado = None
    for p in probs_por_plano.values():
        acumulado = p.astype(np.uint16) if acumulado is None else acumulado + p
    tope = 255 * len(probs_por_plano)
    clases = np.zeros(acumulado.shape[1:], dtype=np.uint8)
    for z0 in range(0, clases.shape[0], 32):
        a = acumulado[:, z0:z0 + 32].astype(np.int32)
        clases[z0:z0 + 32] = np.concatenate([(tope - a.sum(0))[None], a], axis=0).argmax(0)
    return clases


def metricas_por_fragmento(gt, pred, clase_pred):
    """Por fragmento real: IoU con su pareja (0 si no tiene). -> lista de (id real, iou)."""
    union = (gt > 0) | (pred > 0)
    g, p = gt[union].astype(np.int64), pred[union].astype(np.int64)
    gt_ids = [int(i) for i in np.unique(g) if i > 0]
    pr_ids = [int(i) for i in np.unique(p) if i > 0]
    if not gt_ids:
        return []
    gs, ps = np.bincount(g), np.bincount(p, minlength=max(pr_ids, default=0) + 1)
    P = max(pr_ids, default=0) + 1
    claves, inter = np.unique(g * P + p, return_counts=True)
    iou = np.zeros((len(gt_ids), len(pr_ids)))
    fila = {v: k for k, v in enumerate(gt_ids)}
    col = {v: k for k, v in enumerate(pr_ids)}
    for k, n in zip(claves, inter):
        a, b = int(k // P), int(k % P)
        if a == 0 or b == 0 or clase_de_id_real(a) != clase_pred.get(b):
            continue
        iou[fila[a], col[b]] = n / (gs[a] + ps[b] - n)
    pareja = {}
    if pr_ids:
        for r, c in zip(*linear_sum_assignment(-iou)):
            pareja[gt_ids[r]] = float(iou[r, c])
    return [(i, pareja.get(i, 0.0)) for i in gt_ids], len(pr_ids)


def main():
    if not ZIP_FUSION.is_file():
        raise SystemExit(f"Falta {ZIP_FUSION} (lo genera notebooks/Test_final_fusion_B.ipynb).")
    z = zipfile.ZipFile(ZIP_FUSION)
    casos = pd.read_csv(DATASET_CSV, dtype={"case_id": str}).query("split == 'test'")["case_id"].tolist()
    filas, n_pred_total, t0 = [], 0, time.time()
    for caso in casos:
        probs = {}
        for plano in inferencia.PLANOS:
            nombre = next(n for n in z.namelist() if n.startswith(f"probs/{caso}_{plano}_"))
            probs[plano] = np.load(io.BytesIO(z.read(nombre)))
        clases = fusionar(probs)
        del probs
        gt, spacing = inferencia.leer_canonico(inferencia.ETIQUETAS_DIR / f"{caso}.mha", es_etiqueta=True)
        inst, clase_de = inferencia.fragmentos_desde_clases(clases, spacing)
        del clases
        resultado, n_pred = metricas_por_fragmento(gt, inst, clase_de)
        n_pred_total += n_pred
        for gid, iou in resultado:
            filas.append({"caso": caso, "id_real": gid, "hueso": clase_de_id_real(gid), "iou": iou,
                          "dice": 2 * iou / (1 + iou) if iou else 0.0})
        print(f"  {caso}: {len(resultado)} fragmentos reales, {n_pred} predichos | "
              f"Dice medio {np.mean([2 * i / (1 + i) if i else 0 for _, i in resultado]):.3f}", flush=True)
    t = pd.DataFrame(filas)
    acierto = t["iou"] >= 0.5
    metricas = {
        "split": "test", "n_casos": len(casos), "n_fragmentos_reales": int(len(t)),
        "n_fragmentos_predichos": int(n_pred_total),
        "dice_por_fragmento_3d": float(t["dice"].mean()), "iou_por_fragmento_3d": float(t["iou"].mean()),
        "sq_iou_medio_aciertos": float(t.loc[acierto, "iou"].mean()) if acierto.any() else float("nan"),
        "recall_iou50": float(acierto.mean()), "precision_iou50": float(acierto.sum() / n_pred_total) if n_pred_total else float("nan"),
        "objetivos": {"dice_por_fragmento_3d": 0.85, "iou_por_fragmento_3d": 0.70},
        "definicion": ("misma que la métrica 2D de src/evaluacion.py: pareja uno a uno por IoU dentro del hueso, sin umbral; "
                       "fragmento real sin pareja = 0; promedio sobre todos los fragmentos reales"),
        "reconstruccion": "fusión de los 3 planos (modelos B, pesos iguales), componentes conexos 3D >= 500 mm³",
        "fuente_probabilidades": ZIP_FUSION.relative_to(ROOT).as_posix(),
    }
    SALIDA.mkdir(parents=True, exist_ok=True)
    (SALIDA / "metricas_test.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False), encoding="utf-8")
    t.to_csv(SALIDA / "por_fragmento_test.csv", index=False)
    print(f"\nTest ({len(casos)} pacientes, {len(t)} fragmentos reales, {n_pred_total} predichos) en {time.time() - t0:.0f} s")
    for k in ("dice_por_fragmento_3d", "iou_por_fragmento_3d", "sq_iou_medio_aciertos", "recall_iou50", "precision_iou50"):
        print(f"  {k:24s} {metricas[k]:.3f}")
    print(f"Guardado en {SALIDA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
