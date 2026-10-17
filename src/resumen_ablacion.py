"""Comparaciones con intervalo de confianza: etapa C vs etapa B (3 planos) y ablación de CBAM y transfer learning.

Lee lo que escribe src/evaluacion.py en resultados/metricas/evaluacion/{etiqueta}/ y calcula, para cada par de
modelos evaluados sobre los MISMOS cortes, la diferencia de cada métrica con un intervalo de confianza del 95 %
por bootstrap PAREADO POR CASO: se remuestrean pacientes (no cortes, que dentro de un paciente no son independientes)
y en cada réplica se recalcula la métrica de los dos modelos sobre los mismos pacientes.

Métricas con intervalo (se rehacen exactamente como en src/evaluacion.py, a partir de sus CSV por corte/fragmento):
- Dice e IoU por fragmento: suma sobre todos los fragmentos reales / número de fragmentos (cortes_{split}.csv).
- F1 macro de pertenencia del sistema completo, con la cabeza de pertenencia y con la regla semántica del decoder (la
  que usa el frontend) (pertenencia_detalle_{split}_pred.csv).
mAP@0.50, mAP@[.50:.95] e IoU de cajas se reportan sin intervalo (necesitan todas las detecciones).

Lectura: la diferencia es "significativa" si el intervalo no incluye el 0, y "relevante" si además supera el
margen de 0.01 (el mismo que usó el equipo en la regla de negativos). La decisión C vs B se tomó con validación;
test solo se reporta.

    python -m src.resumen_ablacion --split test
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rutas import METRICAS_DIR  # noqa: E402

EVAL_DIR = METRICAS_DIR / "evaluacion"
MARGEN = 0.01
N_BOOTSTRAP = 2000
PLANOS = ("axial", "coronal", "sagital")
VARIANTES_ABLACION = {             # etiqueta de evaluación -> (CBAM, transfer learning); la base es axial_B
    "axial_B": ("sí", "no"),
    "ablacion_axial_sin_cbam": ("no", "no"),
    "ablacion_axial_con_tl": ("sí", "sí"),
    "ablacion_axial_sin_cbam_con_tl": ("no", "sí"),
}
SIN_INTERVALO = {"mAP@0.50": "bbox_map50", "mAP@[.50:.95]": "bbox_map50_95", "IoU de cajas": "bbox_mean_iou"}


def cargar(etiqueta, split):
    carpeta = EVAL_DIR / etiqueta
    metricas = carpeta / f"metricas_{split}.json"
    if not metricas.is_file():
        return None
    cortes = pd.read_csv(carpeta / f"cortes_{split}.csv", dtype={"case_id": str})
    if "n_gt_fragments" not in cortes:
        raise ValueError(f"{carpeta.name}/cortes_{split}.csv no tiene n_gt_fragments: repetir la evaluación con el "
                         "src/evaluacion.py actual")
    cortes = cortes[cortes["n_gt_fragments"] > 0]
    por_caso = (cortes.assign(dice=cortes["model_fragment_dice"] * cortes["n_gt_fragments"],
                              iou=cortes["model_fragment_iou"] * cortes["n_gt_fragments"])
                .groupby("case_id")[["dice", "iou", "n_gt_fragments"]].sum())
    perte = pd.read_csv(carpeta / f"pertenencia_detalle_{split}_pred.csv", dtype={"case_id": str})
    return {"etiqueta": etiqueta, "json": json.loads(metricas.read_text(encoding="utf-8")), "por_caso": por_caso,
            "pertenencia": {c: g[["clase_real", "pred_cabeza", "pred_semantica"]].to_numpy()
                            for c, g in perte.groupby("case_id")}}


def _f1(filas, col):
    return f1_score(filas[:, 0], filas[:, col], labels=[1, 2, 3], average="macro", zero_division=0)


def metricas(m, casos):
    """Las cuatro métricas con intervalo, calculadas solo con `casos` (con repetición, para el bootstrap)."""
    pc = m["por_caso"].loc[casos]
    filas = np.concatenate([m["pertenencia"][c] for c in casos if c in m["pertenencia"]])
    return {"Dice por fragmento": pc["dice"].sum() / pc["n_gt_fragments"].sum(),
            "IoU por fragmento": pc["iou"].sum() / pc["n_gt_fragments"].sum(),
            "F1 pertenencia (cabeza)": _f1(filas, 1),
            "F1 pertenencia (regla semántica)": _f1(filas, 2)}


def comparar(nuevo, ref, seed=42):
    """Filas con valor de cada modelo, diferencia (nuevo - ref) e IC 95 % por bootstrap pareado por caso."""
    casos = sorted(set(nuevo["por_caso"].index) & set(ref["por_caso"].index))
    if casos != sorted(nuevo["por_caso"].index) or casos != sorted(ref["por_caso"].index):
        raise ValueError(f"{nuevo['etiqueta']} y {ref['etiqueta']} no se evaluaron sobre los mismos casos")
    a, b = metricas(nuevo, casos), metricas(ref, casos)
    rng = np.random.default_rng(seed)
    difs = {k: [] for k in a}
    for _ in range(N_BOOTSTRAP):
        muestra = list(rng.choice(casos, size=len(casos), replace=True))
        ma, mb = metricas(nuevo, muestra), metricas(ref, muestra)
        for k in difs:
            difs[k].append(ma[k] - mb[k])
    filas = []
    for k in a:
        lo, hi = np.percentile(difs[k], [2.5, 97.5])
        d = a[k] - b[k]
        filas.append({"metrica": k, "referencia": b[k], "nuevo": a[k], "diferencia": d, "ic95_inf": lo, "ic95_sup": hi,
                      "significativa": bool(lo > 0 or hi < 0), "relevante": bool((lo > 0 or hi < 0) and abs(d) >= MARGEN)})
    for k, clave in SIN_INTERVALO.items():
        va, vb = nuevo["json"].get(clave), ref["json"].get(clave)
        filas.append({"metrica": k, "referencia": vb, "nuevo": va, "diferencia": None if va is None or vb is None else va - vb,
                      "ic95_inf": None, "ic95_sup": None, "significativa": None, "relevante": None})
    return pd.DataFrame(filas).assign(casos=len(casos))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    args = parser.parse_args()
    salida = METRICAS_DIR / "comparaciones"
    salida.mkdir(parents=True, exist_ok=True)

    # --- etapa C (modelo en uso) frente a etapa B, por plano ---
    tablas = []
    for plano in PLANOS:
        try:
            c, b = cargar(plano, args.split), cargar(f"{plano}_B", args.split)
        except ValueError as e:
            print(f"[C vs B] {plano}: {e}; se omite")
            continue
        if c is None or b is None:
            print(f"[C vs B] {plano}: falta la evaluación de {'C' if c is None else 'B'} en {args.split}; se omite")
            continue
        tablas.append(comparar(c, b).assign(plano=plano, referencia_modelo="B", nuevo_modelo="C"))
    if tablas:
        t = pd.concat(tablas, ignore_index=True)
        t.to_csv(salida / f"c_vs_b_{args.split}.csv", index=False)
        print(f"\nEtapa C frente a B ({args.split}; diferencia = C - B):")
        print(t[["plano", "metrica", "referencia", "nuevo", "diferencia", "ic95_inf", "ic95_sup", "relevante"]]
              .round(4).to_string(index=False))

    # --- ablación (axial, etapa B): cada variante frente a la base (CBAM, sin transfer learning) ---
    try:
        base = cargar("axial_B", args.split)
    except ValueError as e:
        print(f"[ablación] {e}; se omite")
        base = None
    tablas = []
    for etiqueta, (cbam, tl) in VARIANTES_ABLACION.items():
        if etiqueta == "axial_B":
            continue
        v = cargar(etiqueta, args.split)
        if v is None or base is None:
            print(f"[ablación] falta la evaluación de {etiqueta if v is None else 'axial_B'} en {args.split}; se omite")
            continue
        tablas.append(comparar(v, base).assign(variante=etiqueta.removeprefix("ablacion_axial_"), cbam=cbam,
                                               transfer_learning=tl))
    if tablas:
        t = pd.concat(tablas, ignore_index=True)
        t.to_csv(salida / f"ablacion_{args.split}.csv", index=False)
        print(f"\nAblación en el plano axial, etapa B ({args.split}; diferencia = variante - base con CBAM y sin TL):")
        print(t[["variante", "metrica", "referencia", "nuevo", "diferencia", "ic95_inf", "ic95_sup", "relevante"]]
              .round(4).to_string(index=False))
        # tabla compacta para el informe: una fila por modelo
        filas = []
        for etiqueta, (cbam, tl) in VARIANTES_ABLACION.items():
            m = cargar(etiqueta, args.split)
            if m is None:
                continue
            casos = sorted(m["por_caso"].index)
            filas.append({"modelo": etiqueta.removeprefix("ablacion_axial_").replace("axial_B", "base"), "CBAM": cbam,
                          "transfer_learning": tl, **metricas(m, casos),
                          **{k: m["json"].get(clave) for k, clave in SIN_INTERVALO.items()}})
        pd.DataFrame(filas).to_csv(salida / f"ablacion_tabla_{args.split}.csv", index=False)
    print(f"\nGuardado en {salida.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
