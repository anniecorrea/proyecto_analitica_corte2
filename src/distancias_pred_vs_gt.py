"""Distancia de separación: predicción vs referencia (consigna 3.3), con el pipeline final del frontend.

La comparación de salida_3d.ipynb (10.7) solo usa los fragmentos reales acertados con IoU 3D >= 0.5. Con la salida 3D
congelada casi todos los conminutos quedan fundidos con el principal de su hueso, así que esa tabla solo traía
principales (0 mm contra 0 mm) y no medía nada. Aquí hay una fila por CADA fragmento real, con su estado:

- separado: un fragmento predicho que no es el principal lo cubre con IoU 3D >= --iou-min; se compara su distancia.
- fusionado: la mayor parte del fragmento real cae dentro del principal predicho de su hueso. Para el modelo el trozo
  es parte del hueso, así que su distancia predicha es 0 mm; el error es la distancia real entera.
- omitido: ninguno de los dos casos (cae en el fondo o en un trozo predicho que no lo cubre bien); sin distancia.

Así se ve cuánto error pasa de la segmentación a la medida, que es lo que pide la consigna.

Desde la raíz del repositorio, con los datos en data_mha/:

    python -m src.distancias_pred_vs_gt                      # val, los tres planos
    python -m src.distancias_pred_vs_gt --split test --plano axial

Guarda resultados/metricas/distancias_pred_vs_gt/distancias_{split}.csv (una fila por fragmento real y plano) y
resumen_{split}.csv (una fila por plano).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from distancias_fragmentos import measure_fragment_separations  # noqa: E402
from rutas import DATA_MHA_DIR, DATASET_CSV, METRICAS_DIR  # noqa: E402

SALIDA_DIR = METRICAS_DIR / "distancias_pred_vs_gt"
IOU_MIN = 0.5                 # el mismo umbral "fragmentwise" de salida_3d.ipynb
FRACCION_FUSION = 0.5         # fracción del fragmento real dentro del principal predicho para darlo por fusionado

COLUMNAS = ["caso", "region", "gt_id", "gt_es_principal", "gt_voxeles", "gt_distancia_mm", "estado", "pred_id",
            "iou_3d", "fraccion_en_principal_pred", "pred_distancia_mm", "error_abs_mm", "principal_ok"]


def _solapes(gt, pred):
    """{(id real, id predicho): vóxeles en común} y tamaños, solo sobre los vóxeles con algo en alguno de los dos."""
    union = (gt > 0) | (pred > 0)
    claves, cuentas = np.unique(gt[union].astype(np.int64) * 65536 + pred[union].astype(np.int64), return_counts=True)
    g_ids, p_ids = claves // 65536, claves % 65536
    inter, gt_size, pred_size = {}, {}, {}
    for g, p, n in zip(g_ids.tolist(), p_ids.tolist(), cuentas.tolist()):
        if g:
            gt_size[g] = gt_size.get(g, 0) + n
        if p:
            pred_size[p] = pred_size.get(p, 0) + n
        if g and p:
            inter[(g, p)] = n
    return inter, gt_size, pred_size


def comparar_distancias(pred_volume, pred_region_by_id, gt_volume, spacing_xyz, case_id=None,
                        iou_min=IOU_MIN, fraccion_fusion=FRACCION_FUSION):
    """Una fila por fragmento real con su distancia en la referencia y en la predicción (ver el docstring del módulo).
    pred_region_by_id: {id predicho: región}; los IDs reales siguen la codificación del dataset (1-10, 11-20, 21-30)."""
    if pred_volume.shape != gt_volume.shape:
        raise ValueError(f"La predicción {pred_volume.shape} no tiene la geometría de la etiqueta {gt_volume.shape}")
    gt_dist = measure_fragment_separations(gt_volume.astype(np.int32), spacing_xyz, case_id=case_id)
    if gt_dist.empty:
        return pd.DataFrame(columns=COLUMNAS)
    pr_dist = measure_fragment_separations(pred_volume.astype(np.int32), spacing_xyz, case_id=case_id,
                                           region_by_id=pred_region_by_id)
    pr_dist = pr_dist.set_index("fragment_id") if not pr_dist.empty else pr_dist
    pr_main = {} if pr_dist.empty else {r: int(m) for r, m in zip(pr_dist["region"], pr_dist["main_fragment_id"])}
    inter, gt_size, pred_size = _solapes(gt_volume, pred_volume)

    def iou(g, p):
        n = inter.get((g, p), 0)
        return n / (gt_size[g] + pred_size[p] - n) if n else 0.0

    gt_main = {r: int(m) for r, m in zip(gt_dist["region"], gt_dist["main_fragment_id"])}
    principal_ok = {r: (r in pr_main and iou(gt_main[r], pr_main[r]) >= iou_min) for r in gt_main}

    filas = []
    for fila in gt_dist.itertuples(index=False):
        g, region = int(fila.fragment_id), fila.region
        candidatos = [p for p, r in pred_region_by_id.items() if r == region and (g, p) in inter]
        main_p = pr_main.get(region)
        frac_main = inter.get((g, main_p), 0) / gt_size[g] if main_p is not None else 0.0
        no_principales = [(iou(g, p), p) for p in candidatos if p != main_p]
        mejor_iou, mejor_p = max(no_principales) if no_principales else (0.0, None)

        if fila.is_main:
            estado, pred_id, iou_val = "principal", main_p, iou(g, main_p) if main_p is not None else 0.0
            pred_d = 0.0 if main_p is not None else np.nan
        elif mejor_p is not None and mejor_iou >= iou_min:
            estado, pred_id, iou_val = "separado", mejor_p, mejor_iou
            pred_d = float(pr_dist.at[mejor_p, "distance_mm"])
        elif frac_main >= fraccion_fusion:
            estado, pred_id, iou_val, pred_d = "fusionado", main_p, iou(g, main_p), 0.0
        else:
            estado, pred_id, iou_val, pred_d = "omitido", mejor_p, mejor_iou, np.nan

        filas.append({"caso": case_id, "region": region, "gt_id": g, "gt_es_principal": bool(fila.is_main),
                      "gt_voxeles": int(fila.n_voxels), "gt_distancia_mm": float(fila.distance_mm), "estado": estado,
                      "pred_id": pred_id, "iou_3d": round(float(iou_val), 3),
                      "fraccion_en_principal_pred": round(float(frac_main), 3), "pred_distancia_mm": pred_d,
                      "error_abs_mm": abs(pred_d - float(fila.distance_mm)) if not np.isnan(pred_d) else np.nan,
                      "principal_ok": bool(principal_ok.get(region, False))})
    return pd.DataFrame(filas, columns=COLUMNAS)


def resumir(tabla):
    """Una fila por plano: cuántos conminutos hay en cada estado y el error de la distancia (mm)."""
    filas = []
    for plano, t in tabla.groupby("plano", sort=False):
        c = t[~t["gt_es_principal"]]
        medidos = c[c["estado"].isin(["separado", "fusionado"])]
        sep_real = c[c["gt_distancia_mm"] > 0]
        filas.append({
            "plano": plano, "casos": t["caso"].nunique(), "conminutos": len(c),
            **{f"n_{e}": int((c["estado"] == e).sum()) for e in ("separado", "fusionado", "omitido")},
            "principales_ok": int(t[t["gt_es_principal"]]["principal_ok"].sum()),
            "principales": int(t["gt_es_principal"].sum()),
            "mae_mm": medidos["error_abs_mm"].mean(), "mediana_error_mm": medidos["error_abs_mm"].median(),
            "max_error_mm": medidos["error_abs_mm"].max(),
            "mae_separados_mm": c.loc[c["estado"] == "separado", "error_abs_mm"].mean(),
            "conminutos_separados_en_gt": len(sep_real),
            "mae_separados_en_gt_mm": sep_real["error_abs_mm"].mean()})
    return pd.DataFrame(filas)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--plano", default="todos", choices=["todos", "axial", "coronal", "sagital"])
    parser.add_argument("--casos", nargs="*", help="IDs concretos (por defecto, todos los del split)")
    parser.add_argument("--iou-min", type=float, default=IOU_MIN)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "app"))
    import inferencia                                    # torch y los cuadernos solo hacen falta aquí

    dataset = pd.read_csv(DATASET_CSV, dtype={"case_id": str})
    casos = args.casos or dataset.loc[dataset["split"] == args.split, "case_id"].tolist()
    planos = inferencia.PLANOS if args.plano == "todos" else (args.plano,)
    pipeline = inferencia.cargar_pipeline()
    print(f"Pipeline en {pipeline['dispositivo']} | configuración 3D {pipeline['config_3d_creada']} | "
          f"{len(casos)} casos de {args.split}", flush=True)

    tablas = []
    for case_id in casos:
        ruta = next(DATA_MHA_DIR.glob(f"*images*/{case_id}.mha"), None)
        if ruta is None:
            raise SystemExit(f"No encuentro {case_id}.mha en {DATA_MHA_DIR}/*images*/")
        volume, spacing = inferencia.leer_canonico(ruta)
        gt, _ = inferencia.leer_canonico(inferencia.ETIQUETAS_DIR / f"{case_id}.mha", es_etiqueta=True)
        for plano in planos:
            t0 = time.time()
            pred = inferencia.predecir_plano(pipeline, plano, volume, spacing, case_id=case_id)
            t = comparar_distancias(pred["instance_volume"], pred["region_by_id"], gt, spacing, case_id,
                                    iou_min=args.iou_min).assign(plano=plano)
            tablas.append(t)
            estados = t.loc[~t["gt_es_principal"], "estado"].value_counts().to_dict()
            print(f"  {case_id} {plano:8s} {len(pred['fragmentos'])} fragmentos predichos | conminutos {estados} "
                  f"| {time.time() - t0:.0f} s", flush=True)
            del pred
        del volume, gt

    tabla = pd.concat(tablas, ignore_index=True)
    resumen = resumir(tabla)
    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA_DIR / f"distancias_{args.split}.csv", index=False)
    resumen.to_csv(SALIDA_DIR / f"resumen_{args.split}.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(resumen.round(2).to_string(index=False))
    print("Guardado en", (SALIDA_DIR / f"distancias_{args.split}.csv").relative_to(ROOT))


if __name__ == "__main__":
    main()
