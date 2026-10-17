"""Verificación en val de los tres modelos por plano: control de "pelvis presente" y fragmentos conminutos.

Desde la raíz del repositorio, con los datos en data_mha/ y los pesos descargados (git lfs pull):

    python -m src.verificar_planos_val

PARTE 1. CONTROL DE PELVIS (src/control_pelvis.py)
  El control se calibró con el modelo axial anterior. Aquí se prueba con los tres modelos que se entregan.
  - Con pelvis: cada volumen completo de val debe dar "presente".
  - Sin pelvis: se simula el caso del profesor (un .mha sin pelvis) recortando cada volumen en z a los cortes que
    quedan ANTES y DESPUÉS de la pelvis según la etiqueta. Cada recorte pasa por el pipeline completo de la app
    (inferencia del plano sobre el recorte) y debe dar "no presente". Se anota además cuántos fragmentos habría
    entregado la salida 3D sin el control.
  Se guardan los scores por corte, para poder recalibrar det_gate / MIN_MM sin volver a correr nada.

PARTE 2. CONMINUTOS (diagnóstico, no ajuste)
  Con la configuración congelada casi todos los conminutos salen fundidos con el principal de su hueso
  (resultados/metricas/distancias_pred_vs_gt/). Se reutiliza la misma inferencia de la parte 1 y solo se repite el
  armado 3D cambiando una cosa a la vez, para ver qué paso los funde:
  - actual:          config_3d.json tal cual.
  - sin_absorcion:   absorb_max_voxels = 0 (hoy se absorbe todo trozo < 100 000 vóxeles que toque a uno mayor).
  - centros_finos:   center_threshold 0.25 y min_center_distance 7 (los valores iniciales de salida_3d, antes del
                     ajuste; hoy 0.55 y 26 px, que separan menos instancias dentro de un mismo corte).
  - ambos:           las dos cosas.
  Son 5 casos y 12 conminutos: sirve para saber la causa, no para elegir una configuración nueva.

Salidas en resultados/metricas/verificacion_planos/:
  control_pelvis_val.csv     una fila por (caso, plano, estudio): decisión del control, racha en mm, esperado, ok
  scores_cortes_val.csv      mejor score por corte de cada estudio (para recalibrar)
  conminutos_val.csv         una fila por (caso, plano, variante, fragmento real), formato de distancias_pred_vs_gt
  resumen_conminutos_val.csv una fila por (plano, variante)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _d in (ROOT / "src", ROOT / "app"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import control_pelvis  # noqa: E402
import cortes_vacios  # noqa: E402
import planos as planos_mod  # noqa: E402
from distancias_pred_vs_gt import comparar_distancias, resumir  # noqa: E402
from rutas import DATA_MHA_DIR, DATASET_CSV, METRICAS_DIR  # noqa: E402

SALIDA_DIR = METRICAS_DIR / "verificacion_planos"
VARIANTES = {
    "actual": ({}, {}),
    "sin_absorcion": ({}, {"absorb_max_voxels": 0}),
    "centros_finos": ({"center_threshold": 0.25, "min_center_distance": 7}, {}),
    "ambos": ({"center_threshold": 0.25, "min_center_distance": 7}, {"absorb_max_voxels": 0}),
}


def armar(inferencia, pipeline, bundle, plano, spacing, dec_cfg, post_cfg):
    """Salida 3D de un bundle ya inferido (igual que predecir_plano, sin el control): volumen canónico + regiones."""
    ns, cfg = pipeline["ns"], pipeline["cfg_3d"]
    decoded = inferencia._decodificar(ns, bundle, plano, spacing, dec_cfg)
    result = ns["assemble_case"](bundle, dec_cfg, cfg["link"], post_cfg, stride=1, decoded=decoded)
    inst = inferencia._a_canonico(result["volume"], plano)
    tabla = result["table"]
    region_by_id = {int(f): ns["REGION_OF_CLASS"][int(c)] for f, c in zip(tabla["fragment_id"], tabla["class"])}
    return inst, region_by_id, len(tabla), result["log"]


def fila_control(caso, plano, estudio, esperado, mm_estudio, pelvis, n_frag_sin_control, segundos):
    return {"caso": caso, "plano": plano, "estudio": estudio, "largo_estudio_mm": round(mm_estudio, 1),
            "esperado": "presente" if esperado else "no presente",
            "decision": "presente" if pelvis["presente"] else "no presente",
            "ok": pelvis["presente"] == esperado, "racha_mm": round(pelvis["racha_mm"], 1),
            "racha_cortes": pelvis["racha_cortes"], "cortes_activos": pelvis["cortes_activos"],
            "n_cortes": pelvis["n_cortes"], "fragmentos_sin_control": n_frag_sin_control,
            "segundos": round(segundos, 1)}


def filas_scores(caso, plano, estudio, scores):
    return pd.DataFrame({"caso": caso, "plano": plano, "estudio": estudio, "corte": np.arange(len(scores)),
                         "score_max": np.round(scores, 4)})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--casos", nargs="*", help="IDs concretos (por defecto, todos los de val)")
    parser.add_argument("--plano", default="todos", choices=["todos", "axial", "coronal", "sagital"])
    parser.add_argument("--sin-conminutos", action="store_true", help="solo la parte 1 (control de pelvis)")
    args = parser.parse_args()

    import inferencia                                    # torch y los cuadernos solo hacen falta aquí

    dataset = pd.read_csv(DATASET_CSV, dtype={"case_id": str})
    casos = args.casos or dataset.loc[dataset["split"] == "val", "case_id"].tolist()
    planos = inferencia.PLANOS if args.plano == "todos" else (args.plano,)
    pipeline = inferencia.cargar_pipeline()
    ns, cfg = pipeline["ns"], pipeline["cfg_3d"]
    print(f"Pipeline en {pipeline['dispositivo']} | configuración 3D {pipeline['config_3d_creada']} | "
          f"{len(casos)} casos de val | planos {', '.join(planos)}", flush=True)

    control, scores, conminutos = [], [], []
    t_total = time.time()
    for caso in casos:
        ruta = next(DATA_MHA_DIR.glob(f"*images*/{caso}.mha"), None)
        if ruta is None:
            raise SystemExit(f"No encuentro {caso}.mha en {DATA_MHA_DIR}/*images*/")
        volume, spacing = inferencia.leer_canonico(ruta)
        gt, _ = inferencia.leer_canonico(inferencia.ETIQUETAS_DIR / f"{caso}.mha", es_etiqueta=True)
        sz = float(spacing[2])
        tramos = cortes_vacios.tramos_sin_pelvis(gt.reshape(gt.shape[0], -1).any(axis=1))

        for plano in planos:
            sp_eje = planos_mod.slice_axis_spacing(plano, spacing)

            # ---- 1a. volumen completo (con pelvis) + 2. variantes de armado sobre la MISMA inferencia
            t0 = time.time()
            bundle = inferencia.inferir_plano(pipeline, plano, volume, spacing)
            s = control_pelvis.scores_de_bundle(bundle)
            pelvis = control_pelvis.pelvis_presente(s, sp_eje)
            scores.append(filas_scores(caso, plano, "completo", s))
            n_frag = None
            if not args.sin_conminutos:
                for variante, (dec_extra, post_extra) in VARIANTES.items():
                    inst, region_by_id, n_frag_v, log = armar(inferencia, pipeline, bundle, plano, spacing,
                                                              {**cfg["decode"], **dec_extra},
                                                              {**cfg["post"], **post_extra})
                    n_frag = n_frag_v if variante == "actual" else n_frag
                    t = comparar_distancias(inst, region_by_id, gt, spacing, caso)
                    conminutos.append(t.assign(plano=plano, variante=variante, fragmentos_predichos=n_frag_v,
                                               absorbidos=log.get("absorbidos", 0)))
                    estados = t.loc[~t["gt_es_principal"], "estado"].value_counts().to_dict()
                    print(f"    {caso} {plano:8s} {variante:14s} {n_frag_v:2d} fragmentos | conminutos {estados}",
                          flush=True)
                    del inst
            control.append(fila_control(caso, plano, "completo", True, volume.shape[0] * sz, pelvis, n_frag,
                                        time.time() - t0))
            print(f"  {caso} {plano:8s} completo        racha {pelvis['racha_mm']:6.1f} mm -> "
                  f"{'presente' if pelvis['presente'] else 'NO PRESENTE'} | {time.time() - t0:.0f} s", flush=True)
            del bundle

            # ---- 1b. recortes sin pelvis (el estudio que traería el profesor)
            for posicion, ini, fin in tramos:
                t0 = time.time()
                recorte = np.ascontiguousarray(volume[ini:fin])
                estudio = f"sin_pelvis_{posicion}"
                try:
                    bundle = inferencia.inferir_plano(pipeline, plano, recorte, spacing)
                except Exception as exc:                      # un recorte muy corto no debe tumbar la corrida
                    print(f"  {caso} {plano:8s} {estudio:15s} ERROR: {exc!r}", flush=True)
                    control.append({"caso": caso, "plano": plano, "estudio": estudio,
                                    "largo_estudio_mm": round((fin - ini) * sz, 1), "esperado": "no presente",
                                    "decision": f"error: {exc!r}", "ok": False})
                    continue
                s = control_pelvis.scores_de_bundle(bundle)
                pelvis = control_pelvis.pelvis_presente(s, sp_eje)
                # cuántos fragmentos habría entregado la app si no existiera el control
                try:
                    _, _, n_frag, _ = armar(inferencia, pipeline, bundle, plano, spacing, cfg["decode"], cfg["post"])
                except Exception as exc:                      # p. ej. ningún corte activo: nada que armar
                    n_frag = 0 if not (s >= cfg["decode"]["det_gate"]).any() else f"error: {exc!r}"
                scores.append(filas_scores(caso, plano, estudio, s))
                control.append(fila_control(caso, plano, estudio, False, (fin - ini) * sz, pelvis, n_frag,
                                            time.time() - t0))
                print(f"  {caso} {plano:8s} {estudio:15s} ({(fin - ini) * sz:5.0f} mm) racha {pelvis['racha_mm']:6.1f} mm"
                      f" -> {'PRESENTE' if pelvis['presente'] else 'no presente'} | sin control: {n_frag} fragmentos",
                      flush=True)
                del bundle, recorte
        del volume, gt

    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    control = pd.DataFrame(control)
    control.to_csv(SALIDA_DIR / "control_pelvis_val.csv", index=False)
    pd.concat(scores, ignore_index=True).to_csv(SALIDA_DIR / "scores_cortes_val.csv", index=False)

    print("\n=== PARTE 1: control de pelvis (MIN_MM = {:.0f} mm, det_gate = {}) ===".format(
        control_pelvis.MIN_MM, control_pelvis.DET_GATE))
    for plano, g in control.groupby("plano", sort=False):
        con = g[g["esperado"] == "presente"]
        sin = g[g["esperado"] == "no presente"]
        print(f"  {plano:8s} con pelvis: {int(con['ok'].sum())}/{len(con)} aceptados "
              f"(racha mínima {con['racha_mm'].min():.0f} mm) | sin pelvis: {int(sin['ok'].sum())}/{len(sin)} rechazados "
              f"(racha máxima {sin['racha_mm'].max():.0f} mm)")
    print("  RESULTADO:", "TODO OK" if control["ok"].all() else "HAY FALLAS (ver control_pelvis_val.csv)")

    if conminutos:
        tabla = pd.concat(conminutos, ignore_index=True)
        tabla.to_csv(SALIDA_DIR / "conminutos_val.csv", index=False)
        resumen = []
        for variante, t in tabla.groupby("variante", sort=False):
            r = resumir(t)
            r.insert(1, "variante", variante)
            n_pred = t.drop_duplicates(["caso", "plano"]).groupby("plano", sort=False)["fragmentos_predichos"].sum()
            r["fragmentos_predichos"] = r["plano"].map(n_pred)
            r["fragmentos_reales"] = r["plano"].map(t.groupby("plano", sort=False).size())
            resumen.append(r)
        resumen = pd.concat(resumen, ignore_index=True).sort_values(["plano", "variante"], key=lambda c: c.map(
            {**{p: i for i, p in enumerate(planos)}, **{v: i for i, v in enumerate(VARIANTES)}}))
        resumen.to_csv(SALIDA_DIR / "resumen_conminutos_val.csv", index=False)
        cols = ["plano", "variante", "fragmentos_reales", "fragmentos_predichos", "principales_ok", "conminutos",
                "n_separado", "n_fusionado", "n_omitido"]
        print("\n=== PARTE 2: conminutos según el armado 3D ===")
        print(resumen[cols].to_string(index=False))

    print(f"\nGuardado en {SALIDA_DIR.relative_to(ROOT)}/ | {(time.time() - t_total) / 60:.0f} min en total")


if __name__ == "__main__":
    main()
