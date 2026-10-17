"""analisis_pesos_cabezas.py -- Seccion 4.1: pesos entre cabezas (w_det, w_seg) y por que la pertenencia va aparte.
Autor: Fabian.

No entrena nada: lee lo que ya registro Entrenamiento_multitarea_planos.ipynb para cada plano y resume dos preguntas.

1. QUIEN MUEVE EL TRONCO EN LA ETAPA C (w_det = w_seg = 1)
   El cuaderno mide, una vez por epoca, la norma del gradiente que cada perdida deja en la primera convolucion del
   backbone (trunk_grad_det, trunk_grad_seg; un lote fijo de val, en eval). Si una es mucho mayor, esa tarea es la que
   ajusta el extractor compartido aunque los pesos nominales sean iguales.

2. CUANTO CAMBIA LA ETAPA C LOS RESULTADOS
   Se compara la mAP@0.50 al final de la etapa A (solo deteccion) y el Dice con priors predichos al final de la etapa
   B (solo decoder) contra lo mejor de la etapa C (conjunta). Si C apenas mueve las metricas, afinar w_det/w_seg con
   un barrido (reentrenar C varias veces por plano) tiene poco beneficio para su costo.

Salidas:
  resultados/metricas/pesos_cabezas/resumen_pesos_cabezas.csv
  resultados/figuras/pesos_cabezas/normas_gradiente_tronco.png

Uso: python -m src.analisis_pesos_cabezas
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.rutas import FIGURAS_DIR, METRICAS_DIR, MODELOS_DIR  # noqa: E402

PLANOS = ("axial", "coronal", "sagital")
OUT_CSV = METRICAS_DIR / "pesos_cabezas" / "resumen_pesos_cabezas.csv"
OUT_FIG = FIGURAS_DIR / "pesos_cabezas" / "normas_gradiente_tronco.png"
# Paleta categorica validada (slots 1 y 2); la forma del marcador es la codificacion secundaria.
COLOR_DET, COLOR_SEG = "#2a78d6", "#eb6834"
TINTA, TINTA_SECUNDARIA, REJILLA = "#1f1f1d", "#5f5e5a", "#e6e5e1"


def _historial(plano: str, etapa: str) -> pd.DataFrame:
    return pd.read_csv(METRICAS_DIR / f"entrenamiento_multitarea_{plano}" / f"historial_{etapa}.csv")


def _config(plano: str) -> dict:
    return json.loads((METRICAS_DIR / f"entrenamiento_multitarea_{plano}" / "config.json").read_text(encoding="utf-8"))


def resumen() -> pd.DataFrame:
    filas = []
    for plano in PLANOS:
        a, b, c = (_historial(plano, e) for e in ("A_deteccion", "B_decoder", "C_conjunto"))
        etapa_c = next(s for s in _config(plano)["STAGES"] if s["name"] == "C_conjunto")
        meta_path = MODELOS_DIR / "nms_pertenencia" / f"cabeza_pertenencia_{plano}_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        det, seg = c["trunk_grad_det"], c["trunk_grad_seg"]
        filas.append({
            "plano": plano,
            "w_det": etapa_c["w_det"], "w_seg": etapa_c["w_seg"],
            "grad_tronco_det_mediana": det.median(), "grad_tronco_seg_mediana": seg.median(),
            "razon_seg_det": seg.median() / det.median(),
            "grad_tronco_seg_min": seg.min(), "grad_tronco_seg_max": seg.max(),
            "map50_fin_etapa_A": a["val_det_map50"].iloc[-1], "map50_mejor_etapa_C": c["val_det_map50"].max(),
            "dice_pred_fin_etapa_B": b["val_seg_dice_pred"].iloc[-1], "dice_pred_mejor_etapa_C": c["val_seg_dice_pred"].max(),
            "pertenencia_f1_mascaras_reales_tronco_congelado": meta.get("val_f1_macro"),
        })
    tabla = pd.DataFrame(filas)
    tabla["cambio_map50_por_C"] = tabla["map50_mejor_etapa_C"] - tabla["map50_fin_etapa_A"]
    tabla["cambio_dice_por_C"] = tabla["dice_pred_mejor_etapa_C"] - tabla["dice_pred_fin_etapa_B"]
    return tabla


def figura() -> Path:
    fig, ejes = plt.subplots(1, 3, figsize=(12, 3.9), sharey=True)
    for ax, plano in zip(ejes, PLANOS):
        c = _historial(plano, "C_conjunto")
        for col, color, marca, nombre in (("trunk_grad_det", COLOR_DET, "o", "Deteccion"),
                                          ("trunk_grad_seg", COLOR_SEG, "s", "Segmentacion")):
            ax.plot(c["epoch"], c[col], color=color, lw=2, marker=marca, ms=6, markeredgecolor="white",
                    markeredgewidth=1.2, label=nombre)
            ax.annotate(nombre, (c["epoch"].iloc[-1], c[col].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=TINTA_SECUNDARIA)
        ax.set_yscale("log")
        ax.set_title(plano.capitalize(), loc="left", fontsize=11, color=TINTA, fontweight="bold")
        ax.set_xlabel("Epoca de la etapa C", fontsize=9, color=TINTA_SECUNDARIA)
        ax.set_xlim(0.5, c["epoch"].max() + 3.2)
        ax.set_xticks(range(1, int(c["epoch"].max()) + 1, 3))
        ax.grid(axis="y", which="major", color=REJILLA, lw=0.8)
        ax.tick_params(colors=TINTA_SECUNDARIA, labelsize=8.5, length=0)
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)
        ax.spines["bottom"].set_color(REJILLA)
    ejes[0].set_ylabel("Norma del gradiente en backbone.conv1\n(escala logaritmica)", fontsize=9, color=TINTA_SECUNDARIA)
    ejes[0].legend(loc="lower left", fontsize=8.5, frameon=False)
    fig.suptitle("Quien ajusta el tronco compartido en la etapa C (w_det = w_seg = 1)", x=0.01, ha="left",
                 fontsize=12.5, color=TINTA, fontweight="bold")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return OUT_FIG


def main() -> None:
    tabla = resumen()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(OUT_CSV, index=False)
    print(tabla.round(3).T.to_string(header=False))
    print(f"\nTabla: {OUT_CSV}\nFigura: {figura()}")


if __name__ == "__main__":
    main()
