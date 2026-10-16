"""Dashboard integrado: métricas del modelo en validación, entrenamiento de los modelos por plano, comparación con SAM,
salida 3D y latencia.
Todo se lee de resultados/ (lo generan los cuadernos y src/medir_latencia.py); aquí no se recalcula nada."""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rutas import METRICAS_DIR

NOMBRE_REGION = {"sacro": "Sacro", "coxal_izquierdo": "Coxal izquierdo", "coxal_derecho": "Coxal derecho"}
COLOR_MODELO, COLOR_SAM = "#2B8CBE", "#9E9E9E"


def leer_json(ruta):
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.is_file() else None


def leer_csv(ruta, **kw):
    return pd.read_csv(ruta, **kw) if ruta.is_file() else None


def layout(fig, alto=340, **kw):
    fig.update_layout(height=alto, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      legend={"orientation": "h", "y": -0.2}, **kw)
    return fig


st.title("Dashboard de resultados")
PLANOS = {"axial": "Axial", "coronal": "Coronal", "sagital": "Sagital"}
# src/evaluacion.py --plano todos: una carpeta por modelo; sin ellas, la evaluación del modelo anterior
evaluaciones = {p: leer_json(METRICAS_DIR / "evaluacion" / p / "metricas_val.json") for p in PLANOS}
evaluaciones = {p: e for p, e in evaluaciones.items() if e is not None}
anterior = leer_json(METRICAS_DIR / "evaluacion" / "metricas_val.json")

resumen, entrenamiento, sam, salida3d, latencia = st.tabs(
    ["Resumen", "Entrenamiento", "Modelo vs SAM", "Fragmentos 3D", "Latencia"])

METRICAS_CONSIGNA = [("Clasificación F1", "classification_f1_macro"), ("Clasificación AUC", "classification_auc_macro"),
                     ("Detección IoU", "bbox_mean_iou"), ("Detección mAP@0.50", "bbox_map50"),
                     ("Detección mAP@[.50:.95]", "bbox_map50_95"), ("Segmentación Dice", "modelo_fragment_dice"),
                     ("Segmentación IoU", "modelo_fragment_iou")]

# ---------------------------------------------------------------------------------------------------------------
with resumen:
    if evaluaciones:
        plano_eval = st.radio("Modelo", list(evaluaciones), format_func=PLANOS.get, horizontal=True,
                              key="plano_resumen")
        evaluacion = evaluaciones[plano_eval]
        st.caption(f"Validación con los cortes del plano {PLANOS[plano_eval].lower()} ({evaluacion['n_slices']} cortes "
                   f"de {evaluacion['n_cases']} casos), preparados igual que en el entrenamiento del modelo.")
    else:
        evaluacion = anterior
    if evaluacion is None:
        st.info("Falta la evaluación: `python -m src.evaluacion --plano todos --split val "
                "--nms-function src.deteccion_grid:nms_por_clase`.")
    else:
        st.subheader("Objetivos de la consigna")
        st.caption("Flecha verde: el valor supera el objetivo; roja: queda por debajo.")
        for fila in (METRICAS_CONSIGNA[:4], METRICAS_CONSIGNA[4:]):
            cols = st.columns(4)
            for col, (nombre, clave) in zip(cols, fila):
                valor, objetivo = evaluacion[clave], evaluacion["targets"][clave]["minimum"]
                col.metric(nombre, f"{valor:.3f}", f"{valor - objetivo:+.3f} vs objetivo {objetivo:.2f}",
                           border=True)
        st.caption("Dice e IoU de segmentación son por fragmento 2D (cada fragmento real contra su mejor "
                   f"predicción en el corte). Por región anatómica, el Dice es {evaluacion['modelo_dice_macro']:.3f} "
                   f"y el IoU {evaluacion['modelo_iou_macro']:.3f}.")
        if len(evaluaciones) > 1:
            st.subheader("Comparación de los tres modelos")
            tabla = pd.DataFrame({PLANOS[p]: {nombre: e[clave] for nombre, clave in METRICAS_CONSIGNA}
                                  for p, e in evaluaciones.items()})
            tabla.insert(0, "Objetivo", [evaluaciones[next(iter(evaluaciones))]["targets"][clave]["minimum"]
                                         for _, clave in METRICAS_CONSIGNA])
            st.dataframe(tabla, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.3f") for c in tabla.columns})

        sal =leer_csv(METRICAS_DIR / "salida_3d" / "resultado_val_config_ajustada.csv")
        if sal is not None:
            total = sal[sal["caso"].astype(str) == "TODOS"].iloc[0]
            st.subheader("Fragmentos en 3D")
            c = st.columns(4)
            c[0].metric("Fragmentos reales", int(total["n_gt"]), border=True)
            c[1].metric("Fragmentos predichos", int(total["n_pred"]), border=True)
            c[2].metric("Precisión (IoU ≥ 0.5)", f"{total['precision@0.5']:.2f}", border=True)
            c[3].metric("F1 por fragmento", f"{total['f1@0.5']:.2f}", border=True)

# ---------------------------------------------------------------------------------------------------------------
with entrenamiento:
    planos = PLANOS
    resumen_planos = leer_csv(METRICAS_DIR / "entrenamiento_multitarea_planos" / "resumen_planos.csv")
    if resumen_planos is not None:
        st.subheader("Modelos por plano (validación)")
        st.caption("El frontend usa un modelo por plano anatómico. Los tres comparten arquitectura (backbone + CBAM + "
                   "detección + decoder) y entrada 2.5D; el sagital añade un canal con la posición lateral del corte.")
        tabla = resumen_planos.set_index("plano").reindex(list(planos)).reset_index()
        tabla = pd.DataFrame({"Plano": tabla["plano"].map(planos), "mAP@0.50": tabla["mAP50"],
                              "Dice (cajas predichas)": tabla["Dice_pred"],
                              "Recall de fragmentos": tabla["frag_recall50_pred"],
                              "Cortes vacíos que pasan": tabla["vacios_que_pasan_cal"],
                              "Época elegida": tabla["epoca_elegida"]})
        st.dataframe(tabla, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in ("mAP@0.50", "Dice (cajas predichas)", "Recall de fragmentos",
                                              "Cortes vacíos que pasan")})
    plano = st.radio("Plano", list(planos), format_func=planos.get, horizontal=True, key="plano_entrenamiento")
    hist = leer_csv(METRICAS_DIR / f"entrenamiento_multitarea_{plano}" / "historial_completo.csv")
    if hist is None:
        st.info(f"Falta el historial de entrenamiento del modelo {planos[plano].lower()}.")
    else:
        st.subheader(f"Curvas de entrenamiento: modelo {planos[plano].lower()}")
        hist = hist.reset_index(drop=True)
        hist["epoca_global"] = hist.index + 1
        etapas = {"A_deteccion": "A: detección", "B_decoder": "B: decoder", "C_conjunto": "C: conjunto"}
        st.caption("Entrenamiento en tres etapas sobre el mismo backbone: A entrena la detección, B el decoder de "
                   "instancias con el backbone congelado y C ajusta todo junto.")

        def con_etapas(fig):
            for etapa, nombre in etapas.items():
                ep = hist.loc[hist["stage"] == etapa, "epoca_global"]
                if len(ep):
                    fig.add_vrect(x0=ep.min() - 0.5, x1=ep.max() + 0.5, line_width=0, fillcolor="#888",
                                  opacity=0.10 if etapa == "B_decoder" else 0.0,
                                  annotation_text=nombre, annotation_position="top left")
            return layout(fig, xaxis_title="Época", hovermode="x unified")

        izq, der = st.columns(2)
        fig = go.Figure([go.Scatter(x=hist["epoca_global"], y=hist["val_det_map50"], name="mAP@0.50 (val)",
                                    line={"color": COLOR_MODELO}),
                         go.Scatter(x=hist["epoca_global"], y=hist["val_seg_dice_pred"], name="Dice (val)",
                                    line={"color": "#E34A33"})])
        izq.markdown("**Métricas de validación**")
        izq.plotly_chart(con_etapas(fig).update_yaxes(range=[0.6, 1.0]), width="stretch")
        fig = go.Figure([go.Scatter(x=hist["epoca_global"], y=hist["train_total"], name="Pérdida total (train)",
                                    line={"color": "#555"})])
        der.markdown("**Pérdida de entrenamiento**")
        der.plotly_chart(con_etapas(fig).update_yaxes(type="log"), width="stretch")

# ---------------------------------------------------------------------------------------------------------------
with sam:
    comp = leer_csv(METRICAS_DIR / "sam" / "comparacion_sam_vs_modelo_val.csv", index_col=0)
    if comp is None:
        st.info("Falta resultados/metricas/sam/comparacion_sam_vs_modelo_val.csv (sam_baseline.ipynb).")
    else:
        st.caption("SAM es solo una referencia (zero-shot): recibe como prompt las mismas cajas que predice "
                   "nuestro detector, y se compara la máscara de cada uno contra la anotación real.")
        modelo, base = comp.index[comp.index.str.contains("propio")][0], comp.index[comp.index.str.contains("SAM")][0]
        c = st.columns(4)
        c[0].metric("Dice modelo propio", f"{comp.loc[modelo, 'dice_medio']:.3f}", border=True)
        c[1].metric("Dice SAM", f"{comp.loc[base, 'dice_medio']:.3f}", border=True)
        c[2].metric("IoU modelo propio", f"{comp.loc[modelo, 'iou_medio']:.3f}", border=True)
        c[3].metric("IoU SAM", f"{comp.loc[base, 'iou_medio']:.3f}", border=True)
        regiones = list(NOMBRE_REGION)
        izq, der = st.columns(2)
        for col, metrica in ((izq, "dice"), (der, "iou")):
            fig = go.Figure([go.Bar(name="Modelo propio", x=[NOMBRE_REGION[r] for r in regiones],
                                    y=[comp.loc[modelo, f"{metrica}_{r}"] for r in regiones], marker_color=COLOR_MODELO),
                             go.Bar(name="SAM", x=[NOMBRE_REGION[r] for r in regiones],
                                    y=[comp.loc[base, f"{metrica}_{r}"] for r in regiones], marker_color=COLOR_SAM)])
            col.markdown(f"**{metrica.upper() if metrica == 'iou' else 'Dice'} por región**")
            col.plotly_chart(layout(fig, barmode="group", yaxis_range=[0, 1]), width="stretch")
        peores_sam = METRICAS_DIR / "sam" / "peores_sam_val.png"
        if peores_sam.is_file():
            with st.expander("Cortes donde SAM falla más"):
                st.image(str(peores_sam), width="stretch")

# ---------------------------------------------------------------------------------------------------------------
with salida3d:
    sal = leer_csv(METRICAS_DIR / "salida_3d" / "resultado_val_config_ajustada.csv", dtype={"caso": str})
    if sal is not None:
        st.subheader("Fragmentos 3D por caso")
        st.caption("Un fragmento predicho cuenta como acierto si su IoU 3D con un fragmento real es al menos 0.5. "
                   "Las fusiones son fragmentos reales que se tocan en la superficie de fractura y el modelo "
                   "une en uno solo (la limitación principal).")
        casos_df = sal[sal["caso"] != "TODOS"]
        fig = go.Figure([go.Bar(name="Reales", x=casos_df["caso"], y=casos_df["n_gt"], marker_color=COLOR_SAM),
                         go.Bar(name="Predichos", x=casos_df["caso"], y=casos_df["n_pred"], marker_color=COLOR_MODELO),
                         go.Bar(name="Fusionados", x=casos_df["caso"], y=casos_df["gt_fusionado"],
                                marker_color="#E34A33")])
        izq, der = st.columns(2)
        izq.plotly_chart(layout(fig, barmode="group", xaxis_title="Caso", yaxis_title="Fragmentos"), width="stretch")
        tabla = sal[["caso", "n_gt", "n_pred", "precision@0.5", "recall@0.5", "f1@0.5", "dice_regiones"]].rename(
            columns={"caso": "Caso", "n_gt": "Reales", "n_pred": "Predichos", "precision@0.5": "Precisión",
                     "recall@0.5": "Recall", "f1@0.5": "F1", "dice_regiones": "Dice por hueso"})
        der.dataframe(tabla, hide_index=True, width="stretch",
                      column_config={c: st.column_config.NumberColumn(format="%.2f")
                                     for c in ("Precisión", "Recall", "F1", "Dice por hueso")})

# ---------------------------------------------------------------------------------------------------------------
with latencia:
    lat = leer_json(METRICAS_DIR / "latencia" / "latencia.json")
    if lat is None:
        st.info("Aún no hay medición. Ejecuta `python -m src.medir_latencia` en una máquina con los datos.")
    else:
        st.caption(f"Caso {lat['caso']}." "La latencia por "
                   "imagen es un corte a la vez: preprocesamiento, backbone con CBAM, detección con NMS y decoder.")
        cols = st.columns(len(lat["resultados"]))
        for col, r in zip(cols, lat["resultados"]):
            with col.container(border=True):
                st.markdown(f"**{r['dispositivo'].upper()}** · {r['hardware']}")
                a, b = st.columns(2)
                a.metric("Por imagen", f"{r['ms_por_imagen']:.0f} ms")
                b.metric("Por tomografía", f"{r['s_por_volumen']:.0f} s", help=f"{r['cortes_volumen']} cortes: los tres modelos (axial, coronal y sagital)")
        if len(lat["resultados"]) > 1:
            gpu = next(r for r in lat["resultados"] if r["dispositivo"] == "cuda")
            cpu = next(r for r in lat["resultados"] if r["dispositivo"] == "cpu")
            st.write(f"En GPU cada imagen es **{cpu['ms_por_imagen'] / gpu['ms_por_imagen']:.0f} veces** más rápida "
                     "que en CPU.")
