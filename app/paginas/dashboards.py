"""Dashboard integrado: métricas del modelo en validación, comparación con SAM, salida 3D, distancias y latencia.
Todo se lee de resultados/ (lo generan los cuadernos y src/medir_latencia.py); aquí no se recalcula nada."""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rutas import DATOS_DIR, FIGURAS_DIR, METRICAS_DIR

COLOR_REGION = {"sacro": "#D9A62E", "coxal_izquierdo": "#2B8CBE", "coxal_derecho": "#E34A33"}
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
evaluacion = leer_json(METRICAS_DIR / "evaluacion" / "metricas_val.json")

resumen, entrenamiento, sam, salida3d, latencia, casos = st.tabs(
    ["Resumen", "Entrenamiento", "Modelo vs SAM", "Fragmentos 3D y distancias", "Latencia", "Casos difíciles"])

# ---------------------------------------------------------------------------------------------------------------
with resumen:
    if evaluacion is None:
        st.info("Falta resultados/metricas/evaluacion/metricas_val.json (src/evaluacion.py).")
    else:
        st.subheader("Objetivos de la consigna")
        st.caption("Flecha verde: el valor supera el objetivo; roja: queda por debajo.")
        metricas = [("Clasificación F1", "classification_f1_macro"), ("Clasificación AUC", "classification_auc_macro"),
                    ("Detección IoU", "bbox_mean_iou"), ("Detección mAP@0.50", "bbox_map50"),
                    ("Detección mAP@[.50:.95]", "bbox_map50_95"), ("Segmentación Dice", "modelo_fragment_dice"),
                    ("Segmentación IoU", "modelo_fragment_iou")]
        for fila in (metricas[:4], metricas[4:]):
            cols = st.columns(4)
            for col, (nombre, clave) in zip(cols, fila):
                valor, objetivo = evaluacion[clave], evaluacion["targets"][clave]["minimum"]
                col.metric(nombre, f"{valor:.3f}", f"{valor - objetivo:+.3f} vs objetivo {objetivo:.2f}",
                           border=True)
        st.caption("Dice e IoU de segmentación son por fragmento 2D (cada fragmento real contra su mejor "
                   f"predicción en el corte). Por región anatómica, el Dice es {evaluacion['modelo_dice_macro']:.3f} "
                   f"y el IoU {evaluacion['modelo_iou_macro']:.3f}.")

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
    hist = leer_csv(METRICAS_DIR / "entrenamiento_multitarea" / "historial_completo.csv")
    if hist is None:
        st.info("Falta el historial de entrenamiento.")
    else:
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

    st.subheader("Distancia de separación: predicción frente a referencia")
    dist = leer_csv(METRICAS_DIR / "salida_3d" / "distancias_pred_vs_gt_val.csv", dtype={"caso": str})
    conminutos = None if dist is None else dist[~dist["gt_is_main"]]
    if dist is not None:
        c = st.columns(3)
        c[0].metric("Fragmentos emparejados", len(dist), help="Fragmentos reales con un predicho de IoU 3D ≥ 0.5",
                    border=True)
        c[1].metric("Fragmento principal correcto", f"{dist['main_ok'].mean():.0%}", border=True)
        c[2].metric("Error medio en conminutos", f"{conminutos['abs_error_mm'].mean():.1f} mm" if len(conminutos)
                    else "sin datos", border=True)
        if not len(conminutos):
            st.warning("En validación solo se emparejan los fragmentos principales, cuya distancia es 0 mm por "
                       "definición. Los fragmentos conminutos pequeños no se recuperan o se fusionan con el principal "
                       "(contacto en la superficie de fractura), así que su distancia todavía no se puede comparar "
                       "con la referencia. Abajo, la distribución real de esas distancias.")
    izq, der = st.columns(2)
    if conminutos is not None and len(conminutos):
        fig = go.Figure(go.Scatter(x=conminutos["gt_distance_mm"], y=conminutos["pred_distance_mm"], mode="markers",
                                   marker={"color": conminutos["region"].map(COLOR_REGION), "size": 9},
                                   text=conminutos["caso"], hovertemplate="Caso %{text}<br>real %{x:.1f} mm<br>"
                                                                          "predicha %{y:.1f} mm<extra></extra>"))
        tope = max(1.0, float(conminutos[["gt_distance_mm", "pred_distance_mm"]].max().max()))
        fig.add_shape(type="line", x0=0, y0=0, x1=tope, y1=tope, line={"dash": "dot", "color": "#888"})
        izq.markdown("**Distancia predicha frente a la real (conminutos)**")
        izq.plotly_chart(layout(fig, xaxis_title="Distancia real (mm)", yaxis_title="Distancia predicha (mm)"),
                         width="stretch")
    ref = leer_csv(DATOS_DIR / "distancias_referencia.csv", dtype={"case_id": str})
    if ref is not None:
        conm = ref[~ref["is_main"]]
        fig = go.Figure([go.Histogram(x=conm.loc[conm["region"] == r, "distance_mm"], name=NOMBRE_REGION[r],
                                      marker_color=COLOR_REGION[r], nbinsx=30) for r in NOMBRE_REGION])
        destino = der if conminutos is not None and len(conminutos) else izq
        destino.markdown(f"**Distancia real de los fragmentos conminutos** ({len(conm)} fragmentos, "
                         f"{ref['case_id'].nunique()} casos)")
        destino.plotly_chart(layout(fig, barmode="stack", xaxis_title="Distancia al fragmento principal (mm)",
                                    yaxis_title="Fragmentos"), width="stretch")

# ---------------------------------------------------------------------------------------------------------------
with latencia:
    lat = leer_json(METRICAS_DIR / "latencia" / "latencia.json")
    if lat is None:
        st.info("Aún no hay medición. Ejecuta `python -m src.medir_latencia` en una máquina con los datos.")
    else:
        st.caption(f"Caso {lat['caso']} completo con el pipeline del frontend ({lat['creado']}). La latencia por "
                   "imagen es un corte a la vez: preprocesamiento, backbone con CBAM, detección con NMS y decoder.")
        cols = st.columns(len(lat["resultados"]))
        for col, r in zip(cols, lat["resultados"]):
            with col.container(border=True):
                st.markdown(f"**{r['dispositivo'].upper()}** · {r['hardware']}")
                a, b = st.columns(2)
                a.metric("Por imagen", f"{r['ms_por_imagen']:.0f} ms")
                b.metric("Por tomografía", f"{r['s_por_volumen']:.0f} s", help=f"{r['cortes_volumen']} cortes")
                extra = "precisión mixta (AMP)" if r["precision_mixta"] else f"{r['hilos_cpu']} hilos, float32"
                st.caption(f"{extra}; media de {r['repeticiones']} repeticiones de {r['imagenes_por_repeticion']} "
                           "imágenes.")
        if len(lat["resultados"]) > 1:
            gpu = next(r for r in lat["resultados"] if r["dispositivo"] == "cuda")
            cpu = next(r for r in lat["resultados"] if r["dispositivo"] == "cpu")
            st.write(f"En GPU cada imagen es **{cpu['ms_por_imagen'] / gpu['ms_por_imagen']:.0f} veces** más rápida "
                     "que en CPU.")

# ---------------------------------------------------------------------------------------------------------------
with casos:
    st.caption("Cortes de validación con menor Dice: a la izquierda la anotación real y a la derecha la predicción. "
               "Suelen ser cortes de los extremos de la pelvis con fragmentos de muy pocos píxeles.")
    peores = sorted((FIGURAS_DIR / "evaluacion").glob("val_peor*.png"))
    aciertos = sorted((FIGURAS_DIR / "evaluacion").glob("val_acierto*.png"))
    if peores:
        st.subheader("Cinco peores cortes")
        for img in peores:
            _, caso, _, corte = img.stem.split("_", 1)[1].split("_")
            st.image(str(img), caption=f"Caso {caso}, corte {int(corte)}", width="stretch")
    if aciertos:
        with st.expander("Mejores cortes"):
            for img in aciertos:
                st.image(str(img), width="stretch")
