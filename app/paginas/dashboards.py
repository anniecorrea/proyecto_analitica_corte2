"""Dashboard integrado: métricas de los modelos por plano en test, entrenamiento, comparación con SAM y latencia.
Todo se lee de resultados/ (lo generan src/evaluacion.py --split test, los cuadernos y src/medir_latencia.py --split
test); aquí no se recalcula nada."""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rutas import METRICAS_DIR

COLOR_MODELO, COLOR_SAM = "#2B8CBE", "#9E9E9E"
PLANOS = {"axial": "Axial", "coronal": "Coronal", "sagital": "Sagital"}
COMANDO_EVALUACION = ("python -m src.evaluacion --plano todos --split test --nms-function src.deteccion_grid:nms_por_clase "
                      "--sam-checkpoint resultados/modelos/modelos_sam/sam_vit_b_01ec64.pth")


def leer_json(ruta):
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.is_file() else None


def leer_csv(ruta, **kw):
    return pd.read_csv(ruta, **kw) if ruta.is_file() else None


def layout(fig, alto=340, **kw):
    fig.update_layout(height=alto, margin={"l": 10, "r": 10, "t": 30, "b": 10},
                      legend={"orientation": "h", "y": -0.2}, **kw)
    return fig


st.title("Dashboard de resultados")
st.caption("Resultados sobre el conjunto de **test** (20 casos que no se usaron para entrenar ni para elegir modelos o "
           "umbrales), evaluado una sola vez con todo congelado.")
# src/evaluacion.py --plano todos --split test: una carpeta por modelo
evaluaciones = {p: leer_json(METRICAS_DIR / "evaluacion" / p / "metricas_test.json") for p in PLANOS}
evaluaciones = {p: e for p, e in evaluaciones.items() if e is not None}

resumen, entrenamiento, sam, latencia = st.tabs(["Resumen", "Entrenamiento", "Modelo vs SAM", "Latencia"])

METRICAS_CONSIGNA = [("Clasificación F1", "classification_f1_macro"), ("Clasificación AUC", "classification_auc_macro"),
                     ("Detección IoU", "bbox_mean_iou"), ("Detección mAP@0.50", "bbox_map50"),
                     ("Detección mAP@[.50:.95]", "bbox_map50_95"), ("Segmentación Dice", "modelo_fragment_dice"),
                     ("Segmentación IoU", "modelo_fragment_iou")]

# ---------------------------------------------------------------------------------------------------------------
with resumen:
    if not evaluaciones:
        st.info(f"Falta la evaluación en test: `{COMANDO_EVALUACION}`.")
    else:
        plano_eval = st.radio("Modelo", list(evaluaciones), format_func=PLANOS.get, horizontal=True,
                              key="plano_resumen")
        evaluacion = evaluaciones[plano_eval]
        st.caption(f"Cortes del plano {PLANOS[plano_eval].lower()} ({evaluacion['n_slices']} cortes de "
                   f"{evaluacion['n_cases']} casos), preparados igual que en el entrenamiento del modelo.")
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

# ---------------------------------------------------------------------------------------------------------------
with entrenamiento:
    st.caption("El frontend usa un modelo por plano anatómico. Los tres comparten arquitectura (backbone + CBAM + "
               "detección + decoder) y entrada 2.5D; el sagital añade un canal con la posición lateral del corte.")
    plano = st.radio("Plano", list(PLANOS), format_func=PLANOS.get, horizontal=True, key="plano_entrenamiento")
    hist = leer_csv(METRICAS_DIR / f"entrenamiento_multitarea_{plano}" / "historial_completo.csv")
    if hist is None:
        st.info(f"Falta el historial de entrenamiento del modelo {PLANOS[plano].lower()}.")
    else:
        st.subheader(f"Pérdida de entrenamiento: modelo {PLANOS[plano].lower()}")
        hist = hist.reset_index(drop=True)
        hist["epoca_global"] = hist.index + 1
        etapas = {"A_deteccion": "A: detección", "B_decoder": "B: decoder", "C_conjunto": "C: conjunto"}
        st.caption("Entrenamiento en tres etapas sobre el mismo backbone: A entrena la detección, B el decoder de "
                   "instancias con el backbone congelado y C ajusta todo junto.")
        fig = go.Figure([go.Scatter(x=hist["epoca_global"], y=hist["train_total"], name="Pérdida total (train)",
                                    line={"color": "#555"})])
        for etapa, nombre in etapas.items():
            ep = hist.loc[hist["stage"] == etapa, "epoca_global"]
            if len(ep):
                fig.add_vrect(x0=ep.min() - 0.5, x1=ep.max() + 0.5, line_width=0, fillcolor="#888",
                              opacity=0.10 if etapa == "B_decoder" else 0.0,
                              annotation_text=nombre, annotation_position="top left")
        st.plotly_chart(layout(fig, xaxis_title="Época", hovermode="x unified").update_yaxes(type="log"),
                        width="stretch")

# ---------------------------------------------------------------------------------------------------------------
with sam:
    con_sam = {p: e for p, e in evaluaciones.items() if e.get("sam") == "evaluated"}
    if not con_sam:
        st.info(f"Falta la comparación con SAM en test: `{COMANDO_EVALUACION}`.")
    else:
        st.caption("SAM es solo una referencia (zero-shot): recibe como prompt las mismas cajas que predice nuestro "
                   "detector en cada corte de test, y se compara la máscara de cada uno contra la anotación real "
                   "(Dice e IoU por región anatómica, promedio de las tres regiones). "
                   f"SAM {next(iter(con_sam.values()))['sam_model_type']}.")
        plano_sam = st.radio("Modelo", list(con_sam), format_func=PLANOS.get, horizontal=True, key="plano_sam")
        e = con_sam[plano_sam]
        c = st.columns(4)
        c[0].metric("Dice modelo propio", f"{e['modelo_dice_macro']:.3f}", border=True)
        c[1].metric("Dice SAM", f"{e['sam_dice_macro']:.3f}", f"{e['sam_dice_macro'] - e['modelo_dice_macro']:+.3f}",
                    border=True)
        c[2].metric("IoU modelo propio", f"{e['modelo_iou_macro']:.3f}", border=True)
        c[3].metric("IoU SAM", f"{e['sam_iou_macro']:.3f}", f"{e['sam_iou_macro'] - e['modelo_iou_macro']:+.3f}",
                    border=True)
        izq, der = st.columns(2)
        nombres = [PLANOS[p] for p in con_sam]
        for col, metrica, titulo in ((izq, "dice", "Dice"), (der, "iou", "IoU")):
            fig = go.Figure([go.Bar(name="Modelo propio", x=nombres,
                                    y=[v[f"modelo_{metrica}_macro"] for v in con_sam.values()], marker_color=COLOR_MODELO),
                             go.Bar(name="SAM", x=nombres, y=[v[f"sam_{metrica}_macro"] for v in con_sam.values()],
                                    marker_color=COLOR_SAM)])
            col.markdown(f"**{titulo} por modelo**")
            col.plotly_chart(layout(fig, barmode="group", yaxis_range=[0, 1]), width="stretch")
        lat_sam = leer_json(METRICAS_DIR / "evaluacion" / plano_sam / "latencia_test_cuda.json")
        if lat_sam and "sam_ms_mean" in lat_sam:
            st.write(f"En GPU, SAM tarda **{lat_sam['sam_ms_mean']:.0f} ms** por corte frente a "
                     f"**{lat_sam['pipeline_ms_mean']:.0f} ms** del pipeline propio completo "
                     f"({lat_sam['sam_ms_mean'] / lat_sam['pipeline_ms_mean']:.0f} veces más).")

# ---------------------------------------------------------------------------------------------------------------
with latencia:
    lat = leer_json(METRICAS_DIR / "latencia" / "latencia_test.json")      # src/medir_latencia.py --split test
    if lat is None:
        st.info("Aún no hay medición. Ejecuta `python -m src.medir_latencia --split test` en una máquina con los datos.")
    else:
        r0 = lat["resultados"][0]
        st.caption(f"Por imagen: promedio de {r0['casos_imagen']} casos de test ({r0['imagenes_por_repeticion']} cortes "
                   f"cada uno). Por tomografía: promedio de los casos {', '.join(r0['casos_volumen'])}. La latencia por "
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
        por_caso = {r["dispositivo"]: r["ms_por_imagen_por_caso"] for r in lat["resultados"]}
        casos_lat = list(next(iter(por_caso.values())))
        fig = go.Figure([go.Bar(name=d.upper(), x=casos_lat, y=[v[c] for c in casos_lat],
                                marker_color=COLOR_MODELO if d == "cuda" else COLOR_SAM)
                         for d, v in por_caso.items()])
        st.markdown("**Latencia por imagen en cada caso**")
        st.plotly_chart(layout(fig, barmode="group", xaxis_title="Caso", yaxis_title="ms por imagen",
                               xaxis_type="category"), width="stretch")
