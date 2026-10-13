import streamlit as st

st.title("📊 Dashboards")
st.warning("🚧 Esta sección está en construcción. Aquí irán los tableros con los resultados del proyecto.", icon="🚧")

TABLEROS = {
    "Entrenamiento": "Curvas de pérdida y métricas por época de las etapas A (detección), B (decoder) y C (conjunto).",
    "Detección y segmentación": "F1/AUC por región, Dice/IoU por fragmento y ejemplos de aciertos y errores en val/test.",
    "Comparación con SAM": "Dice por corte del modelo propio frente a SAM usando las mismas cajas como prompts.",
    "Salida 3D y distancias": "Fragmentos 3D por caso, eventos de fusión/división y distancias al fragmento principal.",
}

for pestaña, (nombre, descripcion) in zip(st.tabs(list(TABLEROS)), TABLEROS.items()):
    with pestaña:
        with st.container(border=True):
            st.subheader(f"🚧 {nombre}")
            st.write(descripcion)
            st.caption("Próximamente.")
