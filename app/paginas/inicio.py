import streamlit as st

st.title("🦴 Proyecto PENGWIN")
st.markdown(
    "Detección, segmentación y clasificación de **fragmentos de fracturas pélvicas** en tomografía computarizada."
)

col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.subheader("🦴 Predicción de fragmentos")
        st.write("Sube una tomografía `.mha` y el modelo multitarea marca los fragmentos de sacro y coxales, "
                 "corte a corte y en 3D.")
        st.page_link("paginas/prediccion.py", label="Ir a la predicción", icon="➡️")
with col2:
    with st.container(border=True):
        st.subheader("📊 Dashboards")
        st.write("Métricas de entrenamiento, evaluación y comparación con SAM. *En construcción.*")
        st.page_link("paginas/dashboards.py", label="Ir a los dashboards", icon="➡️")

st.divider()
st.caption("Pipeline: tomografía → detección por grid + NMS → decoder de instancias → "
           "geometría original del MHA → asociación 3D entre cortes → limpieza de fragmentos.")
