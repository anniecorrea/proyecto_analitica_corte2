"""Frontend del proyecto PENGWIN. Desde la raíz del repositorio:

    streamlit run app/streamlit_app.py
"""
import streamlit as st

st.set_page_config(page_title="PENGWIN · Fracturas pélvicas", page_icon="🦴", layout="wide")

pagina = st.navigation([
    st.Page("paginas/inicio.py", title="Inicio", icon="🏠", default=True),
    st.Page("paginas/prediccion.py", title="Predicción de fragmentos", icon="🦴"),
    st.Page("paginas/dashboards.py", title="Dashboards", icon="📊"),
])
pagina.run()
