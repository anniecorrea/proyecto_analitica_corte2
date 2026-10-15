"""Frontend del proyecto PENGWIN. Desde la raíz del repositorio:

    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for _d in (ROOT / "src", ROOT):            # las páginas importan `rutas`, `inferencia`... sin prefijo
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

AVISO = ("Uso exclusivamente académico. Este sistema no es un dispositivo médico, no está validado clínicamente y "
         "no debe usarse para tomar decisiones quirúrgicas reales.")

st.set_page_config(page_title="PENGWIN - Fracturas pélvicas", layout="wide")

pagina = st.navigation([
    st.Page("paginas/prediccion.py", title="Inicio", default=True),
    st.Page("paginas/dashboards.py", title="Dashboard de resultados"),
])
st.warning(AVISO)
pagina.run()
