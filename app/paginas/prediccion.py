import gc
import hashlib

import numpy as np
import streamlit as st

import inferencia
import visualizacion as vis

st.title("🦴 Predicción de fragmentos")
st.write("Sube una tomografía `.mha` (la imagen, no la etiqueta). El modelo recorre todos los cortes, detecta y "
         "segmenta los fragmentos y los une en 3D. No usa ninguna anotación real.")


@st.cache_resource(show_spinner="Cargando el modelo...")
def obtener_pipeline():
    return inferencia.cargar_pipeline()


try:
    pipeline = obtener_pipeline()
except Exception as e:                    # pesos sin `git lfs pull`, cuaderno sin una definición, etc.
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()

info = pipeline["info"]
colores = pipeline["ns"]["TARGET_COLORS"]
regiones = pipeline["ns"]["REGION_OF_CLASS"]
with st.sidebar:
    st.subheader("Modelo")
    st.caption(f"`{info['modelo']}` · etapa {info['etapa']}, época {info['epoca']} · "
               f"{info['canales_entrada']} canal(es) · {info['dispositivo']}")
    st.caption(f"Configuración 3D ajustada con val ({info['config_3d_creada']})")
    st.subheader("Leyenda")
    for c, nombre in regiones.items():
        st.markdown(f"<span style='color:{colores[c]}'>■</span> {nombre.replace('_', ' ')}", unsafe_allow_html=True)

# ---------------------------------------------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------------------------------------------
locales = inferencia.casos_locales()
fuente = st.radio("Origen de la tomografía", ["Subir archivo .mha", "Caso local de data_mha/"], horizontal=True,
                  disabled=not locales)
if fuente == "Subir archivo .mha":
    archivo = st.file_uploader("Tomografía (.mha)", type=["mha"])
    origen, nombre = (archivo.getvalue(), archivo.name) if archivo else (None, None)
else:
    ruta = st.selectbox("Caso", locales, format_func=lambda p: f"{p.parent.name}/{p.name}")
    origen, nombre = ruta, ruta.name

clave = None
if origen is not None:
    clave = hashlib.md5(origen).hexdigest() if isinstance(origen, bytes) else str(origen)

if st.button("Predecir fragmentos", type="primary", disabled=origen is None):
    st.session_state.pop("prediccion", None)
    gc.collect()
    try:
        with st.spinner("Leyendo la tomografía..."):
            volume_hu, image_itk = inferencia.leer_mha(origen)
        with st.spinner(f"Prediciendo {volume_hu.shape[0]} cortes en {info['dispositivo']}. "
                        "Puede tardar unos minutos..."):
            res = inferencia.predecir_volumen(pipeline, volume_hu, image_itk, case_id=nombre)
        header = inferencia.cabecera(image_itk)
        st.session_state["prediccion"] = {
            "clave": clave, "nombre": nombre, "res": res, "gris": vis.ventanear(volume_hu),
            "mha": inferencia.mha_de_prediccion(res["instance_volume"], header),
        }
    except Exception as e:
        st.error(f"Falló la predicción: {e}")
    finally:
        volume_hu = image_itk = None
        gc.collect()

pred = st.session_state.get("prediccion")
if pred is None:
    st.info("Elige una tomografía y pulsa **Predecir fragmentos**.")
    st.stop()
if pred["clave"] != clave:
    st.caption(f"Mostrando el resultado anterior de `{pred['nombre']}`.")

# ---------------------------------------------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------------------------------------------
res, gris = pred["res"], pred["gris"]
inst = res["instance_volume"]
tabla = res["fragmentos"]
Z, H, W = inst.shape

st.subheader(f"Resultado · {pred['nombre']}")
m = st.columns(5)
m[0].metric("Fragmentos 3D", len(tabla))
for i, (c, nombre_region) in enumerate(regiones.items(), start=1):
    m[i].metric(nombre_region.replace("_", " ").capitalize(), int((tabla["class"] == c).sum()))
m[4].metric("Tiempo", f"{res['segundos']:.0f} s")

pelvis = res.get("pelvis")
if pelvis is not None and not pelvis["presente"]:
    st.warning(inferencia.control_pelvis.MENSAJE_SIN_PELVIS)
    st.caption(f"Bloque continuo más largo con detecciones: {pelvis['racha_mm']:.0f} mm "
               f"(se exigen {pelvis['min_mm']:.0f} mm con score ≥ {pelvis['det_gate']:.2f}).")
elif len(tabla) == 0:
    st.warning("El modelo no encontró fragmentos en este volumen.")

lut = vis.tabla_de_colores(dict(zip(tabla["fragment_id"], tabla["class"])), colores, int(inst.max()))
alpha = st.slider("Opacidad de la máscara", 0.0, 1.0, 0.5, 0.05)

z_con = np.flatnonzero(inst.any(axis=(1, 2)))
z_ini = int(z_con[len(z_con) // 2]) if len(z_con) else Z // 2

axial, coronal, mip = st.tabs(["Corte axial", "Corte coronal", "Proyección (MIP)"])
with axial:
    z = st.slider("Corte z", 0, Z - 1, z_ini)
    ids = [int(i) for i in np.unique(inst[z]) if i]
    st.image(vis.vista_axial(gris, inst, lut, z, alpha), width="stretch",
             caption=f"z = {z} · fragmentos en el corte: {ids or 'ninguno'}")
with coronal:
    y = st.slider("Corte y", 0, H - 1, H // 2)
    st.image(vis.vista_coronal(gris, inst, lut, y, res["spacing_xyz"], alpha), width="stretch")
with mip:
    st.image(vis.vista_mip(gris, inst, lut, res["spacing_xyz"], alpha), width="stretch",
             caption="Proyección coronal de máxima intensidad: todos los fragmentos a la vez.")

st.subheader("Fragmentos")
st.dataframe(inferencia.tabla_para_mostrar(tabla), hide_index=True, width="stretch",
             column_config={"Volumen (cm³)": st.column_config.NumberColumn(format="%.2f"),
                            "Confianza media": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0)})

base = str(pred["nombre"]).removesuffix(".mha")
d1, d2 = st.columns(2)
d1.download_button("Descargar máscara de fragmentos (.mha)", pred["mha"], file_name=f"pred_{base}.mha",
                   mime="application/octet-stream")
d2.download_button("Descargar tabla (.csv)", tabla.to_csv(index=False).encode("utf-8"),
                   file_name=f"pred_{base}_fragmentos.csv", mime="text/csv")

with st.expander("Detalles del pipeline"):
    st.write(f"Cortes con alguna caja detectada: {res['cortes_con_hueso']} de {Z}")
    if pelvis is not None:
        st.write(f"Control de pelvis: bloque continuo de {pelvis['racha_mm']:.0f} mm "
                 f"(cortes {pelvis['inicio']} a {pelvis['fin']}); se exigen {pelvis['min_mm']:.0f} mm.")
    st.write("Limpieza 3D:", res["log_limpieza"])
    st.write(f"Eventos de fusión/división para revisar: {len(res['eventos'])}")
