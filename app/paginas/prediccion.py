import gc
import hashlib

import streamlit as st

import inferencia
import visualizacion as vis

# El cargador no muestra el límite de tamaño ni los formatos aceptados (solo "Subir imagen").
st.markdown("<style>[data-testid='stFileUploaderDropzoneInstructions'] {display: none;}</style>",
            unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def obtener_pipeline():
    return inferencia.cargar_pipeline()


try:
    pipeline = obtener_pipeline()
except Exception as e:                    # pesos sin `git lfs pull`, cuaderno sin una definición, etc.
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()

colores = pipeline["ns"]["TARGET_COLORS"]
regiones = pipeline["ns"]["REGION_OF_CLASS"]

st.markdown("<h3 style='color:#white'>Subir imagen</h3>", unsafe_allow_html=True)
archivo = st.file_uploader("Subir imagen", type=["mha", "nii", "gz"], label_visibility="collapsed")
if archivo is None:
    st.session_state.pop("prediccion", None)
    st.stop()

origen, nombre = archivo.getvalue(), archivo.name
clave = hashlib.md5(origen).hexdigest()
if st.session_state.get("prediccion", {}).get("clave") != clave:
    st.session_state.pop("prediccion", None)
    gc.collect()
    volume_hu = image_itk = None
    try:
        with st.spinner("Leyendo imagen (esto puede tardar unos minutos)"):
            volume_hu, image_itk = inferencia.leer_mha(origen, nombre)
            res = inferencia.predecir_volumen(pipeline, volume_hu, image_itk, case_id=nombre)
            tabla, inst = res["fragmentos"], res["instance_volume"]
            lut = vis.tabla_de_colores(dict(zip(tabla["fragment_id"], tabla["class"])), colores, int(inst.max()))
            st.session_state["prediccion"] = {
                "clave": clave, "res": res, "lut": lut,
                "gris": vis.ventanear(volume_hu),
                "mip": inferencia.mip_hueso(volume_hu),
                "figura_3d": vis.figura_3d(inst, res["spacing_xyz"], tabla, lut) if len(tabla) else None,
                "etiquetas": {int(f["fragment_id"]): vis.etiqueta_fragmento(f) for _, f in tabla.iterrows()},
            }
    except Exception as e:
        st.error(f"No se pudo procesar la imagen: {e}")
        st.stop()
    finally:
        volume_hu = image_itk = None
        gc.collect()

pred = st.session_state["prediccion"]
res, gris, lut = pred["res"], pred["gris"], pred["lut"]
inst, spacing = res["instance_volume"], res["spacing_xyz"]

if not res["pelvis"]["presente"]:
    st.warning(inferencia.control_pelvis.MENSAJE_SIN_PELVIS)
elif len(res["fragmentos"]) == 0:
    st.warning("El modelo no encontró fragmentos en esta imagen.")

# Visualizador 1: volumen crudo (MIP con umbral HU, sin modelo)
st.header("Imagen original")
izq, _ = st.columns([2, 1])
izq.image(vis.imagen_mip_hueso(pred["mip"], spacing, inferencia.UMBRAL_HUESO_HU), width="stretch")

# Visualizador 2: corte por corte
st.header("Inferencia corte por corte")
st.markdown(" &nbsp; ".join(f"<span style='color:{colores[c]}'>&#9632;</span> {inferencia.NOMBRE_REGION[r]}"
                            for c, r in regiones.items()), unsafe_allow_html=True)


@st.fragment
def visor_cortes():
    # Solo este bloque se vuelve a dibujar al mover el slider (la malla 3D no se recalcula).
    Z = inst.shape[0]
    pixeles = inst.reshape(Z, -1).astype(bool).sum(axis=1)
    z = st.slider("Corte", 0, Z - 1, int(pixeles.argmax()) if pixeles.any() else Z // 2)
    izq, _ = st.columns([2, 1])
    izq.image(vis.corte_anotado(gris[z], inst[z], lut, res["cajas"][z], colores, pred["etiquetas"]),
              width="stretch", caption=f"Corte {z + 1} de {Z}")


visor_cortes()

# Visualizador 3: reconstrucción 3D
st.header("Reconstrucción 3D")
if pred["figura_3d"] is None:
    st.info("No hay fragmentos que reconstruir.")
else:
    st.plotly_chart(pred["figura_3d"], width="stretch")
