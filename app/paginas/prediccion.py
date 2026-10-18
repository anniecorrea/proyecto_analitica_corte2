import gc
import hashlib

import numpy as np
import streamlit as st

import inferencia
import visualizacion as vis
from src import plausibilidad_pelvis as plausibilidad

# El cargador no muestra el límite de tamaño ni los formatos aceptados (solo "Subir imagen").
st.markdown("<style>[data-testid='stFileUploaderDropzoneInstructions'] {display: none;}</style>",
            unsafe_allow_html=True)

st.title("PENGWIN")
st.markdown(
    "**Detección, segmentación y medición de fracturas pélvicas en tomografía computarizada.**\n\n"
    "Este proyecto académico, basado en el reto PENGWIN, analiza una tomografía de pelvis con tres modelos de "
    "aprendizaje profundo, uno por plano anatómico (**axial, coronal y sagital**). Cada modelo localiza el sacro y los "
    "dos huesos coxales en su plano; sus resultados se **fusionan en 3D** para separar los fragmentos óseos de cada "
    "hueso. Para cada fragmento se informa su volumen y la distancia de separación al fragmento principal de su hueso.\n\n"
    "**Cómo usar la página:** sube una tomografía." \
    "Las métricas en test están en **Dashboard de resultados**.")


@st.cache_resource(show_spinner=False)
def obtener_pipeline():
    # modelos de la etapa B: la fusión 3D de los tres planos se validó en test con ellos (Test_final_fusion_B.ipynb)
    return inferencia.cargar_pipeline(fusion=True)


@st.cache_resource(show_spinner=False)
def obtener_referencia_pelvis():
    """Rangos de la pelvis en los pacientes de entrenamiento (src/plausibilidad_pelvis.py); None si no están."""
    return plausibilidad.cargar_referencia()


def revisar_plausibilidad(res, spacing):
    """{"avisos": [...], "notas": [...]} sobre la pelvis predicha: avisos si se sale de lo esperable para una pelvis
    humana adulta; notas si solo se sale del rango de los pacientes de entrenamiento. Solo se revisa cuando el control
    de pelvis la dio por presente; sin la referencia no se revisa."""
    referencia = obtener_referencia_pelvis()
    if referencia is None or not res["pelvis"]["presente"]:
        return {"avisos": [], "notas": []}
    regiones_pred = plausibilidad.regiones_desde_instancias(res["instance_volume"], res["region_by_id"])
    return plausibilidad.evaluar_niveles(plausibilidad.medidas_pelvis(regiones_pred, spacing), referencia)


try:
    pipeline = obtener_pipeline()
except Exception as e:                    # pesos sin `git lfs pull`, cuaderno sin una definición, etc.
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()

colores = pipeline["ns"]["TARGET_COLORS"]
regiones = pipeline["ns"]["REGION_OF_CLASS"]
CLASE_DE_REGION = {r: c for c, r in regiones.items()}


# La imagen y su predicción viven en session_state: al ir al dashboard y volver, el cargador aparece vacío (Streamlit
# borra los widgets de la página que no se muestra), pero los resultados siguen ahí. Solo se borran si el usuario
# quita el archivo del cargador.
def _al_cambiar_imagen():
    if st.session_state.get("archivo_imagen") is None:
        st.session_state.pop("prediccion", None)
        st.session_state.pop("referencia", None)


st.subheader("Subir imagen")
archivo = st.file_uploader("Subir imagen", type=["mha", "nii", "gz"], label_visibility="collapsed",
                           key="archivo_imagen", on_change=_al_cambiar_imagen)

if archivo is not None:
    origen, nombre = archivo.getvalue(), archivo.name
    clave = hashlib.md5(origen).hexdigest()
    if st.session_state.get("prediccion", {}).get("clave") != clave:
        st.session_state.pop("prediccion", None)
        st.session_state.pop("referencia", None)
        gc.collect()
        volume = None
        try:
            with st.spinner("Leyendo imagen"):
                volume, spacing = inferencia.leer_canonico(origen, nombre)
            del origen
            barra = st.progress(0.0, text="Analizando la tomografía")
            res = inferencia.predecir_fusion(
                pipeline, volume, spacing, case_id=nombre,
                progreso=lambda k, n, pl: barra.progress(min(k / n, 1.0), text=f"Modelo {pl}: corte {k} de {n} (3 planos)"))
            barra.empty()
            tabla, inst = res["fragmentos"], res["instance_volume"]
            fusion = {
                "res": res,
                "plausibilidad": revisar_plausibilidad(res, spacing),
                "lut": vis.tabla_de_colores(dict(zip(tabla["fragment_id"], tabla["class"])), colores, max(int(inst.max()), 1)),
                "etiquetas": {int(f["fragment_id"]): vis.etiqueta_fragmento(f) for _, f in tabla.iterrows()},
            }
            st.session_state["prediccion"] = {
                "clave": clave, "nombre": nombre, "spacing": spacing, "forma": volume.shape, "fusion": fusion,
                "gris": vis.ventanear(volume), "mip": inferencia.mip_hueso(volume), "figuras": {},
            }
        except Exception as e:
            st.error(f"No se pudo procesar la imagen: {e}")
            st.stop()
        finally:
            volume = None
            gc.collect()

pred = st.session_state.get("prediccion")
if pred is not None and "fusion" not in pred:          # resultado de una versión anterior de la app (por plano)
    st.session_state.pop("prediccion", None)
    pred = None
if pred is None:
    st.stop()
if archivo is None:
    st.caption(f"Mostrando los resultados de **{pred['nombre']}**. Para analizar otra tomografía, súbela arriba.")

spacing = pred["spacing"]

# ---------------------------------------------------------------------------------------------------------------
# Etiqueta de referencia: la del dataset con el mismo nombre que la tomografía (data_mha/), si existe.
# ---------------------------------------------------------------------------------------------------------------
ref = st.session_state.get("referencia")
if ref is not None and ref["clave_imagen"] != pred["clave"]:
    ref = None
fuente_ref = None
ruta_ref = inferencia.etiqueta_local(pred["nombre"])
if ref is None and ruta_ref is not None:
    fuente_ref = (str(ruta_ref), ruta_ref, ruta_ref.name)
rechazada = fuente_ref is not None and st.session_state.get("referencia_rechazada") == (pred["clave"], fuente_ref[0])
if rechazada:
    st.warning(f"La etiqueta {fuente_ref[2]} no se pudo usar con esta tomografía.")
if fuente_ref is not None and (ref is None or ref["clave"] != fuente_ref[0]) and not rechazada:
    ref = None
    st.session_state["referencia_rechazada"] = (pred["clave"], fuente_ref[0])     # se borra abajo si la etiqueta vale
    try:
        with st.spinner("Leyendo la etiqueta de referencia"):
            label, sp_ref = inferencia.leer_canonico(fuente_ref[1], fuente_ref[2], es_etiqueta=True)
        if label.shape != tuple(pred["forma"]) or not np.allclose(sp_ref, spacing, atol=1e-3):
            st.error(f"La etiqueta {fuente_ref[2]} no corresponde a esta tomografía (tamaño {label.shape} frente a "
                     f"{tuple(pred['forma'])}).")
        elif label.min() < 0 or label.max() > 30:
            st.error("La etiqueta no usa los IDs de PENGWIN (1-10 sacro, 11-20 coxal izquierdo, 21-30 coxal derecho).")
        else:
            tabla_ref = inferencia.fragmentos_referencia(label, spacing, case_id=pred["nombre"])
            ref = {"clave": fuente_ref[0], "clave_imagen": pred["clave"], "nombre": fuente_ref[2],
                   "volumen": label.astype(np.uint8), "fragmentos": tabla_ref,
                   "lut": vis.tabla_de_colores({i: CLASE_DE_REGION[inferencia.region_de_id(i)] for i in range(1, 31)},
                                               colores, 30),
                   "etiquetas": {int(f["fragment_id"]): vis.etiqueta_fragmento(f) for _, f in tabla_ref.iterrows()}}
            st.session_state["referencia"] = ref
            st.session_state.pop("referencia_rechazada", None)
        del label
    except Exception as e:
        st.error(f"No se pudo leer la etiqueta: {e}")
    gc.collect()

# ---------------------------------------------------------------------------------------------------------------
# Fragmentos: los de la imagen (si hay etiqueta) y los que predice la fusión de los tres planos
# ---------------------------------------------------------------------------------------------------------------
dp = pred["fusion"]
res, lut = dp["res"], dp["lut"]
nombre_modelo = "fusión de los 3 planos"
if not res["pelvis"]["presente"]:
    st.warning(inferencia.control_pelvis.MENSAJE_SIN_PELVIS)
elif isinstance(dp.get("plausibilidad"), dict):
    if dp["plausibilidad"]["avisos"]:
        st.warning(plausibilidad.mensaje(dp["plausibilidad"]["avisos"]), icon="⚠️")
    if dp["plausibilidad"]["notas"]:
        st.info(plausibilidad.mensaje_nota(dp["plausibilidad"]["notas"]), icon="ℹ️")
c = st.columns(2)
c[0].metric("Fragmentos en la imagen", len(ref["fragmentos"]) if ref is not None else "sin etiqueta", border=True)
c[1].metric(f"Fragmentos predichos ({nombre_modelo})", len(res["fragmentos"]), border=True)


def vista(destino, inst, lut_vista, textos):
    """Vista frontal con los fragmentos coloreados (se calcula una vez y queda en la sesión)."""
    if "vista" not in destino:
        destino["vista"] = vis.vista_fragmentos(pred["mip"], inst, lut_vista, spacing, inferencia.UMBRAL_HUESO_HU, textos)
    return destino["vista"]


if ref is not None:
    c[0].image(vista(ref, ref["volumen"], ref["lut"], ref["etiquetas"]), width="stretch",
               caption="Fragmentos reales (etiqueta de referencia)")
if len(res["fragmentos"]):
    c[1].image(vista(dp, res["instance_volume"], lut, dp["etiquetas"]), width="stretch",
               caption=f"Fragmentos que predice el {nombre_modelo}")
st.caption("Vista frontal de la pelvis: cada color es un fragmento y la etiqueta indica su hueso y su distancia al "
           "fragmento principal. Un fragmento que queda detrás de otro solo se ve por su etiqueta; en el visor de cortes "
           "se ve completo.")

# ---------------------------------------------------------------------------------------------------------------
# Visualizador 1: volumen crudo (MIP con umbral HU, sin modelo)
# ---------------------------------------------------------------------------------------------------------------
st.header("Imagen original")
izq, _ = st.columns([2, 1])
izq.image(vis.imagen_mip_hueso(pred["mip"], spacing, inferencia.UMBRAL_HUESO_HU), width="stretch")

# ---------------------------------------------------------------------------------------------------------------
# Visualizador 2: corte por corte en el plano elegido
# ---------------------------------------------------------------------------------------------------------------
st.header("Inferencia corte por corte")
plano = st.radio("Plano", inferencia.PLANOS, format_func=inferencia.NOMBRE_PLANO.get, horizontal=True,
                 key="plano_detalle", help="Orientación del corte. Las máscaras son las de la fusión de los tres planos; "
                                          "las cajas, las del detector del plano elegido.")
st.markdown(" &nbsp; ".join(f"<span style='color:{colores[c]}'>&#9632;</span> {inferencia.NOMBRE_REGION[r]}"
                            for c, r in regiones.items()), unsafe_allow_html=True)


@st.fragment
def visor_cortes():
    # Solo este bloque se vuelve a dibujar al mover el slider (la malla 3D no se recalcula).
    ver_ref = ref is not None and st.toggle("Ver la referencia en lugar de la predicción", key="ver_referencia")
    inst = ref["volumen"] if ver_ref else res["instance_volume"]
    n = inferencia.planos.n_slices(inst.shape, plano)
    if f"pixeles_{plano}" not in dp:
        dp[f"pixeles_{plano}"] = vis.pixeles_por_corte(res["instance_volume"], plano)
    pixeles = dp[f"pixeles_{plano}"]
    inicio = int(pixeles.argmax()) if pixeles.any() else n // 2
    z = st.slider(f"Corte {inferencia.NOMBRE_PLANO[plano].lower()}", 0, n - 1, inicio, key=f"corte_{plano}")
    gris, etiquetas, volteado = vis.corte_del_plano(pred["gris"], inst, plano, z, spacing)
    cajas = None if ver_ref else res["cajas"][plano][z]
    if volteado:
        cajas = vis.voltear_cajas(cajas, gris.shape[0])
    izq, _ = st.columns([2, 1])
    izq.image(vis.corte_anotado(gris, etiquetas, ref["lut"] if ver_ref else lut, cajas, colores,
                                ref["etiquetas"] if ver_ref else dp["etiquetas"]),
              width="stretch", caption=f"Corte {z + 1} de {n} · {'referencia' if ver_ref else nombre_modelo}")


visor_cortes()

# ---------------------------------------------------------------------------------------------------------------
# Visualizador 3: reconstrucción 3D (la figura se calcula la primera vez que se pide y queda en la sesión)
# ---------------------------------------------------------------------------------------------------------------
def figura(clave, inst, fragmentos, lut_figura):
    if clave not in pred["figuras"]:
        with st.spinner("Construyendo la reconstrucción 3D"):
            pred["figuras"][clave] = vis.figura_3d(inst, spacing, fragmentos, lut_figura) if len(fragmentos) else None
    return pred["figuras"][clave]


st.header("Reconstrucción 3D")
vistas = [(f"Predicción ({nombre_modelo})", "fusion", res["instance_volume"], res["fragmentos"], lut)]
if ref is not None:
    vistas.append(("Referencia", f"referencia:{ref['clave']}", ref["volumen"], ref["fragmentos"], ref["lut"]))
for pestana, (_, clave, inst, fragmentos, lut_figura) in zip(st.tabs([v[0] for v in vistas]), vistas):
    with pestana:
        fig = figura(clave, inst, fragmentos, lut_figura)
        if fig is None:
            st.info("No hay fragmentos que reconstruir.")
        else:
            st.plotly_chart(fig, width="stretch", key=f"fig_{clave}")
