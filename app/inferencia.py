"""Inferencia del frontend: un MHA -> fragmentos 3D con los tres modelos por plano (axial, coronal y sagital).

Los modelos salen de Entrenamiento_multitarea_planos.ipynb: cada uno se entrenó sobre volúmenes en orientación canónica
(LPS, src/orientacion.py) cortados a lo largo de su plano (src/planos.py), con entrada 2.5D (cortes i-1, i, i+1) y, en el
sagital, un 4.º canal con la posición lateral del corte. La inferencia reproduce ese preprocesamiento:
    ventana HU -> píxeles cuadrados (coronal/sagital) -> realce (Otsu + CLAHE) -> letterbox 256.

Las funciones no se copian: se cargan por nombre desde los cuadernos (src/definiciones_notebooks.py), igual que hace
salida_3d.ipynb, y se usa la configuración 3D congelada con val (resultados/metricas/salida_3d/config_3d.json).
Flujo por plano: todos los cortes -> detección + NMS -> decoder -> geometría del corte -> asociación 3D a lo largo del
eje del plano -> limpieza -> volumen canónico (z, y, x).
"""
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _d in (ROOT / "src", ROOT):            # los cuadernos importan `rutas`, `deteccion_grid`... sin el prefijo src.
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

from definiciones_notebooks import cargar_definiciones
import control_pelvis
import planos
from distancias_fragmentos import REGION_BY_DATASET_ID, measure_fragment_separations
from orientacion import canonicalize_zyx, is_canonical
from rutas import DATA_MHA_DIR, METRICAS_DIR, MODELOS_DIR, NOTEBOOKS_DIR

warnings.filterwarnings("ignore", message="'mode' parameter is deprecated")   # letterbox_pair usa mode= (Pillow 13)

PLANOS = ("axial", "coronal", "sagital")
NOMBRE_PLANO = {"axial": "Axial", "coronal": "Coronal", "sagital": "Sagital"}
MODELOS = {p: MODELOS_DIR / f"entrenamiento_multitarea_{p}" / f"modelo_multitarea_{p}.pt" for p in PLANOS}
CONFIG_3D_PATH = METRICAS_DIR / "salida_3d" / "config_3d.json"
ETIQUETAS_DIR = DATA_MHA_DIR / "PENGWIN_CT_train_labels"

# Mismas definiciones y prioridad de cuadernos que la sección 1 de salida_3d.ipynb (si un nombre aparece en varios,
# gana el primero de la lista).
DEFINICIONES_EQUIPO = {"resolve_repo_path", "apply_hu_window", "enhance_bone_contrast", "instance_to_semantic",
                       "boxes_from_semantic", "letterbox_pair", "FundidoraPC", "FundidoraPCPlus", "build_backbone",
                       "build_instance_segmentation_targets", "build_detection_region_priors",
                       "SegmentationConvBlock", "backbone_feature_pyramid", "SkipFusionBlock",
                       "CenterOffsetSkipDecoder", "restore_letterbox_label_mask"}
VALORES_EQUIPO = {"CLASS_ID_TO_NAME", "TARGET_COLORS"}
PRIORIDAD = ["Segmentacion_instancias", "Backbone", "Ventaneo_hu", "Targets_deteccion", "Cabeza_grid", "Dataset_splits",
             "nms_pertenencia_pipeline", "Entrenamiento_multitarea", "cbam"]

# Piezas de salida_3d.ipynb: la decodificación, la asociación 3D y la limpieza (la inferencia por plano es de aquí).
DEFINICIONES_SALIDA_3D = {"trunk_features", "empty_bundle", "_logit", "_sigmoid", "_pick_peaks", "boxes_allowed_mask",
                          "decode_cached_slice", "active_slices", "link_instances_3d", "clean_tracks", "assemble_case",
                          "load_config_3d"}
VALORES_SALIDA_3D = {"DEVICE", "AMP", "IMAGE_SIZE", "CFG_SLICE", "REGION_OF_CLASS"}


def _componentes(ns, ruta):
    """{backbone, cbam, det_head, seg_decoder} desde un modelo_multitarea_{plano}.pt. El backbone recibe in_channels
    canales (3 cortes, + la posición en el sagital) y el decoder solo los image_channels de imagen."""
    payload = torch.load(ruta, map_location=ns["DEVICE"], weights_only=True)
    arch, state = payload["architecture"], payload["model_state_dict"]
    n_in = int(arch["in_channels"])
    n_img = int(arch.get("image_channels", n_in))
    cbam = None
    if arch["use_cbam"]:
        cbam = ns["CompuertaResidualCBAM"](ns["CBAM"](256)) if arch.get("use_gate", True) else ns["CBAM"](256)
    parts = {"backbone": ns["build_backbone"](arch["backbone_name"], in_channels=n_in), "cbam": cbam,
             "det_head": ns["GridDetectionHead"](),
             "seg_decoder": ns["CenterOffsetSkipDecoder"](in_channels=256, image_channels=n_img,
                                                          use_skips=arch.get("use_skips", True))}
    for name, module in parts.items():
        if module is None:
            continue
        module.load_state_dict({k[len(name) + 1:]: v for k, v in state.items() if k.startswith(name + ".")})
        module.to(ns["DEVICE"]).eval()
    comp = {**parts, "features_fn": ns["backbone_feature_pyramid"], "in_channels": n_in, "image_channels": n_img,
            "posicion": bool((arch.get("position") or {}).get("enabled", False)), "plano": arch.get("plane")}
    return comp, payload


def cargar_pipeline():
    """Namespace con las definiciones de los cuadernos + los tres modelos por plano + configuración 3D congelada."""
    for plano, ruta in MODELOS.items():
        if not ruta.is_file() or ruta.stat().st_size < 1024:
            raise FileNotFoundError(f"No encuentro los pesos del modelo {plano} en {ruta}. Si el archivo es un puntero "
                                    "de Git LFS, ejecuta `git lfs pull`.")
    ns = {}
    cuadernos = [NOTEBOOKS_DIR / f"{n}.ipynb" for n in PRIORIDAD if (NOTEBOOKS_DIR / f"{n}.ipynb").is_file()]
    cargar_definiciones(cuadernos, DEFINICIONES_EQUIPO, VALORES_EQUIPO, destino=ns)
    exec("from cbam_modulos import CBAM, CompuertaResidualCBAM", ns)
    cargar_definiciones(NOTEBOOKS_DIR / "salida_3d.ipynb", DEFINICIONES_SALIDA_3D, VALORES_SALIDA_3D, destino=ns)
    exec("from deteccion_grid import GridDetectionHead, postprocess, nms_por_clase", ns)
    # PENGWIN_DEVICE: auto (GPU si hay, si no CPU; por defecto) | cpu | cuda. El servicio de AWS fija cpu; en local se
    # puede forzar cpu para ver lo mismo que el servidor. AMP solo en GPU.
    pedido = os.environ.get("PENGWIN_DEVICE", "auto").strip().lower()
    if pedido not in ("auto", "cpu", "cuda"):
        raise ValueError(f"PENGWIN_DEVICE={pedido!r}: usa auto, cpu o cuda.")
    if pedido == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PENGWIN_DEVICE=cuda, pero PyTorch no ve ninguna GPU.")
    if pedido != "auto":
        ns["DEVICE"] = torch.device(pedido)
        ns["AMP"] = ns["DEVICE"].type == "cuda"

    modelos, info = {}, {}
    for plano, ruta in MODELOS.items():
        comp, payload = _componentes(ns, ruta)
        if comp["plano"] not in (None, plano):
            raise ValueError(f"{ruta.name} es un modelo {comp['plano']}, no {plano}.")
        modelos[plano] = comp
        sel = payload.get("selected_from", {})
        info[plano] = {"modelo": ruta.name, "etapa": sel.get("stage"), "epoca": sel.get("epoch"),
                       "canales_entrada": comp["in_channels"]}
    cfg_3d, meta_3d = ns["load_config_3d"](CONFIG_3D_PATH)
    return {"ns": ns, "modelos": modelos, "cfg_3d": cfg_3d, "info": info, "dispositivo": str(ns["DEVICE"]),
            "config_3d_creada": meta_3d.get("creado")}


# ---------------------------------------------------------------------------------------------------------------
# Lectura en orientación canónica
# ---------------------------------------------------------------------------------------------------------------
def _leer_itk(origen, nombre):
    if isinstance(origen, (str, Path)):
        return sitk.ReadImage(str(origen))
    # SimpleITK solo lee desde disco: los bytes subidos pasan por un temporal
    sufijo = ".nii.gz" if str(nombre).lower().endswith(".nii.gz") else Path(str(nombre)).suffix or ".mha"
    fd, tmp = tempfile.mkstemp(suffix=sufijo)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(origen)
        return sitk.ReadImage(tmp)
    finally:
        os.unlink(tmp)


def leer_canonico(origen, nombre=".mha", es_etiqueta=False):
    """origen: ruta a un .mha/.nii/.nii.gz o bytes de un archivo subido (nombre da la extensión).
    Devuelve (volumen (z, y, x) en orientación canónica LPS, espaciado (x, y, z)): HU float32, o int16 si es etiqueta.
    Es la misma lectura que load_canonical, con la que se entrenaron los tres modelos."""
    image_itk = _leer_itk(origen, nombre)
    if image_itk.GetDimension() != 3:
        raise ValueError(f"Se esperaba un volumen 3D; el archivo tiene {image_itk.GetDimension()} dimensiones.")
    arr = sitk.GetArrayFromImage(image_itk).astype(np.int16 if es_etiqueta else np.float32, copy=False)
    spacing, direction = tuple(float(v) for v in image_itk.GetSpacing()), image_itk.GetDirection()
    del image_itk
    if is_canonical(direction):
        return arr, spacing
    arr, spacing, _ = canonicalize_zyx(arr, spacing, (0.0, 0.0, 0.0), direction)
    return arr, tuple(float(v) for v in spacing)


# ---------------------------------------------------------------------------------------------------------------
# Inferencia de un plano
# ---------------------------------------------------------------------------------------------------------------
def _preparar_corte(ns, volume, plano, idx, spacing):
    """Un corte listo para la red: ventana HU -> píxeles cuadrados -> realce -> letterbox (igual que planos.make_sample)."""
    raw = ns["apply_hu_window"](np.asarray(planos.get_slice(volume, plano, idx), dtype=np.float32))
    img = ns["enhance_bone_contrast"](planos.slice_to_square(raw, plano, spacing, False))
    img256, _, tf = ns["letterbox_pair"](img, np.zeros(img.shape, dtype=np.int16), ns["IMAGE_SIZE"])
    return np.asarray(img256, dtype=np.float32), {**tf, "square_shape": tuple(int(v) for v in img.shape)}


@torch.no_grad()
def inferir_plano(pipeline, plano, volume, spacing, progreso=None):
    """TODOS los cortes del plano -> cajas (grid + NMS) -> priors -> decoder. Mismo contenido que infer_case de
    salida_3d.ipynb (salidas crudas por corte), con el preprocesamiento y los canales de Entrenamiento_multitarea_planos."""
    ns, comp, cfg = pipeline["ns"], pipeline["modelos"][plano], pipeline["ns"]["CFG_SLICE"]
    device, amp, size = ns["DEVICE"], ns["AMP"], ns["IMAGE_SIZE"]
    n = planos.n_slices(volume.shape, plano)
    forma_corte = planos.get_slice(volume, plano, 0).shape
    fov = planos.fov_center_mm(volume.shape, spacing)        # línea media en inferencia: centro del campo de visión
    hechos, bundle = {}, None

    def preparado(k):
        if k not in hechos:
            hechos[k] = _preparar_corte(ns, volume, plano, k, spacing)
        return hechos[k]

    for start in range(0, n, cfg["batch_size"]):
        indices = list(range(start, min(n, start + cfg["batch_size"])))
        lote = []
        for i in indices:
            canales = [preparado(k)[0] for k in planos.neighbor_indices(i, n, comp["image_channels"])]
            if comp["posicion"]:
                canales.append(planos.position_channel(planos.slice_position_mm(plano, i, spacing), fov, size))
            lote.append(np.stack(canales))
        if bundle is None:
            bundle = ns["empty_bundle"](plano, "model", (n, *forma_corte), spacing, preparado(0)[1])
        for k in [k for k in hechos if k < start - 1]:        # solo se guardan los vecinos que faltan por usar
            del hechos[k]
        x = torch.from_numpy(np.stack(lote).astype(np.float32)).to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            feats = ns["trunk_features"](comp, x)
            det_out = comp["det_head"](feats[-1])
        detections = ns["postprocess"](det_out.float(), score_thresh=cfg["det_score_min"], nms_fn=ns["nms_por_clase"],
                                       iou_thresh=cfg["nms_iou"], max_det_per_class=cfg["max_det_per_class"])
        for i, d in enumerate(detections):
            bundle["dets"][indices[i]] = {k: d[k].detach().cpu().numpy() for k in ("boxes", "scores", "labels")}
        con_cajas = [i for i, d in enumerate(detections) if len(d["boxes"]) > 0]
        if con_cajas:
            priors = ns["build_detection_region_priors"]([detections[i] for i in con_cajas], output_size=(size, size),
                                                          device=device)
            with torch.autocast(device_type=device.type, enabled=amp):
                # el decoder recibe solo los canales de imagen (sin el canal de posición del sagital)
                seg = comp["seg_decoder"](x[con_cajas, :comp["image_channels"]],
                                          [None if f is None else f[con_cajas] for f in feats], priors)
            sem = seg["semantic_logits"].float().argmax(1).cpu().numpy().astype(np.uint8)
            ctr = seg["center_logits"].float().cpu().numpy().astype(np.float16)
            off = seg["offsets"].float().cpu().numpy().astype(np.float16)
            for k, i in enumerate(con_cajas):
                z = indices[i]
                bundle["sem"][z], bundle["ctr"][z], bundle["off"][z], bundle["has_seg"][z] = sem[k], ctr[k], off[k], True
        if progreso is not None:
            progreso(indices[-1] + 1, n)
    return bundle


def _decodificar(ns, bundle, plano, spacing, dec_cfg):
    """Como decode_and_restore de salida_3d.ipynb, pero la vuelta a la geometría del corte deshace también el paso a
    píxeles cuadrados (planos.restore_slice_labels). IDs locales a cada corte."""
    n, alto, ancho = bundle["shape"]
    local = np.zeros((n, alto, ancho), dtype=np.uint16)
    clases, scores = [{} for _ in range(n)], [{} for _ in range(n)]
    for z in np.flatnonzero(ns["active_slices"](bundle, dec_cfg["det_gate"], dec_cfg["gate_min_run"])):
        allowed = None if dec_cfg["box_margin"] is None else ns["boxes_allowed_mask"](bundle["dets"][z],
                                                                                      dec_cfg["box_margin"])
        inst, cls, sc = ns["decode_cached_slice"](
            bundle["sem"][z], bundle["ctr"][z], bundle["off"][z], center_threshold=dec_cfg["center_threshold"],
            min_center_distance=dec_cfg["min_center_distance"], min_instance_pixels=dec_cfg["min_instance_pixels"],
            fallback_components=dec_cfg["fallback_components"], allowed=allowed)
        if not cls:
            continue
        local[z] = planos.restore_slice_labels(inst, plano, (alto, ancho), spacing, bundle["transform"],
                                               ns["restore_letterbox_label_mask"], ns["IMAGE_SIZE"])
        clases[z], scores[z] = cls, sc
    return local, clases, scores


def _a_canonico(pila, plano):
    """Pila de cortes del plano (corte, fila, columna) -> volumen canónico (z, y, x)."""
    if plano == "axial":
        return pila
    return np.ascontiguousarray(np.transpose(pila, (1, 0, 2) if plano == "coronal" else (1, 2, 0)))


def predecir_plano(pipeline, plano, volume, spacing, case_id="subido", progreso=None):
    """Un plano completo: inferencia -> fragmentos 3D en el volumen canónico + tabla con distancias. No usa etiquetas."""
    ns, cfg = pipeline["ns"], pipeline["cfg_3d"]
    t0 = time.time()
    bundle = inferir_plano(pipeline, plano, volume, spacing, progreso)
    # Control de "pelvis presente" (src/control_pelvis.py): sin un bloque continuo de detecciones de al menos 80 mm a
    # lo largo del eje del plano, el estudio no tiene una pelvis completa y no se entregan fragmentos.
    pelvis = control_pelvis.pelvis_presente_bundle(bundle, planos.slice_axis_spacing(plano, spacing))
    if pelvis["presente"]:
        decoded = _decodificar(ns, bundle, plano, spacing, cfg["decode"])
        result = ns["assemble_case"](bundle, cfg["decode"], cfg["link"], cfg["post"], stride=1, decoded=decoded)
        del decoded
        inst = _a_canonico(result["volume"], plano)
        tabla, eventos, log = result["table"].copy(), result["events"], result["log"]
    else:
        inst = np.zeros(volume.shape, dtype=np.uint16)
        tabla = pd.DataFrame(columns=["fragment_id", "class", "region", "n_voxels", "n_slices", "mean_score"])
        eventos, log = [], {}
    tabla["volumen_cm3"] = tabla["n_voxels"] * float(np.prod(spacing)) / 1000.0
    region_by_id = {int(f): ns["REGION_OF_CLASS"][int(c)] for f, c in zip(tabla["fragment_id"], tabla["class"])}
    tabla = _con_distancias(tabla, inst, spacing, case_id, region_by_id)
    cajas = cajas_por_corte(bundle, cfg["decode"]["det_gate"]) if pelvis["presente"] else [None] * len(bundle["dets"])
    del bundle
    return {"plano": plano, "instance_volume": inst.astype(np.uint8) if inst.max() < 256 else inst,
            "region_by_id": region_by_id, "fragmentos": tabla, "cajas": cajas, "eventos": eventos,
            "log_limpieza": log, "pelvis": pelvis, "segundos": time.time() - t0}


def _con_distancias(tabla, instance_volume, spacing, case_id, region_by_id):
    """Distancia borde a borde (mm) de cada fragmento al principal de su hueso (src/distancias_fragmentos.py:
    EDT con el espaciado físico del header). El principal es el de más vóxeles y tiene distancia 0."""
    tabla = tabla.copy()
    tabla["distancia_mm"], tabla["es_principal"] = np.nan, False
    if len(tabla) == 0:
        return tabla
    dist = measure_fragment_separations(instance_volume, spacing, case_id=case_id, region_by_id=region_by_id)
    dist = dist.set_index("fragment_id")
    tabla["distancia_mm"] = tabla["fragment_id"].map(dist["distance_mm"])
    tabla["es_principal"] = tabla["fragment_id"].map(dist["is_main"]).fillna(False).astype(bool)
    return tabla


def cajas_por_corte(bundle, det_gate):
    """Cajas del detector (grid + NMS) en píxeles CUADRADOS del corte (antes del letterbox a 256 x 256), que es como
    se dibuja el corte. Solo las de score >= det_gate, el umbral con el que la salida 3D decide qué cortes usar."""
    tr = bundle["transform"]
    salida = []
    for det in bundle["dets"]:
        if det is None or not len(det["scores"]):
            salida.append(None)
            continue
        keep = det["scores"] >= det_gate
        boxes = det["boxes"][keep].astype(np.float32).copy()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - tr["offset_x"]) / tr["scale"]
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - tr["offset_y"]) / tr["scale"]
        salida.append({"boxes": boxes, "labels": det["labels"][keep].astype(int), "scores": det["scores"][keep]}
                      if keep.any() else None)
    return salida


# ---------------------------------------------------------------------------------------------------------------
# Fragmentos reales (etiqueta de referencia)
# ---------------------------------------------------------------------------------------------------------------
def etiqueta_local(nombre):
    """Etiqueta del dataset con el mismo nombre que la tomografía subida (p. ej. 035.mha), si está en data_mha/."""
    ruta = ETIQUETAS_DIR / Path(str(nombre)).name
    return ruta if ruta.is_file() else None


def fragmentos_referencia(label, spacing, case_id="referencia"):
    """Fragmentos de la etiqueta (IDs 1-10 sacro, 11-20 coxal izq., 21-30 coxal der.) con el mismo formato que la
    tabla de la predicción: volumen y distancia al fragmento principal de su hueso."""
    if not label.any():
        return pd.DataFrame(columns=["fragment_id", "class", "region", "n_voxels", "n_slices", "mean_score",
                                     "volumen_cm3", "distancia_mm", "es_principal"])
    dist = measure_fragment_separations(label, spacing, case_id=case_id)
    clase = {r: c for c, r in {1: "sacro", 2: "coxal_izquierdo", 3: "coxal_derecho"}.items()}
    return pd.DataFrame({
        "fragment_id": dist["fragment_id"].astype(int), "class": dist["region"].map(clase), "region": dist["region"],
        "n_voxels": dist["n_voxels"], "n_slices": np.nan, "mean_score": np.nan,
        "volumen_cm3": dist["n_voxels"] * float(np.prod(spacing)) / 1000.0,
        "distancia_mm": dist["distance_mm"], "es_principal": dist["is_main"].astype(bool)})


def region_de_id(fragment_id):
    return REGION_BY_DATASET_ID.get(int(fragment_id))


UMBRAL_HUESO_HU = 300                       # mismo umbral que compute_mip de Visualizacion_3d.ipynb


def mip_hueso(volume_hu, umbral_hu=UMBRAL_HUESO_HU):
    """Visualizador 1: MIP coronal (z, x) del volumen CRUDO canónico dejando solo el hueso (HU > umbral). No usa el modelo."""
    return np.where(volume_hu > umbral_hu, volume_hu, umbral_hu).max(axis=1)


NOMBRE_REGION = {"sacro": "Sacro", "coxal_izquierdo": "Coxal izquierdo", "coxal_derecho": "Coxal derecho"}
COLUMNAS_TABLA = ["fragment_id", "region", "es_principal", "distancia_mm", "volumen_cm3"]


def tabla_para_mostrar(fragmentos):
    tabla = pd.DataFrame(fragmentos)[COLUMNAS_TABLA].copy()
    tabla["region"] = tabla["region"].map(NOMBRE_REGION).fillna(tabla["region"])
    tabla["es_principal"] = np.where(tabla["es_principal"], "Principal", "Conminuto")
    return tabla.sort_values(["region", "fragment_id"]).rename(columns={
        "fragment_id": "Fragmento", "region": "Hueso", "es_principal": "Tipo", "distancia_mm": "Distancia (mm)",
        "volumen_cm3": "Volumen (cm³)"})


def conteo_por_hueso(fragmentos):
    """Número de fragmentos por hueso (las tres regiones siempre, aunque tengan 0) + total."""
    regiones = pd.DataFrame(fragmentos)["region"] if len(fragmentos) else pd.Series(dtype=str)
    conteo = {NOMBRE_REGION[r]: int((regiones == r).sum()) for r in NOMBRE_REGION}
    conteo["Total"] = sum(conteo.values())
    return conteo
