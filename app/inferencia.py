"""Inferencia del frontend: un MHA -> fragmentos 3D con el mismo pipeline de salida_3d.ipynb.

Las funciones no se copian: se cargan por nombre desde los cuadernos (src/definiciones_notebooks.py), igual que hace
salida_3d.ipynb, y se usa la configuración congelada con val (resultados/metricas/salida_3d/config_3d.json).
Flujo: todos los cortes -> detección + NMS -> decoder -> geometría original del MHA -> asociación 3D -> limpieza.
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
from distancias_fragmentos import measure_fragment_separations
from rutas import DATA_MHA_DIR, METRICAS_DIR, MODELOS_DIR, NOTEBOOKS_DIR

warnings.filterwarnings("ignore", message="'mode' parameter is deprecated")   # letterbox_pair usa mode= (Pillow 13)

MODELO_PATH = MODELOS_DIR / "entrenamiento_multitarea" / "modelo_multitarea_final.pt"
CONFIG_3D_PATH = METRICAS_DIR / "salida_3d" / "config_3d.json"

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

# Piezas de salida_3d.ipynb que forman predict_case_3d (sin la lectura por case_id de splits/dataset.csv).
DEFINICIONES_SALIDA_3D = {"build_components_from_export", "trunk_features", "preprocess_slices", "empty_bundle",
                          "infer_case", "_logit", "_sigmoid", "_pick_peaks", "boxes_allowed_mask",
                          "decode_cached_slice", "active_slices", "decode_and_restore", "link_instances_3d",
                          "clean_tracks", "assemble_case", "load_config_3d"}
VALORES_SALIDA_3D = {"DEVICE", "AMP", "IMAGE_SIZE", "CFG_SLICE", "REGION_OF_CLASS"}


def cargar_pipeline():
    """Namespace con las definiciones de los cuadernos + componentes del modelo + configuración 3D congelada."""
    if not MODELO_PATH.is_file() or MODELO_PATH.stat().st_size < 1024:
        raise FileNotFoundError(f"No encuentro los pesos en {MODELO_PATH}. Si el archivo es un puntero de Git LFS, "
                                "ejecuta `git lfs pull`.")
    ns = {}
    cuadernos = [NOTEBOOKS_DIR / f"{n}.ipynb" for n in PRIORIDAD if (NOTEBOOKS_DIR / f"{n}.ipynb").is_file()]
    cargar_definiciones(cuadernos, DEFINICIONES_EQUIPO, VALORES_EQUIPO, destino=ns)
    exec("from cbam_modulos import CBAM, CompuertaResidualCBAM", ns)
    cargar_definiciones(NOTEBOOKS_DIR / "salida_3d.ipynb", DEFINICIONES_SALIDA_3D, VALORES_SALIDA_3D, destino=ns)

    comp, payload = ns["build_components_from_export"](MODELO_PATH)
    cfg_3d, meta_3d = ns["load_config_3d"](CONFIG_3D_PATH)
    sel = payload.get("selected_from", {})
    info = {"modelo": MODELO_PATH.name, "etapa": sel.get("stage"), "epoca": sel.get("epoch"),
            "canales_entrada": comp["in_channels"], "dispositivo": str(ns["DEVICE"]),
            "config_3d_creada": meta_3d.get("creado")}
    return {"ns": ns, "comp": comp, "cfg_3d": cfg_3d, "info": info}


def leer_mha(origen, nombre=".mha"):
    """origen: ruta a un .mha/.nii/.nii.gz o bytes de un archivo subido (nombre da la extensión).
    Devuelve (volumen HU (z, y, x) float32, imagen SimpleITK)."""
    if isinstance(origen, (str, Path)):
        image_itk = sitk.ReadImage(str(origen))
    else:                                   # SimpleITK solo lee desde disco: los bytes subidos pasan por un temporal
        sufijo = ".nii.gz" if str(nombre).lower().endswith(".nii.gz") else Path(str(nombre)).suffix or ".mha"
        fd, tmp = tempfile.mkstemp(suffix=sufijo)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(origen)
            image_itk = sitk.ReadImage(tmp)
        finally:
            os.unlink(tmp)
    if image_itk.GetDimension() != 3:
        raise ValueError(f"Se esperaba una tomografía 3D; el archivo tiene {image_itk.GetDimension()} dimensiones.")
    return sitk.GetArrayFromImage(image_itk).astype(np.float32), image_itk


def predecir_volumen(pipeline, volume_hu, image_itk, case_id="subido"):
    """Igual que predict_case_3d (salida_3d.ipynb) pero con el volumen ya leído. No usa ninguna etiqueta."""
    ns, comp, cfg = pipeline["ns"], pipeline["comp"], pipeline["cfg_3d"]
    t0 = time.time()
    with torch.no_grad():
        bundle = ns["infer_case"](comp, case_id, volume=volume_hu, image_itk=image_itk, verbose=False)
    spacing = tuple(float(s) for s in image_itk.GetSpacing())
    # Control de "pelvis presente" (src/control_pelvis.py): sin un bloque continuo de detecciones de al menos 80 mm,
    # el estudio no tiene una pelvis completa y no se entregan fragmentos (evita inventarlos en otros huesos).
    pelvis = control_pelvis.pelvis_presente_bundle(bundle, spacing[2])
    if pelvis["presente"]:
        result = ns["assemble_case"](bundle, cfg["decode"], cfg["link"], cfg["post"], stride=1)
    else:
        result = {"volume": np.zeros(volume_hu.shape, dtype=np.uint16), "events": [], "log": {},
                  "table": pd.DataFrame(columns=["fragment_id", "class", "region", "n_voxels", "n_slices", "z_min", "z_max", "mean_score"])}
    tabla = result["table"].copy()
    tabla["volumen_cm3"] = tabla["n_voxels"] * float(np.prod(spacing)) / 1000.0
    region_by_id = {int(f): ns["REGION_OF_CLASS"][int(c)] for f, c in zip(tabla["fragment_id"], tabla["class"])}
    tabla = _con_distancias(tabla, result["volume"], spacing, case_id, region_by_id)
    cajas = cajas_por_corte(bundle, cfg["decode"]["det_gate"]) if pelvis["presente"] else [None] * len(bundle["dets"])
    return {"case_id": case_id, "instance_volume": result["volume"], "spacing_xyz": spacing,
            "region_by_id": region_by_id, "fragmentos": tabla, "cajas": cajas,
            "eventos": result["events"], "log_limpieza": result["log"],
            "cortes_con_hueso": int(sum(1 for d in bundle["dets"] if d is not None and len(d["scores"]))),
            "pelvis": pelvis, "n_cortes": int(volume_hu.shape[0]),
            "segundos": time.time() - t0}


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
    """Cajas del detector (grid + NMS) en píxeles del corte ORIGINAL: deshace el letterbox a 256 x 256.
    Solo las de score >= det_gate, el mismo umbral con el que la salida 3D decide qué cortes usar."""
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


UMBRAL_HUESO_HU = 300                       # mismo umbral que compute_mip de Visualizacion_3d.ipynb


def mip_hueso(volume_hu, umbral_hu=UMBRAL_HUESO_HU):
    """Visualizador 1: MIP coronal del volumen CRUDO dejando solo el hueso (HU > umbral). No usa el modelo."""
    return np.where(volume_hu > umbral_hu, volume_hu, umbral_hu).max(axis=1)


def cabecera(image_itk):
    """Geometría del MHA (sin los vóxeles) para no guardar la tomografía completa en la sesión."""
    return {"spacing": image_itk.GetSpacing(), "origin": image_itk.GetOrigin(), "direction": image_itk.GetDirection()}


def mha_de_prediccion(instance_volume, header):
    """Bytes de un .mha con los IDs de fragmento y la MISMA geometría que la tomografía (como save_volume_mha)."""
    image = sitk.GetImageFromArray(instance_volume.astype(np.uint16))
    image.SetSpacing(header["spacing"])
    image.SetOrigin(header["origin"])
    image.SetDirection(header["direction"])
    fd, tmp = tempfile.mkstemp(suffix=".mha")
    os.close(fd)
    try:
        sitk.WriteImage(image, tmp, useCompression=True)
        return Path(tmp).read_bytes()
    finally:
        os.unlink(tmp)


def casos_locales():
    """Tomografías de ejemplo en data_mha/ (carpetas *images*). En el servidor sirven para la demo: el túnel de
    Cloudflare no deja subir archivos de más de 100 MB y una tomografía pesa ~400 MB."""
    if not DATA_MHA_DIR.is_dir():
        return []
    return sorted(p for p in DATA_MHA_DIR.glob("*images*/*.mha"))


NOMBRE_REGION = {"sacro": "Sacro", "coxal_izquierdo": "Coxal izquierdo", "coxal_derecho": "Coxal derecho"}
COLUMNAS_TABLA = ["fragment_id", "region", "es_principal", "distancia_mm", "volumen_cm3", "n_slices", "mean_score"]


def tabla_para_mostrar(fragmentos):
    tabla = pd.DataFrame(fragmentos)[COLUMNAS_TABLA].copy()
    tabla["region"] = tabla["region"].map(NOMBRE_REGION).fillna(tabla["region"])
    tabla["es_principal"] = np.where(tabla["es_principal"], "Principal", "Conminuto")
    return tabla.sort_values(["region", "fragment_id"]).rename(columns={
        "fragment_id": "Fragmento", "region": "Hueso", "es_principal": "Tipo", "distancia_mm": "Distancia (mm)",
        "volumen_cm3": "Volumen (cm³)", "n_slices": "Cortes", "mean_score": "Confianza media"})
