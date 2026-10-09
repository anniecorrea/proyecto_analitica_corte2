"""
pipeline_pengwin.py -- Unión detección -> NMS -> segmentación de instancia -> pertenencia (PENGWIN, semana 10).
Autor: Fabián.

FLUJO POR CORTE (inferir_cortes)
  imagen [B,1,256,256]
    -> backbone (pirámide f1..f4) -> CBAM/compuerta sobre f4                     (Miguel / Annie)
    -> cabeza grid -> decode_predictions (raw) -> nms_por_clase (final)          (Fabián)
    -> priors de región con las cajas FINALES -> decoder de instancias           (Annie)
    -> IDs de instancia por corte -> PertenenciaHead sobre f4 -> clase por instancia   (Fabián)
  Durante la inferencia NO se usa ninguna anotación real.

FLUJO POR VOLUMEN (inferir_volumen)
  MHA (z, y, x) en HU -> preprocesamiento por corte -> inferir_cortes por lotes
    -> restaurar máscaras a la cuadrícula original -> asociar instancias entre cortes (IDs 3D)
    -> pertenencia 3D = promedio de probabilidades por corte ponderado por área
    -> instance_volume + region_by_id: entrada directa de measure_fragment_separations (Daniela).

POR QUÉ RECIBE FUNCIONES COMO PARÁMETROS
  Las funciones de segmentación viven en el notebook (sección 23). En vez de copiarlas (dos versiones del
  mismo código), el notebook arma un diccionario `componentes` y se lo pasa al pipeline:

    componentes = {
        "backbone": backbone, "cbam": compuerta_cbam (o None), "det_head": det_head,
        "seg_decoder": seg_decoder, "cls_head": cls_head,
        "features_fn": backbone_feature_pyramid,          # 23.1
        "priors_fn": build_detection_region_priors,       # 23.2
        "decode_fn": decode_center_offset_instances,      # 23.4
        "restore_fn": restore_letterbox_label_mask,       # 23.9   (solo volumen)
        "link_fn": link_slice_instances_to_volume,        # 23.10  (solo volumen)
        "preprocess_fn": preprocesar_corte,               # HU -> ventaneo -> CLAHE -> letterbox (solo volumen)
    }

REGLAS DOCUMENTADAS
  * Clase final de una instancia = argmax de la cabeza de pertenencia. Si no coincide con la clase semántica
    del decoder o con la de la caja que la contiene, se marca conflicto; NO se corrige en silencio.
  * Una instancia que no cae dentro de ninguna caja final se CONSERVA y se marca sin_caja (no se borran
    fragmentos reales para que la salida se vea mejor).
  * Toda salida vacía conserva su forma: cajas [0,4], máscaras en ceros, listas vacías. Nunca None.
"""
import numpy as np
import pandas as pd
import torch

from deteccion_grid import NMS_CONFIG, decode_predictions, nms_por_clase, postprocess
from pertenencia import NOMBRES, clase_de_id, clasificar_instancias, instancias_de_mascara

CONFIG_PIPELINE = {
    "score_thresh": 0.05,                 # candidatas mínimas que salen del grid
    **NMS_CONFIG,                         # iou_thresh, max_det_per_class
    "center_threshold": 0.25,             # decoder de Annie (elegir con val)
    "min_center_distance": 7,
    "min_instance_pixels": 1,             # no se borran fragmentos pequeños
    "output_size": (256, 256),
}

COMPONENTES_CORTE = ("backbone", "det_head", "seg_decoder", "cls_head", "features_fn", "priors_fn", "decode_fn")
COMPONENTES_VOLUMEN = COMPONENTES_CORTE + ("restore_fn", "link_fn", "preprocess_fn")


def _cfg(cfg):
    out = dict(CONFIG_PIPELINE)
    out.update(cfg or {})
    return out


def _revisar(componentes, requeridos):
    faltan = [k for k in requeridos if componentes.get(k) is None]
    if faltan:
        raise KeyError(f"Faltan componentes del pipeline: {faltan}")


def _a_cpu(pred):
    return {k: v.detach().cpu() for k, v in pred.items()}


def _caja_que_contiene(mask, boxes, labels):
    """Índice de la caja final que contiene más píxeles de la instancia y su clase; (-1, 0) si ninguna."""
    ys, xs = np.nonzero(mask)
    mejor, mejor_n = -1, 0
    for k, box in enumerate(boxes):
        n = int(((xs >= box[0]) & (xs < box[2]) & (ys >= box[1]) & (ys < box[3])).sum())
        if n > mejor_n:
            mejor, mejor_n = k, n
    return mejor, (int(labels[mejor]) if mejor >= 0 else 0)


def _poner_en_eval(componentes):
    for k in ("backbone", "cbam", "det_head", "seg_decoder", "cls_head"):
        m = componentes.get(k)
        if isinstance(m, torch.nn.Module):
            m.eval()


# --------------------------------------------------------------------------------------------------
# 1. Inferencia por lote de cortes
# --------------------------------------------------------------------------------------------------
@torch.no_grad()
def inferir_cortes(componentes, images, cfg=None):
    """images [B,1,H,W] (tensor, en el dispositivo del modelo). Devuelve una lista de B dicts:
      raw / final: dicts de detección (boxes, scores, labels, class_probs, obj, cells) antes / después del NMS
      instance_mask [H,W] uint16 (IDs locales) | semantic_mask [H,W] uint8 | class_map [H,W] uint8 (clase final)
      instancias: lista de dicts {id, area_px, probs[3], clase, clase_semantica, caja, clase_caja,
                                  sin_caja, conflicto_semantica, conflicto_caja}
    """
    _revisar(componentes, COMPONENTES_CORTE)
    cfg = _cfg(cfg)
    _poner_en_eval(componentes)
    if images.ndim != 4:
        raise ValueError(f"images debe ser [B,1,H,W]; llegó {tuple(images.shape)}")
    B = images.shape[0]
    if B == 0:
        return []

    feats = list(componentes["features_fn"](componentes["backbone"], images))
    if componentes.get("cbam") is not None:
        refinado = componentes["cbam"](feats[-1])
        feats[-1] = refinado[0] if isinstance(refinado, tuple) else refinado
    f4 = feats[-1]

    det_out = componentes["det_head"](f4)
    raw = decode_predictions(det_out, score_thresh=cfg["score_thresh"])
    final = postprocess(det_out, score_thresh=cfg["score_thresh"], nms_fn=nms_por_clase,
                        iou_thresh=cfg["iou_thresh"], max_det_per_class=cfg["max_det_per_class"])

    priors = componentes["priors_fn"](final, output_size=cfg["output_size"], device=images.device)
    seg_out = componentes["seg_decoder"](images, feats, priors, output_size=cfg["output_size"])

    inst_masks, sem_masks, sem_classes = [], [], []
    for b in range(B):
        dec = componentes["decode_fn"](seg_out, center_threshold=cfg["center_threshold"],
                                       min_center_distance=cfg["min_center_distance"],
                                       min_instance_pixels=cfg["min_instance_pixels"], batch_index=b)
        inst_masks.append(np.asarray(dec[0]).astype(np.uint16))
        sem_masks.append(np.asarray(dec[1]).astype(np.uint8))
        sem_classes.append(dict(dec[2]))

    probs = clasificar_instancias(componentes["cls_head"], f4, inst_masks)

    salida = []
    for b in range(B):
        fin = _a_cpu(final[b])
        boxes, labels = fin["boxes"].numpy().reshape(-1, 4), fin["labels"].numpy().reshape(-1)
        inst = inst_masks[b]
        class_map = np.zeros(inst.shape, dtype=np.uint8)
        instancias = []
        for i, p in sorted(probs[b].items()):
            mask = inst == i
            clase = int(np.argmax(p)) + 1
            k_caja, clase_caja = _caja_que_contiene(mask, boxes, labels)
            clase_sem = int(sem_classes[b].get(i, 0))
            class_map[mask] = clase
            instancias.append({
                "id": int(i), "area_px": int(mask.sum()), "probs": p.astype(np.float32), "clase": clase,
                "clase_semantica": clase_sem, "caja": k_caja, "clase_caja": clase_caja,
                "sin_caja": k_caja < 0,
                "conflicto_semantica": clase_sem not in (0, clase),
                "conflicto_caja": clase_caja not in (0, clase),
            })
        salida.append({"raw": _a_cpu(raw[b]), "final": fin, "instance_mask": inst,
                       "semantic_mask": sem_masks[b], "class_map": class_map, "instancias": instancias})
    return salida


# --------------------------------------------------------------------------------------------------
# 2. Inferencia por volumen (MHA completo)
# --------------------------------------------------------------------------------------------------
COLUMNAS_FRAGMENTOS = ["case_id", "fragment_id", "region", "clase", "prob_sacro", "prob_coxal_izquierdo",
                       "prob_coxal_derecho", "clase_enlace", "conflicto_enlace", "n_voxels", "n_cortes"]


@torch.no_grad()
def inferir_volumen(componentes, volume_hu_zyx, spacing_xyz, cfg=None, batch_size=8, device=None, case_id=None):
    """volume_hu_zyx: array (z, y, x) en HU leído del MHA. spacing_xyz: GetSpacing() del MHA (orden x, y, z).
    Devuelve dict con:
      instance_volume (z,y,x) int32 con IDs 3D | region_by_id {id: región} -> measure_fragment_separations
      fragmentos: DataFrame (una fila por fragmento 3D) | cortes: DataFrame (una fila por corte) | eventos del enlace.
    """
    _revisar(componentes, COMPONENTES_VOLUMEN)
    cfg = _cfg(cfg)
    vol = np.asarray(volume_hu_zyx)
    if vol.ndim != 3:
        raise ValueError(f"Se esperaba volumen (z, y, x); llegó {vol.shape}")
    if device is None:
        device = next(componentes["det_head"].parameters()).device
    Z, H, W = vol.shape

    inst_full, cls_full, probs_z, area_z, filas_cortes = [], [], [], [], []
    for inicio in range(0, Z, batch_size):
        zs = range(inicio, min(inicio + batch_size, Z))
        prep = [componentes["preprocess_fn"](vol[z]) for z in zs]
        x = torch.from_numpy(np.stack([np.asarray(p[0], dtype=np.float32) for p in prep]))[:, None].to(device)
        resultados = inferir_cortes(componentes, x, cfg)
        for z, (_, transform), r in zip(zs, prep, resultados):
            inst = np.asarray(componentes["restore_fn"](r["instance_mask"], (H, W), transform)).astype(np.int64)
            cls = np.asarray(componentes["restore_fn"](r["class_map"], (H, W), transform)).astype(np.uint8)
            inst_full.append(inst)
            cls_full.append(cls)
            ids_presentes = set(int(v) for v in np.unique(inst) if v != 0)
            probs_z.append({d["id"]: d["probs"] for d in r["instancias"] if d["id"] in ids_presentes})
            area_z.append({i: int((inst == i).sum()) for i in ids_presentes})
            filas_cortes.append({
                "case_id": case_id, "z": z, "n_cajas_raw": int(r["raw"]["boxes"].shape[0]),
                "n_cajas_final": int(r["final"]["boxes"].shape[0]), "n_instancias": len(r["instancias"]),
                "n_sin_caja": sum(d["sin_caja"] for d in r["instancias"]),
                "n_conflicto_semantica": sum(d["conflicto_semantica"] for d in r["instancias"]),
                "n_conflicto_caja": sum(d["conflicto_caja"] for d in r["instancias"]),
            })

    inst_zyx = np.stack(inst_full) if inst_full else np.zeros((0, H, W), dtype=np.int64)
    cls_zyx = np.stack(cls_full) if cls_full else np.zeros((0, H, W), dtype=np.uint8)
    volume_ids, volume_classes, slice_maps, eventos = componentes["link_fn"](inst_zyx, cls_zyx)

    fragmentos = agregar_pertenencia_3d(slice_maps, probs_z, area_z, volume_classes, case_id=case_id)
    region_by_id = {int(r.fragment_id): r.region for r in fragmentos.itertuples(index=False)}
    return {"case_id": case_id, "spacing_xyz": tuple(float(s) for s in spacing_xyz),
            "instance_volume": np.asarray(volume_ids).astype(np.int32), "region_by_id": region_by_id,
            "fragmentos": fragmentos, "cortes": pd.DataFrame(filas_cortes), "eventos": list(eventos)}


def agregar_pertenencia_3d(slice_maps, probs_z, area_z, volume_classes=None, case_id=None):
    """Probabilidades por corte -> pertenencia por fragmento 3D (promedio ponderado por área).
    slice_maps[z]: {id_local: id_3d} (salida de link_slice_instances_to_volume)."""
    acum = {}
    for z, mapa in enumerate(slice_maps):
        for local, global_id in mapa.items():
            if local not in probs_z[z]:
                continue
            a = area_z[z].get(local, 0)
            d = acum.setdefault(int(global_id), {"suma": np.zeros(3), "area": 0, "cortes": 0})
            d["suma"] += a * np.asarray(probs_z[z][local], dtype=np.float64)
            d["area"] += a
            d["cortes"] += 1
    filas = []
    for gid, d in sorted(acum.items()):
        p = d["suma"] / max(d["area"], 1)
        clase = int(np.argmax(p)) + 1
        clase_enlace = int((volume_classes or {}).get(gid, 0))
        filas.append({"case_id": case_id, "fragment_id": gid, "region": NOMBRES[clase], "clase": clase,
                      "prob_sacro": p[0], "prob_coxal_izquierdo": p[1], "prob_coxal_derecho": p[2],
                      "clase_enlace": clase_enlace, "conflicto_enlace": clase_enlace not in (0, clase),
                      "n_voxels": int(d["area"]), "n_cortes": d["cortes"]})
    return pd.DataFrame(filas, columns=COLUMNAS_FRAGMENTOS)


# --------------------------------------------------------------------------------------------------
# 3. Evaluación de la pertenencia (instancias reales o predichas)
# --------------------------------------------------------------------------------------------------
def _emparejar_instancias(gt_mask, pred_mask, iou_min=0.5):
    """Hungarian sobre IoU entre instancias reales y predichas del mismo corte. Devuelve {gt_id: (pred_id, iou)}."""
    from scipy.optimize import linear_sum_assignment

    g_masks, g_ids = instancias_de_mascara(gt_mask)
    p_masks, p_ids = instancias_de_mascara(pred_mask)
    if len(g_ids) == 0 or len(p_ids) == 0:
        return {}
    g = g_masks.reshape(len(g_ids), -1).float()
    p = p_masks.reshape(len(p_ids), -1).float()
    inter = g @ p.T
    union = g.sum(1, keepdim=True) + p.sum(1)[None, :] - inter
    iou = (inter / union.clamp(min=1)).numpy()
    filas, cols = linear_sum_assignment(-iou)
    return {int(g_ids[r]): (int(p_ids[c]), float(iou[r, c])) for r, c in zip(filas, cols) if iou[r, c] >= iou_min}


@torch.no_grad()
def evaluar_pertenencia(componentes, images, targets, cfg=None, modo="pred", iou_min=0.5):
    """Una fila por instancia REAL. modo='gt': la cabeza clasifica las máscaras reales (aísla la clasificación).
    modo='pred': clasifica las instancias predichas emparejadas por IoU >= iou_min; las no emparejadas quedan
    con clase 0 (no detectada) y cuentan como error, no se excluyen.
    Columnas: case_id, slice_index, gt_id, clase_real, iou, prob_*, pred_cabeza, pred_semantica, pred_caja."""
    from pertenencia import regla_caja_detectada, regla_mayoria_semantica

    if modo not in ("gt", "pred"):
        raise ValueError("modo debe ser 'gt' o 'pred'")
    res = inferir_cortes(componentes, images, cfg)
    filas = []
    if modo == "gt":
        feats = list(componentes["features_fn"](componentes["backbone"], images))
        if componentes.get("cbam") is not None:
            r = componentes["cbam"](feats[-1])
            feats[-1] = r[0] if isinstance(r, tuple) else r
        probs_gt = clasificar_instancias(componentes["cls_head"], feats[-1], [t["instance_mask"] for t in targets])
    for b, (r, t) in enumerate(zip(res, targets)):
        gt_mask = np.asarray(t["instance_mask"].cpu() if torch.is_tensor(t["instance_mask"]) else t["instance_mask"])
        sem_rule = regla_mayoria_semantica(gt_mask if modo == "gt" else r["instance_mask"], r["semantic_mask"])
        box_rule = regla_caja_detectada(gt_mask if modo == "gt" else r["instance_mask"],
                                        r["final"]["boxes"], r["final"]["labels"])
        pares = None if modo == "gt" else _emparejar_instancias(gt_mask, r["instance_mask"], iou_min)
        probs_pred = {d["id"]: d["probs"] for d in r["instancias"]}
        for gid in [int(v) for v in np.unique(gt_mask) if v != 0]:
            if modo == "gt":
                key, iou, p = gid, 1.0, probs_gt[b].get(gid)
            else:
                key, iou = pares.get(gid, (None, 0.0))
                p = probs_pred.get(key) if key is not None else None
            fila = {"case_id": t.get("case_id"), "slice_index": t.get("slice_index"), "gt_id": gid,
                    "clase_real": clase_de_id(gid), "iou": iou,
                    "prob_sacro": np.nan, "prob_coxal_izquierdo": np.nan, "prob_coxal_derecho": np.nan,
                    "pred_cabeza": 0, "pred_semantica": 0, "pred_caja": 0}
            if p is not None:
                fila.update({"prob_sacro": p[0], "prob_coxal_izquierdo": p[1], "prob_coxal_derecho": p[2],
                             "pred_cabeza": int(np.argmax(p)) + 1,
                             "pred_semantica": sem_rule.get(key, 0), "pred_caja": box_rule.get(key, 0)})
            filas.append(fila)
    return pd.DataFrame(filas)


def resumen_evaluacion_pertenencia(df):
    """Tabla comparativa: cabeza de pertenencia vs. reglas de mayoría semántica y de caja detectada.
    F1 y exactitud se calculan sobre TODAS las instancias reales (las no detectadas cuentan como error).
    El AUC solo existe para la cabeza (las reglas no dan probabilidades) y usa las instancias con probabilidad."""
    from pertenencia import metricas_pertenencia

    filas = []
    con_prob = df.dropna(subset=["prob_sacro"])
    for nombre, col in (("cabeza_pertenencia", "pred_cabeza"), ("regla_mayoria_semantica", "pred_semantica"),
                        ("regla_caja_detectada", "pred_caja")):
        m = metricas_pertenencia(df["clase_real"].to_numpy(), y_pred=df[col].to_numpy())
        fila = {"metodo": nombre, "n_instancias": m["n"], "n_sin_clase": int((df[col] == 0).sum()),
                "exactitud": m["exactitud"], "f1_macro": m["f1_macro"],
                **{k: v for k, v in m.items() if k.startswith("f1_") and k != "f1_macro"}}
        if col == "pred_cabeza" and len(con_prob):
            probs = con_prob[["prob_sacro", "prob_coxal_izquierdo", "prob_coxal_derecho"]].to_numpy()
            fila["auc_macro_ovr"] = metricas_pertenencia(con_prob["clase_real"].to_numpy(), probs=probs)["auc_macro_ovr"]
        else:
            fila["auc_macro_ovr"] = np.nan
        filas.append(fila)
    return pd.DataFrame(filas)
