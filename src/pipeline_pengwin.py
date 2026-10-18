"""
pipeline_pengwin.py -- Unión detección -> NMS -> segmentación de instancia -> pertenencia (PENGWIN, semana 10).

FLUJO POR CORTE (inferir_cortes)
  imagen [B,C,256,256]   (C = 1, o 3 con entrada 2.5D: cortes z-1, z, z+1)
    -> backbone (pirámide f1..f4) -> CBAM/compuerta sobre f4
    -> cabeza grid -> decode_predictions (raw) -> nms_por_clase (final)
    -> priors de región con las cajas FINALES -> decoder de instancias
    -> IDs de instancia por corte -> PertenenciaHead sobre f4 -> clase por instancia
  Durante la inferencia NO se usa ninguna anotación real.

FLUJO POR VOLUMEN (inferir_volumen)
  MHA (z, y, x) en HU -> preprocesamiento por corte -> inferir_cortes por lotes
    -> restaurar máscaras a la cuadrícula original -> asociar instancias entre cortes (IDs 3D)
    -> pertenencia 3D = promedio de probabilidades por corte ponderado por área
    -> instance_volume + region_by_id: entrada directa de measure_fragment_separations (src/distancias_fragmentos.py).

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
        "in_channels": 1,                                 # opcional: 3 si el modelo usa entrada 2.5D (solo volumen)
    }
"""
import numpy as np
import pandas as pd
import torch

from src.deteccion_grid import NMS_CONFIG, decode_predictions, nms_por_clase, postprocess
from src.entrada_25d import apilar_cortes
from src.pertenencia import NOMBRES, clase_de_id, clasificar_instancias, instancias_de_mascara

CONFIG_PIPELINE = {
    "score_thresh": 0.05,                 # candidatas mínimas que salen del grid
    **NMS_CONFIG,                         # iou_thresh, max_det_per_class
    "center_threshold": 0.25,             # decoder de instancias (elegir con val)
    "min_center_distance": 7,
    "min_instance_pixels": 1,             # no se borran fragmentos pequeños
    "output_size": (256, 256),
    "umbral_corte": None,                 # None = procesar todos los cortes; float = filtro de cortes sin hueso
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


def _corte_procesado(final_b, score_max, umbral_corte):
    """Un corte pasa al decoder si el filtro está apagado (None) o si tiene caja final con score >= umbral."""
    if umbral_corte is None:
        return True
    return final_b["boxes"].shape[0] > 0 and score_max >= float(umbral_corte)


def _poner_en_eval(componentes):
    for k in ("backbone", "cbam", "det_head", "seg_decoder", "cls_head"):
        m = componentes.get(k)
        if isinstance(m, torch.nn.Module):
            m.eval()


def mapa_compartido(componentes, images):
    """Piramide del backbone con CBAM aplicado al ultimo nivel. Devuelve (feats, f4).
    f4 es el mapa que comparten las tres cabezas: deteccion, decoder (via la piramide) y pertenencia."""
    feats = list(componentes["features_fn"](componentes["backbone"], images))
    if componentes.get("cbam") is not None:
        refinado = componentes["cbam"](feats[-1])
        feats[-1] = refinado[0] if isinstance(refinado, tuple) else refinado
    return feats, feats[-1]


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

    feats, f4 = mapa_compartido(componentes, images)

    det_out = componentes["det_head"](f4)
    raw = decode_predictions(det_out, score_thresh=cfg["score_thresh"])
    final = postprocess(det_out, score_thresh=cfg["score_thresh"], nms_fn=nms_por_clase,
                        iou_thresh=cfg["iou_thresh"], max_det_per_class=cfg["max_det_per_class"])

    # Compuerta por corte: solo pasan al decoder los cortes cuya mejor caja final supera umbral_corte.
    score_max = [float(f["scores"].max()) if f["scores"].numel() else 0.0 for f in final]
    procesado = [_corte_procesado(f, s, cfg["umbral_corte"]) for f, s in zip(final, score_max)]
    idx = [b for b in range(B) if procesado[b]]

    H, W = cfg["output_size"]
    inst_masks = [np.zeros((H, W), dtype=np.uint16) for _ in range(B)]
    sem_masks = [np.zeros((H, W), dtype=np.uint8) for _ in range(B)]
    sem_classes = [{} for _ in range(B)]
    if idx:
        sel = torch.tensor(idx, device=images.device)
        feats_sel = [f.index_select(0, sel) if torch.is_tensor(f) else f for f in feats]
        priors = componentes["priors_fn"]([final[b] for b in idx], output_size=cfg["output_size"],
                                          device=images.device)
        seg_out = componentes["seg_decoder"](images.index_select(0, sel), feats_sel, priors,
                                             output_size=cfg["output_size"])
        for j, b in enumerate(idx):
            dec = componentes["decode_fn"](seg_out, center_threshold=cfg["center_threshold"],
                                           min_center_distance=cfg["min_center_distance"],
                                           min_instance_pixels=cfg["min_instance_pixels"], batch_index=j)
            inst_masks[b] = np.asarray(dec[0]).astype(np.uint16)
            sem_masks[b] = np.asarray(dec[1]).astype(np.uint8)
            sem_classes[b] = dict(dec[2])

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
                       "semantic_mask": sem_masks[b], "class_map": class_map, "instancias": instancias,
                       "score_max": score_max[b], "procesado": procesado[b]})
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
    n_canales = componentes.get("in_channels", 1)     # 3 si el modelo se entrenó con entrada 2.5D

    inst_full, cls_full, probs_z, area_z, filas_cortes = [], [], [], [], []
    for inicio in range(0, Z, batch_size):
        zs = range(inicio, min(inicio + batch_size, Z))
        prep = [componentes["preprocess_fn"](vol[z]) for z in zs]
        if n_canales == 1:
            x = torch.from_numpy(np.stack([np.asarray(p[0], dtype=np.float32) for p in prep]))[:, None].to(device)
        else:                                  # entrada 2.5D: cada corte con sus vecinos z-1, z+1 (src/entrada_25d.py)
            x = torch.from_numpy(apilar_cortes(vol, zs, lambda corte: componentes["preprocess_fn"](corte)[0],
                                               n_canales)).to(device)
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
                "n_cajas_final": int(r["final"]["boxes"].shape[0]), "score_max": r["score_max"],
                "procesado": r["procesado"], "n_instancias": len(r["instancias"]),
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


@torch.no_grad()
def scores_por_corte(componentes, images, cfg=None):
    """Mejor score después del NMS en cada corte (0.0 si no queda ninguna caja). Solo usa tronco + cabeza grid:
    es lo que decide el filtro de cortes, sin pasar por el decoder. images [B,1,H,W] -> np.ndarray [B]."""
    _revisar(componentes, ("backbone", "det_head", "features_fn"))
    cfg = _cfg(cfg)
    _poner_en_eval(componentes)
    if images.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    _, f4 = mapa_compartido(componentes, images)
    final = postprocess(componentes["det_head"](f4), score_thresh=cfg["score_thresh"], nms_fn=nms_por_clase,
                        iou_thresh=cfg["iou_thresh"], max_det_per_class=cfg["max_det_per_class"])
    return np.array([float(f["scores"].max()) if f["scores"].numel() else 0.0 for f in final], dtype=np.float32)


def calibrar_umbral_corte(scores, tiene_hueso, recall_min=0.99, paso=0.01):
    """Elige umbral_corte con cortes de val etiquetados (tiene_hueso = el label del corte tiene algún vóxel de pelvis).
    Regla: el MAYOR umbral que conserva al menos `recall_min` de los cortes con hueso. Se prioriza no perder cortes
    con hueso (un corte omitido borra sus fragmentos) sobre descartar cortes vacíos.
    Devuelve (umbral, tabla) con, para cada umbral candidato: recall_hueso (cortes con hueso que se procesan)
    y rechazo_vacios (cortes sin hueso que se omiten)."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    pos = np.asarray(tiene_hueso, dtype=bool).reshape(-1)
    if scores.shape != pos.shape:
        raise ValueError("scores y tiene_hueso deben tener la misma longitud")
    if not pos.any():
        raise ValueError("Se necesita al menos un corte con hueso para calibrar")
    filas = []
    for u in np.round(np.arange(0.0, 1.0 + 1e-9, paso), 6):
        procesa = (scores > 0) & (scores >= u)          # igual que _corte_procesado: sin cajas = no se procesa
        filas.append({"umbral": float(u), "recall_hueso": float(procesa[pos].mean()),
                      "rechazo_vacios": float((~procesa[~pos]).mean()) if (~pos).any() else float("nan"),
                      "cortes_procesados": int(procesa.sum())})
    tabla = pd.DataFrame(filas)
    validos = tabla[tabla["recall_hueso"] >= recall_min]
    umbral = float(validos["umbral"].max()) if len(validos) else 0.0
    return umbral, tabla


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
def probs_pertenencia_gt(componentes, images, targets):
    """Probabilidades de la cabeza sobre las mascaras REALES (modo 'gt'). Lista de B dicts {gt_id: probs[3]}."""
    _, f4 = mapa_compartido(componentes, images)
    return clasificar_instancias(componentes["cls_head"], f4, [t["instance_mask"] for t in targets])


def filas_pertenencia(res, targets, modo="pred", probs_gt=None, iou_min=0.5):
    """Una fila por instancia REAL, a partir de salidas YA calculadas por inferir_cortes (no vuelve a inferir).
    modo='gt': la cabeza clasifica las mascaras reales (requiere probs_gt de probs_pertenencia_gt): aisla la
    clasificacion. modo='pred': clasifica las instancias predichas emparejadas por IoU >= iou_min; las no
    emparejadas quedan con clase 0 (no detectada) y cuentan como error, no se excluyen.
    Columnas: case_id, slice_index, gt_id, clase_real, iou, prob_*, pred_cabeza, pred_semantica, pred_caja,
    area_gt_px (fragmento real), pred_id y area_pred_px (mascara que clasifico la cabeza; 0 si no hubo)."""
    from src.pertenencia import regla_caja_detectada, regla_mayoria_semantica

    if modo not in ("gt", "pred"):
        raise ValueError("modo debe ser 'gt' o 'pred'")
    if modo == "gt" and probs_gt is None:
        raise ValueError("modo='gt' necesita probs_gt (ver probs_pertenencia_gt)")
    filas = []
    for b, (r, t) in enumerate(zip(res, targets)):
        gt_mask = np.asarray(t["instance_mask"].cpu() if torch.is_tensor(t["instance_mask"]) else t["instance_mask"])
        sem_rule = regla_mayoria_semantica(gt_mask if modo == "gt" else r["instance_mask"], r["semantic_mask"])
        box_rule = regla_caja_detectada(gt_mask if modo == "gt" else r["instance_mask"],
                                        r["final"]["boxes"], r["final"]["labels"])
        pares = None if modo == "gt" else _emparejar_instancias(gt_mask, r["instance_mask"], iou_min)
        probs_pred = {d["id"]: d["probs"] for d in r["instancias"]}
        area_pred = {d["id"]: d["area_px"] for d in r["instancias"]}
        for gid in [int(v) for v in np.unique(gt_mask) if v != 0]:
            area_gt = int((gt_mask == gid).sum())
            if modo == "gt":
                key, iou, p = gid, 1.0, probs_gt[b].get(gid)
                area_clasificada = area_gt
            else:
                key, iou = pares.get(gid, (None, 0.0))
                p = probs_pred.get(key) if key is not None else None
                area_clasificada = area_pred.get(key, 0) if key is not None else 0
            fila = {"case_id": t.get("case_id"), "slice_index": t.get("slice_index"), "gt_id": gid,
                    "clase_real": clase_de_id(gid), "iou": iou,
                    "prob_sacro": np.nan, "prob_coxal_izquierdo": np.nan, "prob_coxal_derecho": np.nan,
                    "pred_cabeza": 0, "pred_semantica": 0, "pred_caja": 0,
                    "area_gt_px": area_gt, "pred_id": 0 if (modo == "gt" or key is None) else int(key),
                    "area_pred_px": int(area_clasificada)}
            if p is not None:
                fila.update({"prob_sacro": p[0], "prob_coxal_izquierdo": p[1], "prob_coxal_derecho": p[2],
                             "pred_cabeza": int(np.argmax(p)) + 1,
                             "pred_semantica": sem_rule.get(key, 0), "pred_caja": box_rule.get(key, 0)})
            filas.append(fila)
    return pd.DataFrame(filas)


@torch.no_grad()
def evaluar_pertenencia(componentes, images, targets, cfg=None, modo="pred", iou_min=0.5):
    """Corre el sistema (inferir_cortes) y devuelve la tabla por instancia REAL de filas_pertenencia.
    Para evaluar un split completo sin inferir dos veces, ver src/evaluacion.py (usa filas_pertenencia)."""
    if modo not in ("gt", "pred"):
        raise ValueError("modo debe ser 'gt' o 'pred'")
    res = inferir_cortes(componentes, images, cfg)
    probs_gt = probs_pertenencia_gt(componentes, images, targets) if modo == "gt" else None
    return filas_pertenencia(res, targets, modo, probs_gt, iou_min)


def resumen_evaluacion_pertenencia(df):
    """Tabla comparativa: cabeza de pertenencia vs. reglas de mayoría semántica y de caja detectada.
    F1 y exactitud se calculan sobre TODAS las instancias reales (las no detectadas cuentan como error).
    El AUC solo existe para la cabeza (las reglas no dan probabilidades) y usa las instancias con probabilidad."""
    from src.pertenencia import metricas_pertenencia

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
