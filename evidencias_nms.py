"""
evidencias_nms.py -- Evidencia del NMS propio por clase (PENGWIN, semana 10). Autor: Fabián.

QUÉ HACE
  1. logits_oraculo: fabrica salidas de la cabeza grid [B, 8, 16, 16] A PARTIR DE LAS CAJAS REALES,
     imitando los errores típicos de un detector grid: la celda del centro acierta, las celdas vecinas
     repiten la misma caja un poco corrida (duplicados) y aparecen algunos falsos positivos de bajo score.
     NO es un modelo ni mide desempeño: sirve para comprobar que el mecanismo NMS hace lo que debe
     mientras el modelo entrenado de Miguel no está listo.
  2. metricas_deteccion: AP por clase, mAP@0.50, mAP@[0.50:0.95], IoU medio, FP y FN.
  3. barrido_nms: compara "sin NMS" contra varias configuraciones de NMS (iou_thresh, max_det_per_class).
  4. figura_antes_despues: cajas de un corte antes y después del NMS.

CUANDO EXISTA EL MODELO ENTRENADO
  Se reemplaza logits_oraculo(targets) por model(images) sobre VAL y se repite el mismo barrido.
  La configuración final del NMS se elige con VAL; test solo se usa en la evaluación final.

CONVENCIONES (las mismas de deteccion_grid.py)
  Cajas xyxy en píxeles de la imagen 256 x 256. Labels 1 = sacro, 2 = coxal izquierdo, 3 = coxal derecho.
"""
import numpy as np
import pandas as pd
import torch

from deteccion_grid import (GRID_S, IMG_SIZE, NMS_CONFIG, STRIDE, decode_predictions,
                            nms_por_clase, postprocess)

CLASES = (1, 2, 3)
NOMBRES = {1: "sacro", 2: "coxal_izquierdo", 3: "coxal_derecho"}
COLORES = {1: "#D9A62E", 2: "#2B8CBE", 3: "#E34A33"}                 # mismos TARGET_COLORS del notebook
IOU_THRS_COCO = np.round(np.arange(0.50, 0.951, 0.05), 2)            # 0.50, 0.55, ..., 0.95


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(np.log(p / (1 - p)))


# --------------------------------------------------------------------------------------------------
# 1. Salidas simuladas a partir de las cajas reales
# --------------------------------------------------------------------------------------------------
def logits_oraculo(targets, seed=42, ruido_px=3.0, p_vecino=0.6, n_falsos=2, p_confusion=0.1):
    """targets: lista de dicts con 'boxes' [N,4] y 'labels' [N]. Devuelve logits [B, 8, S, S] (float32).

    Por cada caja real:
      * celda del centro: objectness 0.70-0.95, caja = real + ruido gaussiano (ruido_px), clase correcta.
      * cada una de las 8 celdas vecinas, con probabilidad p_vecino: un DUPLICADO con objectness menor,
        centro forzado a quedar dentro de su celda (así funciona el grid) y tamaño +-10 %.
        Con probabilidad p_confusion el duplicado sale con otra clase.
    Por imagen: n_falsos celdas de fondo con objectness 0.05-0.40, clase y tamaño al azar.
    """
    rng = np.random.default_rng(seed)
    B, S = len(targets), GRID_S
    out = np.zeros((B, 8, S, S), dtype=np.float32)
    out[:, 0] = _logit(0.002)                                          # fondo: casi nunca hay hueso

    def poner(b, r, c, obj, box, cls_idx, cls_p):
        x1, y1, x2, y2 = box
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        out[b, 0, r, c] = _logit(obj)
        out[b, 1, r, c] = _logit(np.clip(cx / STRIDE - c, 0.02, 0.98))  # el centro no puede salir de la celda
        out[b, 2, r, c] = _logit(np.clip(cy / STRIDE - r, 0.02, 0.98))
        out[b, 3, r, c] = _logit(np.clip(w / IMG_SIZE, 0.005, 0.995))
        out[b, 4, r, c] = _logit(np.clip(h / IMG_SIZE, 0.005, 0.995))
        cls_logits = np.zeros(3, dtype=np.float32)                     # softmax([a, 0, 0]) = cls_p si a = log(2p/(1-p))
        cls_logits[cls_idx] = np.log(2 * cls_p / (1 - cls_p))
        out[b, 5:8, r, c] = cls_logits

    for b, t in enumerate(targets):
        boxes = _np(t["boxes"]).reshape(-1, 4).astype(np.float64)
        labels = _np(t["labels"]).reshape(-1).astype(int)
        centros, usadas = [], set()
        for box, lab in zip(boxes, labels):
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            r = int(np.clip(cy // STRIDE, 0, S - 1))
            c = int(np.clip(cx // STRIDE, 0, S - 1))
            obj_c = rng.uniform(0.70, 0.95)
            centros.append((r, c, obj_c, box, lab))
            for dr in (-1, 0, 1):                                      # primero los duplicados ...
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if (dr, dc) == (0, 0) or not (0 <= rr < S and 0 <= cc < S) or rng.random() > p_vecino:
                        continue
                    escala = rng.uniform(0.9, 1.1, size=2)
                    w, h = (box[2] - box[0]) * escala[0], (box[3] - box[1]) * escala[1]
                    dup = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                    cls = lab - 1 if rng.random() > p_confusion else int(rng.choice([k for k in range(3) if k != lab - 1]))
                    poner(b, rr, cc, obj_c * rng.uniform(0.3, 0.9), dup, cls, rng.uniform(0.5, 0.9))
                    usadas.add((rr, cc))
        for r, c, obj_c, box, lab in centros:                          # ... y luego los centros, que ganan si chocan
            ruido = rng.normal(0, ruido_px, size=4)
            poner(b, r, c, obj_c, box + ruido, lab - 1, rng.uniform(0.6, 0.95))
            usadas.add((r, c))
        libres = [(r, c) for r in range(S) for c in range(S) if (r, c) not in usadas]
        for k in rng.choice(len(libres), size=min(n_falsos, len(libres)), replace=False):
            r, c = libres[int(k)]
            w, h = rng.uniform(15, 80, size=2)
            cx, cy = (c + 0.5) * STRIDE, (r + 0.5) * STRIDE
            poner(b, r, c, rng.uniform(0.05, 0.40), np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]),
                  int(rng.integers(0, 3)), rng.uniform(0.4, 0.8))
    return torch.from_numpy(out)


# --------------------------------------------------------------------------------------------------
# 2. Métricas de detección
# --------------------------------------------------------------------------------------------------
def _iou_matriz(a, b):
    """IoU [N, M] entre cajas xyxy (numpy)."""
    a, b = np.asarray(a, np.float64).reshape(-1, 4), np.asarray(b, np.float64).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _emparejar(preds, targets, class_id, iou_thr):
    """Emparejamiento uno a uno, de mayor a menor score sobre TODO el conjunto.
    Cada predicción toma la caja real libre (misma imagen y clase) con mayor IoU; si IoU >= iou_thr es TP,
    si no es FP (los duplicados que llegan tarde son FP). Devuelve scores, es_tp y n_gt."""
    dets, gts, n_gt = [], {}, 0
    for i, (p, t) in enumerate(zip(preds, targets)):
        g_lab = _np(t["labels"]).reshape(-1).astype(int)
        g_box = _np(t["boxes"]).reshape(-1, 4)[g_lab == class_id]
        gts[i] = [g_box, np.zeros(len(g_box), dtype=bool)]
        n_gt += len(g_box)
        p_lab = _np(p["labels"]).reshape(-1).astype(int)
        sel = p_lab == class_id
        for s, box in zip(_np(p["scores"]).reshape(-1)[sel], _np(p["boxes"]).reshape(-1, 4)[sel]):
            dets.append((float(s), i, box))
    dets.sort(key=lambda d: -d[0])
    es_tp = np.zeros(len(dets), dtype=bool)
    for k, (_, i, box) in enumerate(dets):
        g_box, usada = gts[i]
        if len(g_box) == 0:
            continue
        ious = _iou_matriz(box, g_box)[0]
        ious[usada] = -1.0                                             # una caja real solo se empareja una vez
        j = int(ious.argmax())
        if ious[j] >= iou_thr:
            usada[j] = True
            es_tp[k] = True
    return np.array([d[0] for d in dets]), es_tp, n_gt


def average_precision(preds, targets, class_id, iou_thr=0.5):
    """AP de una clase (interpolación en todos los puntos, estilo VOC). None si la clase no tiene cajas reales."""
    _, es_tp, n_gt = _emparejar(preds, targets, class_id, iou_thr)
    if n_gt == 0:
        return None
    tp, fp = np.cumsum(es_tp), np.cumsum(~es_tp)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]                     # envolvente: precisión no creciente
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def metricas_deteccion(preds, targets):
    """Métricas de la sección 5 para detección + conteos útiles para explicar el efecto del NMS."""
    res = {}
    for c in CLASES:
        ap = average_precision(preds, targets, c, 0.5)
        res[f"ap50_{NOMBRES[c]}"] = np.nan if ap is None else ap
    aps50 = [res[f"ap50_{NOMBRES[c]}"] for c in CLASES if not np.isnan(res[f"ap50_{NOMBRES[c]}"])]
    res["map50"] = float(np.mean(aps50)) if aps50 else np.nan
    por_umbral = []
    for thr in IOU_THRS_COCO:
        aps = [a for a in (average_precision(preds, targets, c, thr) for c in CLASES) if a is not None]
        por_umbral.append(np.mean(aps) if aps else np.nan)
    validos = [v for v in por_umbral if not np.isnan(v)]
    res["map50_95"] = float(np.mean(validos)) if validos else np.nan     # sin cajas reales: NaN, nunca 0

    # IoU medio de las cajas emparejadas a IoU >= 0.5 y conteos TP / FP / FN
    tp_total, n_gt_total, ious_tp = 0, 0, []
    for c in CLASES:
        _, es_tp, n_gt = _emparejar(preds, targets, c, 0.5)
        tp_total += int(es_tp.sum())
        n_gt_total += n_gt
    for p, t in zip(preds, targets):
        for c in CLASES:
            g = _np(t["boxes"]).reshape(-1, 4)[_np(t["labels"]).reshape(-1).astype(int) == c]
            sel = _np(p["labels"]).reshape(-1).astype(int) == c
            if len(g) and sel.any():
                m = _iou_matriz(g, _np(p["boxes"]).reshape(-1, 4)[sel])
                ious_tp.extend(v for v in m.max(axis=1) if v >= 0.5)
    n_pred = int(sum(len(_np(p["scores"]).reshape(-1)) for p in preds))
    res.update({"iou_medio_tp": float(np.mean(ious_tp)) if ious_tp else np.nan,
                "n_pred": n_pred, "tp50": tp_total, "fp50": n_pred - tp_total, "fn50": n_gt_total - tp_total,
                "cajas_por_corte": n_pred / max(len(preds), 1)})
    return res


# --------------------------------------------------------------------------------------------------
# 3. Barrido de configuraciones del NMS
# --------------------------------------------------------------------------------------------------
def barrido_nms(out, targets, score_thresh=0.05, iou_thrs=(0.3, 0.4, 0.5, 0.6, 0.7), max_dets=(1, None)):
    """Compara la salida sin NMS contra cada combinación (iou_thresh, max_det_per_class). Devuelve un DataFrame."""
    filas = [{"config": "sin NMS", "iou_thresh": np.nan, "max_det_per_class": "-",
              **metricas_deteccion(decode_predictions(out, score_thresh), targets)}]
    for max_det in max_dets:
        for thr in iou_thrs:
            preds = postprocess(out, score_thresh=score_thresh, nms_fn=nms_por_clase,
                                iou_thresh=thr, max_det_per_class=max_det)
            filas.append({"config": "NMS", "iou_thresh": thr,
                          "max_det_per_class": "sin tope" if max_det is None else max_det,
                          **metricas_deteccion(preds, targets)})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------------------------------
# 4. Figura antes / después
# --------------------------------------------------------------------------------------------------
def figura_antes_despues(image, target, raw, final, titulo="", path=None):
    """image [1,H,W] o [H,W]; target con boxes/labels reales; raw y final: dicts de UNA imagen."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    img = _np(image)
    img = img[0] if img.ndim == 3 else img
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), constrained_layout=True)
    for ax, pred, nombre in ((axes[0], raw, "Antes del NMS"), (axes[1], final, "Después del NMS")):
        ax.imshow(img, cmap="gray")
        for box in _np(target["boxes"]).reshape(-1, 4):               # real: blanco discontinuo
            ax.add_patch(patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                           fill=False, ec="white", lw=1.4, ls="--"))
        boxes, scores = _np(pred["boxes"]).reshape(-1, 4), _np(pred["scores"]).reshape(-1)
        for box, s, lab in zip(boxes, scores, _np(pred["labels"]).reshape(-1).astype(int)):
            ax.add_patch(patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                           fill=False, ec=COLORES[lab], lw=1.6, alpha=0.4 + 0.6 * float(s)))
            ax.text(box[0], box[1] - 2, f"{s:.2f}", color=COLORES[lab], fontsize=7)
        ax.set_title(f"{nombre}: {len(boxes)} cajas")
        ax.axis("off")
    handles = [patches.Patch(color=COLORES[c], label=NOMBRES[c]) for c in CLASES]
    handles.append(patches.Patch(edgecolor="white", facecolor="none", ls="--", label="caja real"))
    axes[1].legend(handles=handles, loc="lower right", fontsize=8, facecolor="black", labelcolor="white")
    if titulo:
        fig.suptitle(titulo)
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig


__all__ = ["logits_oraculo", "average_precision", "metricas_deteccion", "barrido_nms",
           "figura_antes_despues", "NMS_CONFIG"]
