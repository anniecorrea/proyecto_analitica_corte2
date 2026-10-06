"""
deteccion_grid.py -- Cabeza de detección tipo grid y pérdidas (PENGWIN, semana 9). Autor: Fabián.
Mismas funciones que el notebook `cabeza_grid_final.ipynb` (ahí están explicadas y probadas).

CONVENCIONES
  * Imagen [B, 1, 256, 256]; grid de 16 x 16 celdas (cada celda = 16 px); mapa de entrada [B, 256, 16, 16].
  * Cajas xyxy en píxeles de la imagen de 256 x 256. Labels del dataset: 1 = sacro, 2 = coxal izquierdo, 3 = coxal derecho.
  * Salida de la cabeza: LOGITS [B, 8, 16, 16] -> canal 0 = objectness | 1-4 = tx, ty, tw, th | 5-7 = clases.
  * tx, ty = posición del centro dentro de su celda; tw, th = ancho y alto divididos entre 256.

USO
    head = GridDetectionHead()
    out = head(fmap)                                   # fmap: [B, 256, 16, 16] (backbone + CBAM)
    T = build_targets(targets)                         # targets: lista de dicts con 'boxes' y 'labels'
    L = grid_detection_loss(out, T)                    # dict: total, obj, box, cls, mean_iou...
    preds = postprocess(out, nms_fn=None)              # el NMS propio se enchufa en nms_fn
Para probar otros pesos, pasarlos SIEMPRE explícitos: grid_detection_loss(out, T, lambdas=mis_lambdas).

CONTRATO CON LAS OTRAS PIEZAS (backbone -> CBAM -> esta cabeza)
  * Backbone (Miguel): devuelve la tupla (pooled, fmap), con fmap [B, 256, 16, 16].
  * CBAM / compuerta residual (Annie): recibe fmap y devuelve la tupla (y, ch_attn, sp_attn), con y [B, 256, 16, 16].
  * Esta cabeza recibe UN tensor [B, 256, 16, 16]: al empalmar, desempaquetar las tuplas y pasarle solo y.
        _, fmap = backbone(x)
        fmap, _, _ = cbam(fmap)
        out = head(fmap)

AJUSTAR CON EL DATASET COMPLETO
  * LAMBDAS y los priors de GridDetectionHead (obj_prior = 0.0078, wh_prior = 0.20) se eligieron con las
    12 cajas del batch fijo y mapas sintéticos: recalcularlos/reconfirmarlos con el modelo completo.

LIMITACIONES CONOCIDAS
  * Una sola celda positiva por objeto; si dos centros caen en la misma celda, gana la caja más grande.
  * Cajas muy pequeñas (la de 2 x 4 px del caso 001) no convergen: la sigmoide de tx, ty se satura en el borde de la celda.
  * build_targets usa un ciclo de Python y devuelve tensores en CPU (la pérdida los mueve al dispositivo);
    en un entrenamiento largo conviene llamarlo en el collate_fn del DataLoader.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

GRID_S = 16
IMG_SIZE = 256
STRIDE = IMG_SIZE // GRID_S
CH_OBJ, CH_BOX, CH_CLS = 0, slice(1, 5), slice(5, 8)
CLASS_NAMES = {1: "sacro", 2: "coxal_izquierdo", 3: "coxal_derecho"}

def box_iou(boxes1, boxes2, eps=1e-7):
    """IoU entre cajas xyxy en píxeles. Acepta formas [..., 4] compatibles por broadcasting."""
    b1, b2 = boxes1.float(), boxes2.float()
    inter_w = (torch.minimum(b1[..., 2], b2[..., 2]) - torch.maximum(b1[..., 0], b2[..., 0])).clamp(min=0)   # clamp: si no se tocan, 0
    inter_h = (torch.minimum(b1[..., 3], b2[..., 3]) - torch.maximum(b1[..., 1], b2[..., 1])).clamp(min=0)
    inter = inter_w * inter_h
    area1 = (b1[..., 2] - b1[..., 0]).clamp(min=0) * (b1[..., 3] - b1[..., 1]).clamp(min=0)
    area2 = (b2[..., 2] - b2[..., 0]).clamp(min=0) * (b2[..., 3] - b2[..., 1]).clamp(min=0)
    return inter / (area1 + area2 - inter + eps)           # eps: evita 0/0 con cajas de área cero


def pairwise_iou(boxes1, boxes2):
    """Matriz [N, M] con el IoU de cada caja de boxes1 contra cada caja de boxes2 (la usará el NMS)."""
    return box_iou(boxes1[:, None, :], boxes2[None, :, :])


def encode_boxes(boxes_xyxy, img_size=IMG_SIZE, grid_s=GRID_S):
    """Cajas xyxy en px [N,4] -> (fila [N], columna [N], t [N,4] = tx, ty, tw, th)."""
    stride = img_size / grid_s
    b = boxes_xyxy.float()
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2          # centro de la caja
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    col = (cx / stride).floor().clamp(0, grid_s - 1).long()             # celda que contiene el centro
    row = (cy / stride).floor().clamp(0, grid_s - 1).long()
    t = torch.stack([cx / stride - col, cy / stride - row, w / img_size, h / img_size], dim=1)   # posición dentro de la celda y tamaño relativo
    return row, col, t


def decode_boxes(row, col, t, img_size=IMG_SIZE, grid_s=GRID_S):
    """Inversa de encode_boxes: (fila, columna, t) -> cajas xyxy en px [N,4]."""
    stride = img_size / grid_s
    cx = (col.float() + t[:, 0]) * stride
    cy = (row.float() + t[:, 1]) * stride
    w, h = t[:, 2] * img_size, t[:, 3] * img_size
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)


def build_targets(targets, grid_s=GRID_S, img_size=IMG_SIZE, num_classes=3):
    """targets: lista de dicts con 'boxes' [N,4] (xyxy, px) y 'labels' [N] (1..num_classes).
    Devuelve obj_t [B,S,S], box_t [B,S,S,4], cls_t [B,S,S] (0..2; -1 en el fondo) y n_collisions."""
    B = len(targets)
    obj_t = torch.zeros(B, grid_s, grid_s)
    box_t = torch.zeros(B, grid_s, grid_s, 4)
    cls_t = torch.full((B, grid_s, grid_s), -1, dtype=torch.long)
    n_collisions = 0
    for b, tgt in enumerate(targets):
        boxes = torch.as_tensor(tgt["boxes"]).float().reshape(-1, 4)
        labels = torch.as_tensor(tgt["labels"]).long().reshape(-1)
        if boxes.shape[0] == 0:
            continue                                                   # imagen sin cajas: todo es fondo
        if labels.min() < 1 or labels.max() > num_classes:
            raise ValueError(f"labels fuera de 1..{num_classes}: {labels.tolist()}")
        row, col, t = encode_boxes(boxes, img_size, grid_s)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        for k in torch.argsort(areas, stable=True).tolist():           # de menor a mayor área: en una colisión gana la más grande
            i, j = row[k].item(), col[k].item()
            if obj_t[b, i, j] == 1:
                n_collisions += 1
            obj_t[b, i, j] = 1
            box_t[b, i, j] = t[k]
            cls_t[b, i, j] = labels[k] - 1                             # labels 1..3 -> 0..2
    return {"obj_t": obj_t, "box_t": box_t, "cls_t": cls_t, "n_collisions": n_collisions}


class GridDetectionHead(nn.Module):
    """Mapa [B, 256, S, S] -> logits [B, 8, S, S]: canal 0 = objectness, 1-4 = tx, ty, tw, th, 5-7 = clases."""
    def __init__(self, in_channels=256, hidden=128, num_classes=3, obj_prior=0.0078, wh_prior=0.20):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),   # 3x3: cada celda mira a sus vecinas
            nn.GroupNorm(8, hidden),                                    # no depende del tamaño del batch
            nn.ReLU(inplace=True),
        )
        self.pred = nn.Conv2d(hidden, 1 + 4 + num_classes, 1)
        with torch.no_grad():                                           # sesgos iniciales (priors)
            self.pred.bias.zero_()                                      # tx, ty = 0.5: el centro de la celda
            self.pred.bias[CH_OBJ] = math.log(obj_prior / (1 - obj_prior))   # al inicio: "casi nunca hay hueso"
            self.pred.bias[3:5] = math.log(wh_prior / (1 - wh_prior))        # tamaño inicial de la caja

    def forward(self, x):
        return self.pred(self.conv(x))


def split_head_output(out):
    """[B,8,S,S] -> obj_logit [B,S,S], box_logit [B,S,S,4], cls_logit [B,S,S,3]."""
    return out[:, CH_OBJ], out[:, CH_BOX].permute(0, 2, 3, 1), out[:, CH_CLS].permute(0, 2, 3, 1)


LAMBDAS = {"obj": 0.46, "box": 2.90, "cls": 2.03,     # elegidos comparando 4 configuraciones con 3 semillas (config B)
           "noobj": 1.0, "l1": 1.0, "iou": 1.0}       # noobj: peso de las celdas de fondo | l1, iou: las dos partes de la pérdida de caja


def grid_detection_loss(out, tgt, lambdas=LAMBDAS):
    """out: logits [B,8,S,S]; tgt: salida de build_targets. Devuelve un dict con total, obj, box, cls, mean_iou, etc.
    Regla: objectness se calcula en TODAS las celdas; caja y clase solo en las celdas con hueso."""
    out = out.float()                                                  # la pérdida siempre en float32 (compatible con AMP)
    obj_logit, box_logit, cls_logit = split_head_output(out)
    obj_t = tgt["obj_t"].to(out.device)
    box_t = tgt["box_t"].to(out.device)
    cls_t = tgt["cls_t"].to(out.device)
    pos = obj_t > 0.5                                                  # celdas con hueso
    n_pos = pos.sum().clamp(min=1).float()                             # clamp: sin cajas no se divide entre 0
    n_neg = (~pos).sum().clamp(min=1).float()

    # 1) objectness: todas las celdas; positivos y negativos se promedian por separado
    bce = F.binary_cross_entropy_with_logits(obj_logit, obj_t, reduction="none")
    obj_pos, obj_neg = bce[pos].sum() / n_pos, bce[~pos].sum() / n_neg
    obj = obj_pos + lambdas["noobj"] * obj_neg

    # 2) caja y 3) clase: solo celdas con hueso
    b, r, c = pos.nonzero(as_tuple=True)
    t_pred = torch.sigmoid(box_logit[b, r, c])                         # tx, ty, tw, th predichos
    t_gt = box_t[b, r, c]
    box_l1 = (t_pred - t_gt).abs().sum() / (4 * n_pos)
    iou = box_iou(decode_boxes(r, c, t_pred), decode_boxes(r, c, t_gt))
    box_iou_loss = (1 - iou).sum() / n_pos
    box = lambdas["l1"] * box_l1 + lambdas["iou"] * box_iou_loss
    if b.numel() == 0:
        cls = cls_logit.sum() * 0.0                                    # sin cajas: 0, pero conectado al grafo
    else:
        cls = F.cross_entropy(cls_logit[b, r, c], cls_t[b, r, c], reduction="sum") / n_pos

    total = lambdas["obj"] * obj + lambdas["box"] * box + lambdas["cls"] * cls
    mean_iou = iou.detach().mean() if iou.numel() > 0 else torch.zeros((), device=out.device)
    return {"total": total, "obj": obj, "obj_pos": obj_pos, "obj_neg": obj_neg, "box": box,
            "box_l1": box_l1, "box_iou": box_iou_loss, "cls": cls, "mean_iou": mean_iou}


@torch.no_grad()
def decode_predictions(out, score_thresh=0.0, img_size=IMG_SIZE):
    """Logits [B,8,S,S] -> lista de B dicts con una candidata por celda, ordenadas por score (mayor a menor):
    boxes [N,4] xyxy en px | scores [N] = objectness x prob. de clase | labels [N] en 1..3 | cells [N,2] = (fila, columna)."""
    out = out.float()
    obj_logit, box_logit, cls_logit = split_head_output(out)
    B, S = obj_logit.shape[0], obj_logit.shape[1]
    cls_score, cls_idx = torch.softmax(cls_logit, dim=-1).max(dim=-1)
    score = torch.sigmoid(obj_logit) * cls_score
    t = torch.sigmoid(box_logit)
    rows = torch.arange(S, device=out.device).view(S, 1).expand(S, S).reshape(-1)
    cols = torch.arange(S, device=out.device).view(1, S).expand(S, S).reshape(-1)
    results = []
    for b in range(B):
        boxes = decode_boxes(rows, cols, t[b].reshape(-1, 4), img_size, S).clamp(0, img_size)
        sc, lb = score[b].reshape(-1), cls_idx[b].reshape(-1) + 1      # clases 0..2 -> 1..3
        keep = (sc >= score_thresh).nonzero(as_tuple=True)[0]
        keep = keep[sc[keep].argsort(descending=True)]                 # el NMS parte de scores ordenados
        results.append({"boxes": boxes[keep], "scores": sc[keep], "labels": lb[keep],
                        "cells": torch.stack([rows[keep], cols[keep]], dim=1)})
    return results


def postprocess(out, score_thresh=0.05, nms_fn=None, iou_thresh=0.5):
    """decode_predictions + NMS opcional. nms_fn(boxes, scores, labels, iou_thresh) -> índices a conservar."""
    preds = decode_predictions(out, score_thresh)
    if nms_fn is None:
        return preds
    final = []
    for p in preds:
        keep = nms_fn(p["boxes"], p["scores"], p["labels"], iou_thresh)
        final.append({k: v[keep] for k, v in p.items()})
    return final
