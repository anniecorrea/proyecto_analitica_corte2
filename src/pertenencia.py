"""
pertenencia.py -- Cabeza de clasificación por pertenencia anatómica (PENGWIN, semana 10). Autor: Fabián.

QUÉ RESUELVE
  La consigna exige una TERCERA cabeza sobre el backbone compartido que asigne cada instancia (fragmento)
  a sacro, coxal izquierdo o coxal derecho. No basta con los logits de clase del detector ni con la
  clase semántica del decoder: esas dos se usan aquí solo como reglas de comparación (baselines).

UNIDAD DE CLASIFICACIÓN
  Una instancia (fragmento) en un corte 2D. En el volumen, la pertenencia de un fragmento 3D es el
  promedio de sus probabilidades por corte, ponderado por el área (lo hace pipeline_pengwin.py).

CÓMO FUNCIONA (por instancia)
  1. Mapa compartido después de CBAM: f4 [B, 256, 16, 16].
  2. La máscara de la instancia (256 x 256) se reduce a 16 x 16 con interpolación 'area': cada celda
     queda con la FRACCIÓN de la celda que ocupa el fragmento (un fragmento de 1 píxel no desaparece).
  3. Promedio del mapa ponderado por esa máscara -> vector de 128 (tras una 1x1 que reduce canales).
  4. Rasgos geométricos (6): centroide x, y; centroide x relativo al centro del hueso en ese corte;
     ancho, alto y raíz del área. La posición es la pista clave para izquierdo/derecho: los dos coxales
     son casi simétricos y la textura sola no los distingue.
  5. MLP -> 3 logits (0 = sacro, 1 = coxal izquierdo, 2 = coxal derecho; label = índice + 1).

ADVERTENCIA PARA EL ENTRENAMIENTO (Miguel)
  Un flip horizontal como aumento de datos intercambia izquierdo y derecho: si se usa, hay que
  intercambiar también las etiquetas 2 <-> 3 (IDs 11-20 <-> 21-30). Si no, la cabeza aprende ruido.

CONTRATO
  head = PertenenciaHead()
  masks, batch_idx, labels = targets_pertenencia(instance_masks)   # instance_masks: lista de [H,W] con IDs 1..30
  logits = head(f4, masks, batch_idx)                              # [K, 3]
  loss = perdida_pertenencia(logits, labels)                       # escalar; 0 conectado al grafo si K = 0
  En la pérdida conjunta: L = L_det + λ_seg·L_seg + λ_cls·L_cls (λ_cls se justifica con val).
  En entrenamiento las máscaras son las REALES (teacher forcing); en inferencia, las que predice el decoder.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASES = (1, 2, 3)
NOMBRES = {1: "sacro", 2: "coxal_izquierdo", 3: "coxal_derecho"}
N_GEOM = 6


def clase_de_id(instance_id):
    """ID del dataset PENGWIN (1..30) -> clase anatómica 1..3; 0 si es fondo o está fuera de rango."""
    i = int(instance_id)
    if 1 <= i <= 10:
        return 1
    if 11 <= i <= 20:
        return 2
    if 21 <= i <= 30:
        return 3
    return 0


# --------------------------------------------------------------------------------------------------
# Rasgos geométricos
# --------------------------------------------------------------------------------------------------
def rasgos_geometricos(masks, batch_idx):
    """masks [K,H,W] (bool/float), batch_idx [K] -> [K, 6] en float32, aprox. en [-1, 1].
    cx, cy: centroide normalizado | cx_rel: cx menos el centroide de TODO el hueso de esa imagen
    | w, h: tamaño de la caja de la instancia | raiz_area: sqrt(área) / lado."""
    K, H, W = masks.shape
    m = masks.float()
    if K == 0:
        return m.new_zeros((0, N_GEOM))
    ys = torch.arange(H, device=m.device, dtype=torch.float32).view(1, H, 1)
    xs = torch.arange(W, device=m.device, dtype=torch.float32).view(1, 1, W)
    area = m.sum(dim=(1, 2)).clamp(min=1.0)
    cx = (m * xs).sum(dim=(1, 2)) / area
    cy = (m * ys).sum(dim=(1, 2)) / area

    # centro del hueso por imagen: centroide de la unión de todas sus instancias
    B = int(batch_idx.max().item()) + 1
    union_area = torch.zeros(B, device=m.device).index_add_(0, batch_idx, m.sum(dim=(1, 2)))
    union_cx = torch.zeros(B, device=m.device).index_add_(0, batch_idx, (m * xs).sum(dim=(1, 2)))
    centro_x = union_cx / union_area.clamp(min=1.0)

    filas = m.amax(dim=2) > 0                                          # [K, H]: filas ocupadas
    cols = m.amax(dim=1) > 0                                           # [K, W]
    alto = filas.float().sum(dim=1)
    ancho = cols.float().sum(dim=1)
    geom = torch.stack([
        2 * cx / W - 1,
        2 * cy / H - 1,
        2 * (cx - centro_x[batch_idx]) / W,
        ancho / W,
        alto / H,
        torch.sqrt(m.sum(dim=(1, 2))) / max(H, W),
    ], dim=1)
    return geom


# --------------------------------------------------------------------------------------------------
# La cabeza
# --------------------------------------------------------------------------------------------------
class PertenenciaHead(nn.Module):
    """f4 [B, C, h, w] + máscaras de instancia [K, H, W] -> logits [K, 3]."""

    def __init__(self, in_channels=256, hidden=128, num_classes=3, dropout=0.1, usar_geometria=True):
        super().__init__()
        self.usar_geometria = usar_geometria
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        extra = N_GEOM if usar_geometria else 0
        self.mlp = nn.Sequential(
            nn.Linear(hidden + extra, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, fmap, masks, batch_idx):
        masks = torch.as_tensor(masks, device=fmap.device)
        batch_idx = torch.as_tensor(batch_idx, device=fmap.device).long().reshape(-1)
        if masks.ndim != 3 or masks.shape[0] != batch_idx.shape[0]:
            raise ValueError(f"masks debe ser [K,H,W] y batch_idx [K]; llegaron {tuple(masks.shape)} y {tuple(batch_idx.shape)}")
        K = masks.shape[0]
        if K == 0:                                                     # corte sin instancias: salida vacía con forma
            # 0 * (parámetros) mantiene la salida conectada al grafo: backward() no falla en un lote sin fragmentos
            return fmap.new_zeros((0, self.num_classes)) + 0.0 * (fmap.sum() + self.mlp[-1].bias.sum())
        if int(batch_idx.max()) >= fmap.shape[0] or int(batch_idx.min()) < 0:
            raise ValueError("batch_idx apunta a una imagen que no está en fmap")

        pesos = pesos_de_mascaras(masks, fmap.shape[-2:])
        geom = rasgos_geometricos(masks, batch_idx) if self.usar_geometria else None
        return self.forward_desde_pesos(fmap, pesos, geom, batch_idx)

    def forward_desde_pesos(self, fmap, pesos, geom, batch_idx):
        """Mismo cálculo que forward, pero con los pesos [K,h,w] y la geometría [K,6] ya calculados.
        Lo usa el entrenamiento con mapas precalculados (etapa D) para no repetir el backbone en cada época."""
        f = self.reduce(fmap)                                          # [B, hidden, h, w]
        vec = torch.einsum("khw,kchw->kc", pesos.to(f.dtype), f[batch_idx])                 # [K, hidden]
        if self.usar_geometria:
            vec = torch.cat([vec, geom.to(vec.dtype)], dim=1)
        return self.mlp(vec)


def pesos_de_mascaras(masks, size_hw):
    """Máscaras [K,H,W] -> pesos [K,h,w] que suman 1: fracción de cada celda del mapa que ocupa el fragmento."""
    pesos = F.interpolate(masks[:, None].float(), size=tuple(size_hw), mode="area")[:, 0]
    return pesos / pesos.sum(dim=(1, 2), keepdim=True).clamp(min=1e-6)


# --------------------------------------------------------------------------------------------------
# Targets y pérdida
# --------------------------------------------------------------------------------------------------
def instancias_de_mascara(instance_mask, min_pixels=1):
    """Máscara 2D de IDs -> (masks [K,H,W] bool, ids [K]) en orden de ID. Ignora el fondo (0)."""
    # int64 siempre: el decoder entrega uint16 y PyTorch soporta mal ese tipo (torch.unique falla con él)
    m = instance_mask.long().cpu() if torch.is_tensor(instance_mask) else torch.from_numpy(np.asarray(instance_mask).astype(np.int64))
    ids = [int(v) for v in torch.unique(m).tolist() if int(v) != 0]
    masks = [m == i for i in ids]
    keep = [k for k, mk in enumerate(masks) if int(mk.sum()) >= min_pixels]
    if not keep:
        return torch.zeros((0,) + tuple(m.shape), dtype=torch.bool), torch.zeros(0, dtype=torch.long)
    return torch.stack([masks[k] for k in keep]), torch.tensor([ids[k] for k in keep], dtype=torch.long)


def targets_pertenencia(instance_masks, min_pixels=1):
    """Lista de B máscaras 2D con IDs del dataset (1..30) -> masks [K,H,W], batch_idx [K], labels [K] en 0..2."""
    todas, b_idx, labels = [], [], []
    for b, mask in enumerate(instance_masks):
        masks, ids = instancias_de_mascara(mask, min_pixels)
        for mk, i in zip(masks, ids.tolist()):
            c = clase_de_id(i)
            if c == 0:
                raise ValueError(f"ID de instancia fuera de 1..30 en la imagen {b}: {i}")
            todas.append(mk)
            b_idx.append(b)
            labels.append(c - 1)
    if not todas:
        shape = tuple(np.shape(instance_masks[0])) if len(instance_masks) else (256, 256)
        return (torch.zeros((0,) + shape, dtype=torch.bool), torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long))
    return torch.stack(todas), torch.tensor(b_idx, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def pesos_por_clase(labels, num_classes=3):
    """Pesos inversos a la frecuencia (normalizados a media 1) para la entropía cruzada. Calcular con TRAIN."""
    labels = torch.as_tensor(labels).long().reshape(-1)
    conteo = torch.bincount(labels, minlength=num_classes).float().clamp(min=1.0)
    w = conteo.sum() / (num_classes * conteo)
    return w / w.mean()


def perdida_pertenencia(logits, labels, class_weights=None):
    """Entropía cruzada promedio por instancia. Con 0 instancias devuelve 0 conectado al grafo (no rompe AMP)."""
    logits = logits.float()
    labels = torch.as_tensor(labels, device=logits.device).long().reshape(-1)
    if logits.shape[0] == 0:
        return logits.sum() * 0.0
    w = None if class_weights is None else torch.as_tensor(class_weights, device=logits.device, dtype=torch.float32)
    return F.cross_entropy(logits, labels, weight=w)


# --------------------------------------------------------------------------------------------------
# Inferencia sobre instancias predichas
# --------------------------------------------------------------------------------------------------
@torch.no_grad()
def clasificar_instancias(head, fmap, instance_masks_pred):
    """instance_masks_pred: lista de B máscaras 2D [H,W] con IDs locales (salida del decoder).
    Devuelve lista de B dicts {id_local: probs np.ndarray[3]}. Imágenes sin instancias -> {}."""
    todas, b_idx, ids_por_fila = [], [], []
    for b, mask in enumerate(instance_masks_pred):
        masks, ids = instancias_de_mascara(mask)
        for mk, i in zip(masks, ids.tolist()):
            todas.append(mk)
            b_idx.append(b)
            ids_por_fila.append((b, i))
    salida = [dict() for _ in instance_masks_pred]
    if not todas:
        return salida
    masks = torch.stack(todas).to(fmap.device)
    probs = torch.softmax(head(fmap, masks, torch.tensor(b_idx, device=fmap.device)).float(), dim=1).cpu().numpy()
    for (b, i), p in zip(ids_por_fila, probs):
        salida[b][i] = p
    return salida


# --------------------------------------------------------------------------------------------------
# Reglas de comparación (baselines) y métricas
# --------------------------------------------------------------------------------------------------
def regla_mayoria_semantica(instance_mask, semantic_mask):
    """Baseline 1: clase = la más frecuente de la máscara semántica dentro de la instancia (lo que hace el decoder)."""
    inst, sem = np.asarray(instance_mask), np.asarray(semantic_mask)
    res = {}
    for i in [int(v) for v in np.unique(inst) if v != 0]:
        vals, cnt = np.unique(sem[inst == i], return_counts=True)
        ok = (vals >= 1) & (vals <= 3)
        res[i] = int(vals[ok][np.argmax(cnt[ok])]) if ok.any() else 0
    return res


def regla_caja_detectada(instance_mask, boxes, labels):
    """Baseline 2: clase = la de la caja detectada (después del NMS) que más píxeles de la instancia contiene.
    0 si la instancia no cae en ninguna caja."""
    inst = np.asarray(instance_mask)
    boxes = np.asarray(boxes.detach().cpu() if torch.is_tensor(boxes) else boxes, dtype=np.float64).reshape(-1, 4)
    labels = np.asarray(labels.detach().cpu() if torch.is_tensor(labels) else labels).reshape(-1).astype(int)
    res = {}
    for i in [int(v) for v in np.unique(inst) if v != 0]:
        ys, xs = np.nonzero(inst == i)
        mejor, mejor_n = 0, 0
        for box, lab in zip(boxes, labels):
            n = int(((xs >= box[0]) & (xs < box[2]) & (ys >= box[1]) & (ys < box[3])).sum())
            if n > mejor_n:
                mejor, mejor_n = int(lab), n
        res[i] = mejor
    return res


def metricas_pertenencia(y_true, probs=None, y_pred=None):
    """y_true en 1..3; probs [N,3] (para AUC) o y_pred en 1..3 (para reglas sin probabilidades).
    Devuelve F1 macro y por clase, AUC uno-contra-resto macro (si hay probs y las 3 clases), exactitud y matriz."""
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

    y_true = np.asarray(y_true).astype(int).reshape(-1)
    if probs is not None:
        probs = np.asarray(probs, dtype=np.float64).reshape(-1, 3)
        y_pred = probs.argmax(axis=1) + 1
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    res = {"n": int(len(y_true))}
    if len(y_true) == 0:
        return {**res, "f1_macro": np.nan, "auc_macro_ovr": np.nan, "exactitud": np.nan}
    res["exactitud"] = float(accuracy_score(y_true, y_pred))
    res["f1_macro"] = float(f1_score(y_true, y_pred, labels=list(CLASES), average="macro", zero_division=0))
    for c, f in zip(CLASES, f1_score(y_true, y_pred, labels=list(CLASES), average=None, zero_division=0)):
        res[f"f1_{NOMBRES[c]}"] = float(f)
    res["auc_macro_ovr"] = np.nan
    if probs is not None and len(np.unique(y_true)) == 3:
        res["auc_macro_ovr"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=list(CLASES)))
    res["matriz_confusion"] = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])  # fila/col 0 = sin clase
    return res


# --------------------------------------------------------------------------------------------------
# Etapa D: entrenar SOLO la cabeza sobre el tronco entrenado y congelado (backbone + CBAM de Miguel)
# --------------------------------------------------------------------------------------------------
# Por qué así: el tronco ya aprendió en las etapas A-C; congelarlo y entrenar solo esta cabeza es el mismo
# esquema por etapas que la etapa B (solo decoder). Como el tronco no cambia, f4 se calcula UNA vez por corte
# y cada época solo pasa por la cabeza: cabe en CPU.

def cargar_submodulo(modulo, state_dict, prefijo):
    """Carga en `modulo` las claves de state_dict que empiezan por `prefijo` (p. ej. 'backbone.'). Estricto:
    si sobra o falta una clave, falla en vez de cargar pesos a medias."""
    sub = {k[len(prefijo):]: v for k, v in state_dict.items() if k.startswith(prefijo)}
    if not sub:
        raise KeyError(f"El checkpoint no tiene claves con el prefijo {prefijo!r}")
    modulo.load_state_dict(sub, strict=True)
    return modulo


@torch.no_grad()
def precalcular_cache_pertenencia(trunk_fn, dataset, indices, batch_size=16, device="cpu", f4_dtype=torch.float16,
                                  progreso=True):
    """Pasa cada corte por el tronco congelado UNA vez y guarda lo que la cabeza necesita.
    trunk_fn(x [B,1,H,W]) -> f4 [B,C,h,w]. dataset[i] -> (imagen [1,H,W], target con instance_mask, case_id, slice_index).
    Devuelve dict: f4 [N,C,h,w] | pesos [K,h,w] | geom [K,6] | corte [K] (fila de f4) | label [K] (0..2)
    | meta: DataFrame con case_id, slice_index, gt_id por instancia."""
    import pandas as pd

    f4_lista, pesos_l, geom_l, corte_l, label_l, meta = [], [], [], [], [], []
    n_cortes = 0
    indices = list(indices)
    for inicio in range(0, len(indices), batch_size):
        lote = [dataset[i] for i in indices[inicio:inicio + batch_size]]
        x = torch.stack([torch.as_tensor(img) for img, _ in lote]).float().to(device)
        f4 = trunk_fn(x).float()
        mascaras = [t["instance_mask"] for _, t in lote]
        masks, b_idx, labels = targets_pertenencia(mascaras)
        if masks.shape[0]:
            _, ids = zip(*[instancias_de_mascara(m) for m in mascaras])
            gt_ids = torch.cat([i for i in ids if len(i)]).tolist()
            pesos_l.append(pesos_de_mascaras(masks, f4.shape[-2:]).cpu())
            geom_l.append(rasgos_geometricos(masks, b_idx).cpu())
            corte_l.append(b_idx + n_cortes)
            label_l.append(labels)
            for b, gid in zip(b_idx.tolist(), gt_ids):
                meta.append({"case_id": lote[b][1].get("case_id"), "slice_index": lote[b][1].get("slice_index"),
                             "gt_id": int(gid)})
        f4_lista.append(f4.to(f4_dtype).cpu())
        n_cortes += len(lote)
        if progreso and (inicio // batch_size) % 25 == 0:
            print(f"  {n_cortes}/{len(indices)} cortes procesados")
    h, w = f4_lista[0].shape[-2:] if f4_lista else (16, 16)
    vacio = lambda *s, dtype=torch.float32: torch.zeros(s, dtype=dtype)
    return {
        "f4": torch.cat(f4_lista) if f4_lista else vacio(0, 256, h, w, dtype=f4_dtype),
        "pesos": torch.cat(pesos_l) if pesos_l else vacio(0, h, w),
        "geom": torch.cat(geom_l) if geom_l else vacio(0, N_GEOM),
        "corte": torch.cat(corte_l) if corte_l else vacio(0, dtype=torch.long),
        "label": torch.cat(label_l) if label_l else vacio(0, dtype=torch.long),
        "meta": pd.DataFrame(meta, columns=["case_id", "slice_index", "gt_id"]),
    }


@torch.no_grad()
def predecir_cache(head, cache, batch_inst=1024, device="cpu"):
    """Probabilidades [K,3] de la cabeza para todas las instancias de un cache."""
    head.eval()
    salida = []
    for i in range(0, cache["label"].shape[0], batch_inst):
        sel = slice(i, i + batch_inst)
        f = cache["f4"][cache["corte"][sel]].float().to(device)
        idx = torch.arange(f.shape[0], device=device)
        logits = head.forward_desde_pesos(f, cache["pesos"][sel].to(device), cache["geom"][sel].to(device), idx)
        salida.append(torch.softmax(logits.float(), dim=1).cpu())
    return torch.cat(salida) if salida else torch.zeros(0, 3)


def entrenar_pertenencia(head, cache_train, cache_val, epochs=40, lr=1e-3, weight_decay=1e-4, batch_inst=256,
                         paciencia=8, seed=42, device="cpu", usar_pesos_clase=True):
    """Entrena la cabeza con los caches. Selección por F1 macro en VAL (desempate: menor pérdida de val).
    Detiene si val no mejora en `paciencia` épocas. Devuelve (head con los MEJORES pesos, historial DataFrame, mejor)."""
    import copy

    import pandas as pd

    if cache_train["label"].numel() == 0:
        raise ValueError("El cache de entrenamiento no tiene instancias")
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    head = head.to(device)
    pesos_clase = pesos_por_clase(cache_train["label"]).to(device) if usar_pesos_clase else None
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    K = cache_train["label"].shape[0]
    y_val = cache_val["label"].numpy() + 1
    historial, mejor, mejor_estado, sin_mejora = [], None, None, 0

    for epoch in range(1, epochs + 1):
        head.train()
        orden = torch.randperm(K, generator=gen)
        suma, n = 0.0, 0
        for i in range(0, K, batch_inst):
            sel = orden[i:i + batch_inst]
            f = cache_train["f4"][cache_train["corte"][sel]].float().to(device)
            idx = torch.arange(f.shape[0], device=device)
            logits = head.forward_desde_pesos(f, cache_train["pesos"][sel].to(device),
                                              cache_train["geom"][sel].to(device), idx)
            loss = perdida_pertenencia(logits, cache_train["label"][sel].to(device), pesos_clase)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            opt.step()
            suma += loss.item() * len(sel)
            n += len(sel)
        sched.step()

        probs_val = predecir_cache(head, cache_val, device=device)
        loss_val = float(F.nll_loss(torch.log(probs_val.clamp(min=1e-8)), cache_val["label"]).item()) \
            if len(y_val) else float("nan")
        m = metricas_pertenencia(y_val, probs=probs_val.numpy())
        fila = {"epoch": epoch, "lr": sched.get_last_lr()[0], "train_loss": suma / max(n, 1), "val_loss": loss_val,
                "val_f1_macro": m["f1_macro"], "val_auc_macro_ovr": m["auc_macro_ovr"], "val_exactitud": m["exactitud"],
                **{f"val_{k}": v for k, v in m.items() if k.startswith("f1_") and k != "f1_macro"}}
        es_mejor = mejor is None or (fila["val_f1_macro"], -fila["val_loss"]) > (mejor["val_f1_macro"], -mejor["val_loss"])
        fila["es_mejor"] = bool(es_mejor)
        historial.append(fila)
        if es_mejor:
            mejor, mejor_estado, sin_mejora = dict(fila), copy.deepcopy(head.state_dict()), 0
        else:
            sin_mejora += 1
        print(f"época {epoch:2d} | train {fila['train_loss']:.4f} | val {loss_val:.4f} | "
              f"F1 {fila['val_f1_macro']:.3f} | AUC {fila['val_auc_macro_ovr']:.3f}{'  *' if es_mejor else ''}")
        if sin_mejora >= paciencia:
            print(f"Sin mejora en val durante {paciencia} épocas: se detiene.")
            break

    head.load_state_dict(mejor_estado)
    head.eval()
    return head, pd.DataFrame(historial), mejor
