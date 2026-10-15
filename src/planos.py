"""Cortes por plano (axial, coronal, sagital) sobre volumenes en orientacion canonica (ver orientacion.py).

Convenciones (volumen canonico LPS, array (z, y, x), espaciado (x, y, z)):
    axial   vol[z]       -> (y, x)   fila = y (posterior crece), columna = x (izquierda crece)
    coronal vol[:, y, :] -> (z, x)   fila = z (superior crece),  columna = x
    sagital vol[:, :, x] -> (z, y)   fila = z,                   columna = y
Los cortes se guardan SIN voltear (la fila 0 es lo mas inferior en coronal/sagital); al dibujar usar origin="lower".

Flujo de un corte:  cortar -> pixeles cuadrados (lado = sx; coronal/sagital) -> realce -> letterbox 256 -> targets.
Este modulo no redefine las funciones del equipo: `enhance`, `letterbox`, `build_target` (build_detection_target) y
`restore_letterbox` (restore_letterbox_label_mask) se reciben como argumentos.

Entrada 2.5D: con n_channels=3 cada muestra apila los cortes [i-1, i, i+1] a lo largo del eje del plano (mismo preprocesamiento
y misma regla de bordes que src/entrada_25d.py: en los extremos se repite el corte del borde).
Cortes negativos: los que no tienen NI UN voxel de pelvis (mismo criterio que src/cortes_vacios.py, pero a lo largo del eje de cada plano).

Posicion lateral (rama sagital, opcion B): canal extra constante con (x_corte - x_linea_media) / 100 mm, recortado a +-1.5.
En el volumen canonico x crece hacia la izquierda del paciente: valor > 0 = lado izquierdo.

Probar:  python planos.py   (autoprueba con datos sinteticos; solo NumPy y Pillow)
"""
from __future__ import annotations

import numpy as np
from PIL import Image

PLANES = ("axial", "coronal", "sagital")
POSITION_SCALE_MM = 100.0
POSITION_CLIP = 1.5
_REGION_NAMES = {1: "sacro", 2: "coxal_izquierdo", 3: "coxal_derecho"}


# ------------------------------------------------------------------------------------------ geometria
def _check_plane(plane):
    if plane not in PLANES:
        raise ValueError(f"plane debe ser uno de {PLANES}, no {plane!r}")


def n_slices(shape_zyx, plane):
    _check_plane(plane)
    return {"axial": shape_zyx[0], "coronal": shape_zyx[1], "sagital": shape_zyx[2]}[plane]


def get_slice(vol, plane, idx):
    _check_plane(plane)
    return vol[idx] if plane == "axial" else vol[:, idx, :] if plane == "coronal" else vol[:, :, idx]


def put_slice(vol, plane, idx, arr):
    _check_plane(plane)
    if plane == "axial":
        vol[idx] = arr
    elif plane == "coronal":
        vol[:, idx, :] = arr
    else:
        vol[:, :, idx] = arr


def plane_spacing(plane, spacing_xyz):
    """(mm por fila, mm por columna) de un corte del plano dado."""
    _check_plane(plane)
    sx, sy, sz = spacing_xyz
    return {"axial": (sy, sx), "coronal": (sz, sx), "sagital": (sz, sy)}[plane]


def slice_axis_spacing(plane, spacing_xyz):
    """mm entre cortes consecutivos del plano (espaciado del eje perpendicular)."""
    _check_plane(plane)
    sx, sy, sz = spacing_xyz
    return {"axial": sz, "coronal": sy, "sagital": sx}[plane]


def square_shape(shape, row_mm, col_mm, target_mm):
    return (max(1, int(round(shape[0] * row_mm / target_mm))), max(1, int(round(shape[1] * col_mm / target_mm))))


def resize_label_nn(arr, rows, cols):
    img = Image.fromarray(np.ascontiguousarray(arr, dtype=np.int32))
    return np.asarray(img.resize((int(cols), int(rows)), Image.Resampling.NEAREST), dtype=np.int32)


def resize_image_bilinear(arr, rows, cols):
    img = Image.fromarray(np.ascontiguousarray(arr, dtype=np.float32))
    return np.asarray(img.resize((int(cols), int(rows)), Image.Resampling.BILINEAR), dtype=np.float32)


def slice_to_square(arr, plane, spacing_xyz, is_label):
    """Lleva un corte a pixeles cuadrados de lado sx (el espaciado en el plano axial). Si ya es cuadrado no hace nada."""
    row_mm, col_mm = plane_spacing(plane, spacing_xyz)
    shape = square_shape(arr.shape, row_mm, col_mm, spacing_xyz[0])
    if shape == tuple(arr.shape):
        return np.asarray(arr)
    return resize_label_nn(arr, *shape) if is_label else resize_image_bilinear(arr, *shape)


# ------------------------------------------------------------------------------------------ indice de cortes
def region_of_ids(values):
    """ids de fragmento -> region (1 sacro: 1-10, 2 coxal izq.: 11-20, 3 coxal der.: 21-30, 0 fondo)."""
    v = np.asarray(values)
    out = np.zeros(v.shape, dtype=np.uint8)
    out[(v >= 1) & (v <= 10)] = 1
    out[(v >= 11) & (v <= 20)] = 2
    out[(v >= 21) & (v <= 30)] = 3
    return out


def positive_slices(label_zyx, plane, spacing_xyz, min_pixels=20, region_names=None):
    """Cortes con al menos una region de >= min_pixels pixeles (medidos en pixeles cuadrados, antes del letterbox:
    mismo criterio que el manifiesto axial). Devuelve una lista de dicts: slice_index, n_regions, regions, n_instances."""
    _check_plane(plane)
    names = region_names or _REGION_NAMES
    axes = {"axial": (1, 2), "coronal": (0, 2), "sagital": (0, 1)}[plane]
    has_label = np.flatnonzero((label_zyx > 0).any(axis=axes))
    rows = []
    for idx in has_label:
        lab = slice_to_square(get_slice(label_zyx, plane, int(idx)), plane, spacing_xyz, True)
        regions = region_of_ids(lab)
        present = [r for r in (1, 2, 3) if int((regions == r).sum()) >= min_pixels]
        if not present:
            continue
        ids = np.unique(lab[lab > 0])
        rows.append({"slice_index": int(idx), "n_regions": len(present),
                     "regions": "|".join(names[r] for r in present), "n_instances": int(len(ids))})
    return rows


def negative_slices(label_zyx, plane):
    """Cortes del plano SIN un solo voxel de pelvis (cualquier id > 0). Los cortes con muy poca pelvis (1 a 19 pixeles) no son
    negativos ni positivos: quedan fuera de ambos a proposito. Devuelve dicts con slice_index, distancia_a_pelvis (cortes hasta el corte
    con pelvis mas cercano) y posicion ('antes' / 'despues' del tramo con pelvis segun el indice del plano, o 'hueco').
    Un volumen sin pelvis no tiene referencia y devuelve []."""
    _check_plane(plane)
    axes = {"axial": (1, 2), "coronal": (0, 2), "sagital": (0, 1)}[plane]
    counts = np.count_nonzero(np.asarray(label_zyx) > 0, axis=axes)
    has = counts > 0
    if not has.any():
        return []
    idx = np.arange(counts.size)
    con = np.flatnonzero(has)
    pos = np.searchsorted(con, idx)
    left, right = con[np.clip(pos - 1, 0, len(con) - 1)], con[np.clip(pos, 0, len(con) - 1)]
    dist = np.minimum(np.abs(idx - left), np.abs(idx - right))
    first, last = int(con.min()), int(con.max())
    rows = []
    for i in idx[~has]:
        where = "antes" if i < first else "despues" if i > last else "hueco"
        rows.append({"slice_index": int(i), "distancia_a_pelvis": int(dist[i]), "posicion": where})
    return rows


def sample_negatives(neg, n_target, seed=42, near_share=0.6):
    """Elige ~n_target negativos de un DataFrame (case_id, slice_index, distancia_a_pelvis, ...), repartidos por caso en proporcion a
    sus negativos (al menos 1 por caso). Dentro de cada caso, `near_share` sale de la mitad MAS CERCANA a la pelvis (los cortes que mas se
    parecen a uno con pelvis: femur, columna) y el resto de la mitad lejana; sin reemplazo y reproducible. n_target >= len(neg) devuelve todos."""
    import pandas as pd
    neg = neg.sort_values(["case_id", "slice_index"]).reset_index(drop=True)
    if n_target >= len(neg) or len(neg) == 0:
        return neg
    if not 0.0 <= near_share <= 1.0:
        raise ValueError("near_share debe estar en [0, 1]")
    rng = np.random.default_rng(seed)
    frac = float(n_target) / len(neg)
    parts = []
    for _, g in neg.groupby("case_id", sort=True):
        k = min(len(g), max(1, int(round(len(g) * frac))))
        med = g["distancia_a_pelvis"].median()
        near, far = g[g["distancia_a_pelvis"] <= med], g[g["distancia_a_pelvis"] > med]
        k_far = min(len(far), k - min(len(near), int(round(k * near_share))))
        k_near = min(len(near), k - k_far)
        take = []
        for part, kk in ((near, k_near), (far, k_far)):
            if kk > 0:
                take.append(part.iloc[np.sort(rng.choice(len(part), size=kk, replace=False))])
        parts.append(pd.concat(take))
    return pd.concat(parts).sort_values(["case_id", "slice_index"]).reset_index(drop=True)


def neighbor_indices(idx, n_slices_total, n_channels):
    """Cortes que forman los canales de `idx`: [idx] con 1 canal, [idx-1, idx, idx+1] con 3 (recortados al volumen).
    Misma regla que src.entrada_25d.indices_vecinos (la autoprueba de los cuadernos lo comprueba)."""
    if n_channels == 1:
        return [int(idx)]
    if n_channels != 3:
        raise ValueError(f"n_channels debe ser 1 o 3, llego {n_channels}")
    return [max(int(idx) - 1, 0), int(idx), min(int(idx) + 1, n_slices_total - 1)]


# ------------------------------------------------------------------------------------------ un corte listo para la red
def _prepare_slice(volume, plane, idx, spacing_xyz, window, enhance, letterbox, size):
    """Un corte vecino (solo imagen): ventaneo -> pixeles cuadrados -> realce -> letterbox. La etiqueta no influye en la imagen."""
    raw = np.asarray(get_slice(volume, plane, idx), dtype=np.float32)
    if window is not None:
        raw = window(raw)
    img = slice_to_square(raw, plane, spacing_xyz, False)
    if enhance is not None:
        img = enhance(img)
    return np.asarray(letterbox(img, np.zeros(img.shape, dtype=np.int16), size)[0], dtype=np.float32)


def make_sample(volume, label_zyx, plane, idx, spacing_xyz, *, window=None, enhance=None, letterbox, build_target,
                case_id="", size=256, min_pixels=20, n_channels=1, slice_cache=None):
    """Corte -> (imagen, etiqueta de instancias 256x256, target de deteccion, transformacion).
    imagen: 256x256 con n_channels=1; (3, 256, 256) con n_channels=3 = cortes [idx-1, idx, idx+1] del plano (2.5D).
    volume: HU crudos en su tipo original (p. ej. int16) si se pasa `window` (apply_hu_window, que se aplica solo al corte);
    o un volumen ya ventaneado a [0, 1] si window es None. enhance (enhance_bone_contrast) puede ser None.
    Orden: ventaneo -> pixeles cuadrados -> realce -> letterbox (igual que el axial del equipo cuando sx == sy).
    slice_cache: dict opcional {indice: imagen 256x256} para no repetir el preprocesamiento de los vecinos al recorrer cortes
    consecutivos de un mismo volumen (se limpia solo; usar un dict nuevo por volumen)."""
    raw = np.asarray(get_slice(volume, plane, idx), dtype=np.float32)
    if window is not None:
        raw = window(raw)
    img = slice_to_square(raw, plane, spacing_xyz, False)
    lab = slice_to_square(get_slice(label_zyx, plane, idx), plane, spacing_xyz, True)
    if enhance is not None:
        img = enhance(img)
    img256, lab256, tf = letterbox(img, lab, size)
    img256 = np.asarray(img256, dtype=np.float32)
    if n_channels != 1:
        stack = []
        for k in neighbor_indices(idx, n_slices(volume.shape, plane), n_channels):
            if k == int(idx):
                stack.append(img256)
                if slice_cache is not None:
                    slice_cache[k] = img256
            elif slice_cache is not None and k in slice_cache:
                stack.append(slice_cache[k])
            else:
                stack.append(_prepare_slice(volume, plane, k, spacing_xyz, window, enhance, letterbox, size))
                if slice_cache is not None:
                    slice_cache[k] = stack[-1]
        if slice_cache is not None:
            for k in [k for k in slice_cache if k < int(idx) - 1]:
                del slice_cache[k]
        img256 = np.stack(stack).astype(np.float32)
    scaled_min = max(1, int(round(min_pixels * tf["scale"] ** 2)))
    target = build_target(lab256, case_id, int(idx), scaled_min)
    return img256, lab256, target, {**tf, "square_shape": tuple(int(v) for v in img.shape)}


def restore_slice_labels(mask256, plane, original_shape, spacing_xyz, transform, restore_letterbox, size=256):
    """Inversa de make_sample para una mascara: 256x256 -> forma original del corte (original_shape)."""
    back = restore_letterbox(mask256, transform["square_shape"], transform, size)
    if tuple(back.shape) != tuple(original_shape):
        back = resize_label_nn(back, *original_shape)
    return back


def reassemble_volume(masks_by_slice, shape_zyx, plane, dtype=np.int16):
    """masks_by_slice: {indice_de_corte: mascara en la forma original del corte}. Los cortes ausentes quedan en 0."""
    vol = np.zeros(shape_zyx, dtype=dtype)
    for idx, mask in masks_by_slice.items():
        put_slice(vol, plane, int(idx), mask)
    return vol


# ------------------------------------------------------------------------------------------ posicion lateral (sagital)
def midline_mm(label_zyx, spacing_xyz):
    """Linea media de la pelvis: x (mm) del centro del sacro (ids 1-10) en el volumen canonico. NaN si no hay sacro."""
    zyx = np.argwhere((label_zyx >= 1) & (label_zyx <= 10))
    return float(zyx[:, 2].mean() * spacing_xyz[0]) if len(zyx) else float("nan")


def fov_center_mm(shape_zyx, spacing_xyz):
    """Centro del campo de vision en x (mm), en el mismo sistema que slice_position_mm."""
    return float((shape_zyx[2] - 1) / 2 * spacing_xyz[0])


def slice_position_mm(plane, idx, spacing_xyz):
    """Coordenada (mm) del corte a lo largo de su eje perpendicular, en el sistema indice * espaciado."""
    _check_plane(plane)
    k = {"axial": 2, "coronal": 1, "sagital": 0}[plane]
    return float(idx * spacing_xyz[k])


def position_value(x_mm, mid_mm, scale_mm=POSITION_SCALE_MM, clip=POSITION_CLIP):
    return float(np.clip((x_mm - mid_mm) / scale_mm, -clip, clip))


def position_channel(x_mm, mid_mm, size=256):
    """Canal constante (size x size) con la posicion lateral normalizada del corte."""
    return np.full((size, size), position_value(x_mm, mid_mm), dtype=np.float32)


def training_midline(mid_true_mm, fov_mm, rng, sigma_mm=15.0, p_fov=0.3, fov_jitter_mm=5.0):
    """Linea media que se le ensena a la red en un caso de entrenamiento: con prob. p_fov, el centro del FOV
    (lo unico disponible en la primera pasada de inferencia); si no, la real + ruido N(0, sigma). Se sortea UNA vez por
    caso y epoca (mismo valor para todos sus cortes)."""
    if not np.isfinite(mid_true_mm):
        return fov_mm
    if rng.random() < p_fov:
        return float(fov_mm + rng.normal(0.0, fov_jitter_mm))
    return float(mid_true_mm + rng.normal(0.0, sigma_mm))


# ------------------------------------------------------------------------------------------ autoprueba
def _selftest():
    import json
    import warnings
    from pathlib import Path

    warnings.simplefilter("ignore")
    here = Path(__file__).resolve().parent
    rng = np.random.default_rng(0)

    # volumen canonico sintetico, voxeles anisotropos (sz > sx): sacro (1) al centro, coxales (11 y 21) a los lados
    shape = (60, 70, 90)                                       # (z, y, x)
    sp = (0.8, 0.8, 1.6)
    L = np.zeros(shape, np.int16)
    L[10:50, 40:58, 38:52] = 1
    L[15:55, 15:45, 62:80] = 11
    L[15:55, 15:45, 10:28] = 21
    L[30:36, 20:26, 63:67] = 12                                # fragmento pequeño
    vol = rng.random(shape).astype(np.float32) * 0.2 + (L > 0) * 0.5

    # funciones reales del equipo, si estan los cuadernos al lado; si no, versiones minimas equivalentes
    def _team_defs():
        ns = {"np": np, "Image": Image}
        wanted = {"instance_to_semantic", "boxes_from_semantic", "letterbox_pair", "build_detection_target",
                  "restore_letterbox_label_mask"}
        import ast
        for nb_name in ("Targets_deteccion.ipynb", "Segmentacion_instancias.ipynb"):
            p = here / "notebooks" / nb_name
            if not p.exists():
                p = here / nb_name
            if not p.exists():
                return None
            for cell in json.loads(p.read_text(encoding="utf-8"))["cells"]:
                if cell["cell_type"] != "code":
                    continue
                src = "".join(cell["source"])
                try:
                    tree = ast.parse(src)
                except SyntaxError:
                    continue
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name in wanted and node.name not in ns:
                        if node.name == "build_detection_target":
                            ns["CLASS_ID_TO_NAME"] = _REGION_NAMES
                        exec(ast.get_source_segment(src, node), ns)
        return ns if wanted <= set(ns) else None

    ns = _team_defs()
    if ns is None:
        print("(sin los cuadernos del equipo: se usan stand-ins minimos para la prueba)")

        def letterbox(img, lab, size=256):
            h, w = img.shape
            s = min(size / w, size / h)
            nw, nh = max(1, round(w * s)), max(1, round(h * s))
            ox, oy = (size - nw) // 2, (size - nh) // 2
            io, lo = np.zeros((size, size), np.float32), np.zeros((size, size), np.int16)
            io[oy:oy + nh, ox:ox + nw] = resize_image_bilinear(img, nh, nw)
            lo[oy:oy + nh, ox:ox + nw] = resize_label_nn(lab, nh, nw)
            return io, lo, {"scale": s, "offset_x": ox, "offset_y": oy}

        def restore(mask, shape_yx, tf, size=256):
            h, w = shape_yx
            nw, nh = max(1, round(w * tf["scale"])), max(1, round(h * tf["scale"]))
            return resize_label_nn(mask[tf["offset_y"]:tf["offset_y"] + nh, tf["offset_x"]:tf["offset_x"] + nw], h, w)

        def build_target(lab, case_id, idx, min_pixels):
            return {"instance_mask": lab, "case_id": case_id, "slice_index": idx, "min_pixels": min_pixels}
    else:
        letterbox, restore, build_target = ns["letterbox_pair"], ns["restore_letterbox_label_mask"], ns["build_detection_target"]

    n_ok = 0
    for plane in PLANES:
        rows = positive_slices(L, plane, sp)
        assert rows, plane
        assert all(1 <= r["n_regions"] <= 3 for r in rows)
        # el sagital debe tener cortes con una sola region lateral; el axial con las tres
        if plane == "axial":
            assert max(r["n_regions"] for r in rows) == 3
        else:
            assert min(r["n_regions"] for r in rows) == 1 or plane == "coronal"
        mids = {}
        for r in rows[:: max(1, len(rows) // 12)]:
            idx = r["slice_index"]
            img, lab256, target, tf = make_sample(vol, L, plane, idx, sp, enhance=None, letterbox=letterbox,
                                                  build_target=build_target, case_id="t", size=256)
            assert img.shape == (256, 256) and lab256.shape == (256, 256)
            orig = get_slice(L, plane, idx)
            back = restore_slice_labels(lab256, plane, orig.shape, sp, tf, restore, 256)
            assert back.shape == orig.shape, (plane, back.shape, orig.shape)
            fg_o, fg_b = orig > 0, back > 0
            dice = 2 * (fg_o & fg_b).sum() / max(1, fg_o.sum() + fg_b.sum())
            assert dice > 0.9, (plane, idx, dice)
            mids[idx] = back
        vol_back = reassemble_volume(mids, L.shape, plane)
        for idx in mids:
            assert np.array_equal(get_slice(vol_back, plane, idx), mids[idx])
        n_ok += 1

    # posicion lateral
    mid = midline_mm(L, sp)
    assert abs(mid - 44.5 * sp[0]) < 1.0, mid                    # sacro en x = 38..51 -> centro 44.5
    assert abs(fov_center_mm(shape, sp) - 44.5 * sp[0]) < 1e-9
    left, right = slice_position_mm("sagital", 70, sp), slice_position_mm("sagital", 20, sp)
    assert position_value(left, mid) > 0 > position_value(right, mid)       # izquierda > 0, derecha < 0
    assert position_value(1e6, mid) == POSITION_CLIP and position_value(-1e6, mid) == -POSITION_CLIP
    ch = position_channel(left, mid)
    assert ch.shape == (256, 256) and ch.dtype == np.float32 and np.all(ch == ch[0, 0])
    r = np.random.default_rng(1)
    est = [training_midline(mid, mid + 4.0, r) for _ in range(2000)]
    assert 0.6 * 15.0 < np.std(est) < 2.0 * 15.0 and abs(np.mean(est) - mid) < 2.0
    assert training_midline(float("nan"), 33.0, r) == 33.0
    # --- cortes negativos: positivos, ambiguos y negativos son disjuntos y cubren el plano ---
    for plane in PLANES:
        pos_idx = {r["slice_index"] for r in positive_slices(L, plane, sp)}
        neg_rows = negative_slices(L, plane)
        neg_idx = {r["slice_index"] for r in neg_rows}
        total = n_slices(L.shape, plane)
        axes = {"axial": (1, 2), "coronal": (0, 2), "sagital": (0, 1)}[plane]
        con_pelvis = set(np.flatnonzero((L > 0).any(axis=axes)).tolist())
        assert not (pos_idx & neg_idx) and neg_idx == set(range(total)) - con_pelvis, plane
        assert pos_idx <= con_pelvis
        assert all(r["distancia_a_pelvis"] >= 1 for r in neg_rows)
        if neg_rows:
            lo, hi = min(con_pelvis), max(con_pelvis)
            assert all((r["posicion"] == "antes") == (r["slice_index"] < lo) for r in neg_rows)
            assert all((r["posicion"] == "despues") == (r["slice_index"] > hi) for r in neg_rows)
    assert negative_slices(np.zeros((5, 6, 7), np.int16), "axial") == []

    # --- vecinos 2.5D: misma regla de bordes que src/entrada_25d.py ---
    assert neighbor_indices(0, 10, 3) == [0, 0, 1] and neighbor_indices(9, 10, 3) == [8, 9, 9] and neighbor_indices(4, 10, 3) == [3, 4, 5]
    assert neighbor_indices(4, 10, 1) == [4]

    # --- make_sample 2.5D: canal central = muestra de 1 canal; vecinos = muestras de 1 canal de los cortes vecinos; cache no cambia nada ---
    for plane in PLANES:
        rows = positive_slices(L, plane, sp)
        for idx in (rows[0]["slice_index"], rows[len(rows) // 2]["slice_index"], rows[-1]["slice_index"]):
            kw = dict(enhance=None, letterbox=letterbox, build_target=build_target, case_id="t", size=256)
            one = make_sample(vol, L, plane, idx, sp, **kw)[0]
            three = make_sample(vol, L, plane, idx, sp, n_channels=3, **kw)[0]
            assert three.shape == (3, 256, 256) and three.dtype == np.float32
            assert np.array_equal(three[1], one)
            for slot, k in zip((0, 2), (max(idx - 1, 0), min(idx + 1, n_slices(L.shape, plane) - 1))):
                assert np.array_equal(three[slot], make_sample(vol, L, plane, k, sp, **kw)[0])
        cache = {}
        for r in rows[:6]:
            a = make_sample(vol, L, plane, r["slice_index"], sp, n_channels=3, slice_cache=cache, **kw)[0]
            b = make_sample(vol, L, plane, r["slice_index"], sp, n_channels=3, **kw)[0]
            assert np.array_equal(a, b) and len(cache) <= 3

    # --- muestreo de negativos ---
    import pandas as pd
    neg = pd.DataFrame([{"case_id": f"{c:03d}", "slice_index": i, "distancia_a_pelvis": 1 + i} for c in range(5) for i in range(40)])
    sel = sample_negatives(neg, 50, seed=1)
    assert 45 <= len(sel) <= 55 and sel.groupby("case_id").size().between(8, 12).all()
    assert (sel["distancia_a_pelvis"] <= 20).mean() > 0.55                              # sesgo hacia los cercanos (60 % pedido)
    assert sel.equals(sample_negatives(neg, 50, seed=1)) and not sel.equals(sample_negatives(neg, 50, seed=2))
    assert len(sample_negatives(neg, 10_000)) == len(neg)
    assert not sel.duplicated(["case_id", "slice_index"]).any()
    print(f"OK: {n_ok} planos (cortes, indice de positivos, ida y vuelta de etiquetas, reensamblado) y canal de posicion.")


if __name__ == "__main__":
    _selftest()
