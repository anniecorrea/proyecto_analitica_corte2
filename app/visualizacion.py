"""Imágenes y figuras del frontend: los tres visualizadores de la consigna (sección 3.4).

1. MIP del volumen crudo con umbral HU (sin modelo).
2. Corte a corte: cajas del detector + máscaras coloreadas + distancia de cada fragmento junto a su etiqueta.
3. Reconstrucción 3D: marching cubes (skimage) de cada fragmento + malla Plotly Mesh3d con la distancia en la etiqueta.
"""
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import to_hex, to_rgb
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage import measure

VENTANA_HU = (-200.0, 1200.0)          # ventana de hueso para mostrar (solo visualización)
SOMBRAS = (0.0, 0.35, 0.6, 0.2, 0.5)   # misma regla que _instance_colors de salida_3d.ipynb
ALPHA = 0.45                           # opacidad fija de las máscaras
NOMBRE_CORTO = {"sacro": "Sacro", "coxal_izquierdo": "Coxal izq.", "coxal_derecho": "Coxal der."}


def ventanear(volume_hu, ventana=VENTANA_HU):
    """Volumen HU -> uint8 en la ventana de hueso (ocupa 1/4 de memoria que el float32 original)."""
    lo, hi = ventana
    return (np.clip((volume_hu - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def tabla_de_colores(region_by_class, colores_clase, max_id):
    """LUT [max_id+1, 3] en 0..1: cada fragmento con el color de su región y una sombra distinta por ID."""
    lut = np.zeros((max_id + 1, 3))
    for fid, cls in region_by_class.items():
        base = np.array(to_rgb(colores_clase[int(cls)]))
        shade = SOMBRAS[int(fid) % len(SOMBRAS)]
        lut[int(fid)] = base * (1 - shade) + shade
    return lut


def superponer(gris_u8, etiquetas, lut, alpha=ALPHA):
    """Corte en gris + fragmentos coloreados -> RGB uint8."""
    rgb = np.repeat(gris_u8[..., None].astype(np.float32) / 255.0, 3, axis=2)
    mask = etiquetas > 0
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * lut[etiquetas[mask]]
    return (rgb * 255).astype(np.uint8)


def _con_proporcion(rgb, escala_vertical):
    """Corrige la proporción física (cortes más gruesos que el píxel) y deja la cabeza arriba."""
    h, w = rgb.shape[:2]
    img = Image.fromarray(rgb[::-1])
    return np.asarray(img.resize((w, max(1, int(round(h * escala_vertical)))), Image.NEAREST))


def etiqueta_fragmento(fila):
    """'Coxal izq. F3 · 4.2 mm' (o 'principal'): la distancia va pegada a la etiqueta, no solo en la tabla."""
    nombre = f"{NOMBRE_CORTO.get(fila['region'], fila['region'])} F{int(fila['fragment_id'])}"
    if fila["es_principal"]:
        return f"{nombre} · principal"
    return f"{nombre} · {fila['distancia_mm']:.1f} mm"


# ---------------------------------------------------------------------------------------------------------------
# Visualizador 1: volumen crudo
# ---------------------------------------------------------------------------------------------------------------
def imagen_mip_hueso(mip_hu, spacing_xyz, umbral_hu):
    """MIP coronal (z, x) en HU -> imagen en gris con la cabeza arriba y la proporción física correcta."""
    gris = (np.clip((mip_hu - umbral_hu) / (1500.0 - umbral_hu), 0, 1) * 255).astype(np.uint8)
    return _con_proporcion(np.repeat(gris[..., None], 3, axis=2), spacing_xyz[2] / spacing_xyz[0])


# ---------------------------------------------------------------------------------------------------------------
# Visualizador 2: inferencia corte por corte
# ---------------------------------------------------------------------------------------------------------------
def _fuente(tamano):
    try:
        return ImageFont.load_default(size=tamano)
    except TypeError:                  # Pillow < 10.1 no acepta size
        return ImageFont.load_default()


def corte_anotado(gris_u8, inst, lut, cajas, colores_clase, etiquetas, escala=2):
    """Corte con máscaras, cajas del detector (color de su región, con el score) y, sobre cada fragmento, su
    etiqueta con la distancia de separación. etiquetas: {fragment_id: texto}."""
    rgb = superponer(gris_u8, inst, lut)
    img = Image.fromarray(rgb).resize((rgb.shape[1] * escala, rgb.shape[0] * escala), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    fuente = _fuente(13 * escala // 2 + 4)

    if cajas is not None:
        for box, label, score in zip(cajas["boxes"], cajas["labels"], cajas["scores"]):
            x0, y0, x1, y1 = (float(v) * escala for v in box)
            color = colores_clase[int(label)]
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
            draw.text((x0 + 3, y0 + 2), f"{score:.2f}", fill=color, font=fuente)

    ids = [int(i) for i in np.unique(inst) if i]
    if ids:
        centros = ndimage.center_of_mass(np.ones_like(inst), inst, ids)
        for fid, (cy, cx) in zip(ids, centros):
            # El píxel del fragmento más cercano a su centro: si el fragmento tiene dos partes en el corte, la
            # etiqueta queda encima del hueso y no en el hueco entre ellas.
            yy, xx = np.nonzero(inst == fid)
            k = int(np.argmin((yy - cy) ** 2 + (xx - cx) ** 2))
            texto = etiquetas.get(fid, f"F{fid}")
            x, y = xx[k] * escala, yy[k] * escala
            x0, y0, x1, y1 = draw.textbbox((x, y), texto, font=fuente, anchor="mm")
            draw.rectangle([x0 - 3, y0 - 2, x1 + 3, y1 + 2], fill=(0, 0, 0))
            draw.text((x, y), texto, fill=to_hex(lut[fid]), font=fuente, anchor="mm")
    return img


# ---------------------------------------------------------------------------------------------------------------
# Visualizador 3: reconstrucción 3D final
# ---------------------------------------------------------------------------------------------------------------
def figura_3d(inst, spacing_xyz, fragmentos, lut, paso=3, voxeles_pequeno=20000):
    """Una malla por fragmento (marching cubes sobre la máscara apilada, en mm) + su etiqueta con la distancia.
    paso=3 deja la figura en ~3 MB (con 2, ~7 MB: lenta en el celular por el túnel); los fragmentos pequeños usan
    paso 1 para no perder su forma."""
    sx, sy, sz = (float(s) for s in spacing_xyz)
    trazas, lx, ly, lz, textos, colores = [], [], [], [], [], []
    objetos = ndimage.find_objects(inst)
    for _, fila in fragmentos.sort_values("fragment_id").iterrows():
        fid = int(fila["fragment_id"])
        if fid > len(objetos) or objetos[fid - 1] is None:
            continue
        sl = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in objetos[fid - 1])
        mask = np.pad(inst[sl] == fid, 1)            # borde de ceros: la malla queda cerrada
        if mask.sum() < 8:
            continue
        verts, faces, _, _ = measure.marching_cubes(mask.astype(np.uint8), 0.5, spacing=(sz, sy, sx),
                                                    step_size=1 if mask.sum() < voxeles_pequeno else paso)
        z0, y0, x0 = (s.start - 1 for s in sl)
        z, y, x = verts[:, 0] + z0 * sz, verts[:, 1] + y0 * sy, verts[:, 2] + x0 * sx
        color = to_hex(lut[fid])
        texto = etiqueta_fragmento(fila)
        trazas.append(go.Mesh3d(x=x, y=y, z=z, i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color=color,
                                opacity=1.0, name=texto, showlegend=True, hoverinfo="name",
                                lighting={"ambient": 0.45, "diffuse": 0.8, "roughness": 0.5}))
        cz, cy, cx = np.argwhere(mask).mean(axis=0)
        lx.append((cx + x0) * sx)
        ly.append((cy + y0) * sy)
        lz.append((cz + z0) * sz)
        textos.append(texto)
        colores.append(color)
    if textos:
        trazas.append(go.Scatter3d(x=lx, y=ly, z=lz, mode="text", text=textos, textfont={"color": "#111", "size": 12},
                                   hoverinfo="skip", showlegend=False))
    fig = go.Figure(trazas)
    fig.update_layout(
        scene={"xaxis_title": "x (mm)", "yaxis_title": "y (mm)", "zaxis_title": "z (mm)", "aspectmode": "data",
               "camera": {"eye": {"x": 0.0, "y": -1.9, "z": 0.4}}},
        legend={"orientation": "h", "y": -0.02}, margin={"l": 0, "r": 0, "t": 10, "b": 0}, height=620,
        dragmode="orbit")
    return fig
