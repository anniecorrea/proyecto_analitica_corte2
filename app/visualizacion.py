"""Imágenes para el frontend: cortes de la tomografía con los fragmentos predichos superpuestos."""
import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image

VENTANA_HU = (-200.0, 1200.0)          # ventana de hueso para mostrar (solo visualización)
SOMBRAS = (0.0, 0.35, 0.6, 0.2, 0.5)   # misma regla que _instance_colors de salida_3d.ipynb


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


def superponer(gris_u8, etiquetas, lut, alpha=0.5):
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


def vista_axial(gris_zyx, inst_zyx, lut, z, alpha):
    return superponer(gris_zyx[z], inst_zyx[z], lut, alpha)


def vista_coronal(gris_zyx, inst_zyx, lut, y, spacing_xyz, alpha):
    return _con_proporcion(superponer(gris_zyx[:, y, :], inst_zyx[:, y, :], lut, alpha), spacing_xyz[2] / spacing_xyz[1])


def vista_mip(gris_zyx, inst_zyx, lut, spacing_xyz, alpha):
    """Proyección coronal de máxima intensidad: todo el volumen de un vistazo (fusiones, divisiones, trozos sueltos)."""
    return _con_proporcion(superponer(gris_zyx.max(axis=1), inst_zyx.max(axis=1), lut, alpha),
                           spacing_xyz[2] / spacing_xyz[0])
