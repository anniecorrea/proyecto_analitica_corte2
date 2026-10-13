"""Entrada 2.5D: cada corte z entra al modelo junto con sus vecinos z-1 y z+1 como canales.

Un corte aislado puede ser ambiguo: una mancha pequeña puede ser un fragmento real o un artefacto. Con los cortes
vecinos el modelo ve si la estructura continúa. Con IN_CHANNELS = 3 (Entrenamiento_multitarea.ipynb) los canales son
[z-1, z, z+1], cada uno con el mismo preprocesamiento del dataset (ventana HU -> Otsu + CLAHE -> letterbox). En los
bordes del volumen se repite el corte del borde. Con in_channels = 1 todo queda como antes.

Entrenamiento (PengwinDetectionDataset con slices_25d=True) e inferencia sobre volúmenes (apilar_cortes) usan la
misma regla de vecinos (indices_vecinos), para que el modelo vea en inferencia exactamente lo que vio al entrenar.
"""
import numpy as np


def in_channels_del_modelo(payload):
    """Canales de entrada de un checkpoint (exportado o de entrenamiento). Los modelos anteriores al 2.5D dan 1."""
    arquitectura = payload.get("architecture") or {}
    config = payload.get("training_config", payload.get("config")) or {}
    return int(arquitectura.get("in_channels", config.get("IN_CHANNELS", 1)))


def indices_vecinos(z, n_cortes, in_channels):
    """Cortes que forman los canales del corte z: [z] con 1 canal, [z-1, z, z+1] con 3 (recortados al volumen)."""
    if in_channels == 1:
        return [int(z)]
    if in_channels != 3:
        raise ValueError(f"in_channels debe ser 1 o 3, llegó {in_channels}")
    return [max(int(z) - 1, 0), int(z), min(int(z) + 1, n_cortes - 1)]


def apilar_cortes(volume_zyx, z_indices, preprocess_fn, in_channels):
    """Entrada del modelo [B, in_channels, S, S] (float32) para los cortes `z_indices` de un volumen (z, y, x) en HU.

    preprocess_fn(corte_hu) -> imagen 2D ya preprocesada (S x S). Cada corte se preprocesa una sola vez aunque
    sea vecino de varios cortes del lote."""
    n_cortes = volume_zyx.shape[0]
    hechos = {}

    def preparado(z):
        if z not in hechos:
            hechos[z] = np.asarray(preprocess_fn(volume_zyx[z]), dtype=np.float32)
        return hechos[z]

    return np.stack([np.stack([preparado(k) for k in indices_vecinos(z, n_cortes, in_channels)])
                     for z in z_indices]).astype(np.float32)
