import numpy as np

MIN_MM = 80.0
DET_GATE = 0.2
MENSAJE_SIN_PELVIS = ("No se detectó una pelvis completa en este estudio. El modelo está diseñado solo para "
                      "tomografías de pelvis, así que no se muestran fragmentos.")


def scores_de_bundle(bundle):
    """Mejor score por corte a partir de bundle["dets"] (salida de infer_case): 0.0 si el corte no tiene cajas."""
    out = []
    for det in bundle["dets"]:
        scores = None if det is None else np.asarray(det["scores"]).reshape(-1)
        out.append(float(scores.max()) if scores is not None and scores.size else 0.0)
    return np.asarray(out, dtype=np.float64)


def racha_mas_larga(activos):
    """(inicio, fin, largo) de la racha más larga de True consecutivos; (-1, -1, 0) si no hay ninguno.
    Si hay empate gana la primera."""
    activos = np.asarray(activos, dtype=bool).reshape(-1)
    mejor = (-1, -1, 0)
    inicio = None
    for i, a in enumerate(np.r_[activos, False]):
        if a and inicio is None:
            inicio = i
        elif not a and inicio is not None:
            largo = i - inicio
            if largo > mejor[2]:
                mejor = (inicio, i - 1, largo)
            inicio = None
    return mejor


def pelvis_presente(scores_por_corte, spacing_z_mm, det_gate=DET_GATE, min_mm=MIN_MM):
    """Decide si el estudio tiene una pelvis completa. Devuelve un dict con la decisión y la evidencia."""
    s = np.asarray(scores_por_corte, dtype=np.float64).reshape(-1)
    if not np.isfinite(spacing_z_mm) or spacing_z_mm <= 0:
        raise ValueError(f"spacing_z_mm debe ser positivo, llegó {spacing_z_mm}")
    activos = (s > 0) & (s >= det_gate)
    inicio, fin, largo = racha_mas_larga(activos)
    racha_mm = largo * float(spacing_z_mm)
    return {"presente": bool(racha_mm >= min_mm), "racha_cortes": int(largo), "racha_mm": float(racha_mm),
            "inicio": int(inicio), "fin": int(fin), "cortes_activos": int(activos.sum()), "n_cortes": int(s.size),
            "det_gate": float(det_gate), "min_mm": float(min_mm)}


def pelvis_presente_bundle(bundle, spacing_z_mm, det_gate=DET_GATE, min_mm=MIN_MM):
    """Igual que pelvis_presente, a partir del bundle de infer_case (salida_3d / app)."""
    return pelvis_presente(scores_de_bundle(bundle), spacing_z_mm, det_gate=det_gate, min_mm=min_mm)
