"""
control_pelvis.py -- ¿El estudio tiene una pelvis? Control previo a entregar fragmentos (PENGWIN).

POR QUÉ EXISTE
  El modelo se entrenó con tomografías de pelvis. Si recibe un estudio sin pelvis (piernas, abdomen, tórax),
  igual puede marcar huesos parecidos (fémur, columna) y la salida 3D entregaría fragmentos que no existen.
  En validación, con el modelo original y el filtro de la salida 3D (score >= 0,2):
    - con pelvis, el modelo se activa en un bloque continuo de 207 a 295 cortes (200 mm o más);
    - en los tramos sin pelvis, las rachas activas llegan como mucho a 32 cortes (unos 32 mm).
  Una pelvis completa mide mucho más que eso, así que se exige un bloque continuo de al menos MIN_MM.

REGLA (fijada antes de comparar modelos)
  Un corte está "activo" si su mejor caja (después del NMS) tiene score >= det_gate (el mismo `det_gate` de
  config_3d.json). Se busca la racha más larga de cortes activos consecutivos y se mide en milímetros con el
  espaciado z del header. Hay pelvis si esa racha mide al menos MIN_MM (80 mm).

LIMITACIONES
  El umbral sale de 5 casos de validación y de tramos sin pelvis cercanos a ella (fémur, columna lumbar).
  Un estudio que solo traiga una parte pequeña de la pelvis puede ser rechazado: por eso el mensaje dice
  "no se detectó una pelvis completa". Solo aplica a tomografías 3D, no a una imagen 2D suelta.

Todo es NumPy: se prueba sin GPU (tests/test_control_pelvis.py).
"""
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
