"""
cortes_vacios.py -- Cortes sin pelvis: manifiesto de negativos, métricas sobre el volumen completo,
regla de selección del ajuste fino y filtro opcional de tramo principal en z (PENGWIN, semana 10).

POR QUÉ EXISTE
  El modelo multitarea se entrenó solo con cortes que tienen pelvis (manifest_cortes_positivos.csv). En un
  volumen completo, los cortes de arriba y de abajo (fémures, columna) reciben cajas. En val, con el modelo
  2.5D, 229 de 394 cortes sin pelvis tienen alguna caja y 212 pasan el umbral_corte de 0,08 (con el modelo
  anterior, de un corte, eran 310 y 190 con 0,45). Ninguna métrica actual lo ve, porque todas se calculan sobre cortes positivos.

QUÉ HAY AQUÍ (todo NumPy/pandas, sin torch: se prueba sin GPU)
  perfil_cortes              -> por corte: vóxeles de pelvis, si la tiene, distancia a la pelvis y posición.
  manifiesto_negativos       -> cortes SIN un solo vóxel de pelvis, por split (entran al ajuste fino).
  submuestrear_negativos     -> fracción reproducible de negativos, por caso (no se concentra en pocos casos).
  calibrar_umbral            -> misma regla que pipeline_pengwin.calibrar_umbral_corte (probada contra ella).
  metricas_cortes_vacios     -> recall de pelvis, cortes vacíos que pasan y cortes vacíos con alguna caja.
  resumen_cortes_vacios      -> lo anterior con umbral calibrado y con un umbral fijo (comparación justa).
  criterio_seleccion         -> regla escrita ANTES de entrenar para elegir época y configuración.
  tramo_principal            -> filtro opcional en z; solo se adopta si mejora en val (ver notebook).

CONVENCIONES
  * "tiene pelvis" = el label del corte tiene al menos un vóxel con ID 1..30 (igual que nms_pertenencia_pipeline).
  * Un corte "se procesa" si tiene alguna caja final y su mejor score >= umbral (igual que
    pipeline_pengwin._corte_procesado). Score 0 significa "sin cajas después del NMS".
  * Los cortes con muy poca pelvis (menos de 20 píxeles tras el letterbox) no están en el manifiesto de
    positivos; tampoco entran como negativos porque sí tienen pelvis. Quedan fuera del ajuste, a propósito.
"""
import numpy as np
import pandas as pd

COLUMNAS_PERFIL = ["slice_index", "voxeles_pelvis", "tiene_pelvis", "distancia_a_pelvis", "posicion"]
COLUMNAS_NEGATIVOS = ["split", "case_id", "slice_index", "distancia_a_pelvis", "posicion"]


# --------------------------------------------------------------------------------------------------
# 1. Qué cortes no tienen pelvis
# --------------------------------------------------------------------------------------------------
def perfil_cortes(label_zyx):
    """label (z, y, x) con IDs de fragmento (0 = fondo) -> DataFrame con una fila por corte.

    distancia_a_pelvis: cortes hasta el corte con pelvis más cercano (0 si el corte la tiene; -1 si el volumen
    no tiene pelvis). posicion: 'pelvis', 'antes' (índice menor que el primer corte con pelvis), 'despues'
    (mayor que el último) o 'hueco' (sin pelvis pero entre dos cortes con pelvis). No se asume orientación
    anatómica: 'antes' y 'despues' se refieren al índice z del array.
    """
    lab = np.asarray(label_zyx)
    if lab.ndim != 3:
        raise ValueError(f"Se esperaba un volumen (z, y, x); llegó {lab.shape}")
    n_z = lab.shape[0]
    voxeles = (lab.reshape(n_z, -1) != 0).sum(axis=1).astype(np.int64)
    tiene = voxeles > 0
    z = np.arange(n_z)
    if tiene.any():
        con = np.flatnonzero(tiene)
        # distancia al corte con pelvis más cercano: búsqueda binaria sobre los índices con pelvis
        pos = np.searchsorted(con, z)
        izq = con[np.clip(pos - 1, 0, len(con) - 1)]
        der = con[np.clip(pos, 0, len(con) - 1)]
        distancia = np.minimum(np.abs(z - izq), np.abs(z - der))
        primero, ultimo = con.min(), con.max()
        posicion = np.where(tiene, "pelvis",
                            np.where(z < primero, "antes", np.where(z > ultimo, "despues", "hueco")))
    else:
        distancia = np.full(n_z, -1, dtype=np.int64)
        posicion = np.full(n_z, "sin_pelvis", dtype=object)
    return pd.DataFrame({"slice_index": z, "voxeles_pelvis": voxeles, "tiene_pelvis": tiene,
                         "distancia_a_pelvis": distancia.astype(np.int64), "posicion": posicion},
                        columns=COLUMNAS_PERFIL)


def manifiesto_negativos(casos, leer_label):
    """casos: DataFrame con columnas case_id y split. leer_label(case_id) -> array (z, y, x) del label MHA.
    Devuelve los cortes SIN un solo vóxel de pelvis, con su distancia a la pelvis y su posición."""
    for col in ("case_id", "split"):
        if col not in casos.columns:
            raise KeyError(f"casos debe tener la columna {col!r}")
    if casos["case_id"].duplicated().any():
        raise ValueError("casos tiene case_id repetidos")
    partes = []
    for fila in casos.sort_values("case_id").itertuples(index=False):
        perfil = perfil_cortes(leer_label(fila.case_id))
        neg = perfil[~perfil["tiene_pelvis"]].copy()
        neg.insert(0, "case_id", str(fila.case_id))
        neg.insert(0, "split", fila.split)
        partes.append(neg[COLUMNAS_NEGATIVOS])
    if not partes:
        return pd.DataFrame(columns=COLUMNAS_NEGATIVOS)
    return pd.concat(partes, ignore_index=True)


def submuestrear_negativos(negativos, fraccion, seed=42):
    """Toma `fraccion` de los negativos de CADA caso (redondeo hacia arriba, al menos 1 si el caso tiene
    negativos), de forma reproducible. fraccion = 1.0 devuelve todos. Orden de salida: case_id, slice_index."""
    if not 0.0 < float(fraccion) <= 1.0:
        raise ValueError("fraccion debe estar en (0, 1]")
    if len(negativos) == 0:
        return negativos.copy()
    rng = np.random.default_rng(seed)
    partes = []
    for case_id, grupo in negativos.sort_values(["case_id", "slice_index"]).groupby("case_id", sort=True):
        k = int(np.ceil(len(grupo) * float(fraccion)))
        idx = np.sort(rng.choice(len(grupo), size=k, replace=False))
        partes.append(grupo.iloc[idx])
    return pd.concat(partes).sort_values(["case_id", "slice_index"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------------------
# 2. Métricas sobre el volumen completo
# --------------------------------------------------------------------------------------------------
def _validar(scores, tiene_pelvis):
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    p = np.asarray(tiene_pelvis, dtype=bool).reshape(-1)
    if s.shape != p.shape:
        raise ValueError("scores y tiene_pelvis deben tener la misma longitud")
    if not np.isfinite(s).all():
        raise ValueError("Hay scores no finitos")
    return s, p


def procesa(scores, umbral):
    """Misma regla que pipeline_pengwin._corte_procesado: hay alguna caja (score > 0) y score >= umbral."""
    s = np.asarray(scores, dtype=np.float64)
    return (s > 0) & (s >= float(umbral))


def calibrar_umbral(scores, tiene_pelvis, recall_min=0.99, paso=0.01):
    """El MAYOR umbral que conserva al menos `recall_min` de los cortes con pelvis (misma regla y misma rejilla
    que pipeline_pengwin.calibrar_umbral_corte). Devuelve (umbral, tabla)."""
    s, p = _validar(scores, tiene_pelvis)
    if not p.any():
        raise ValueError("Se necesita al menos un corte con pelvis para calibrar")
    filas = []
    for u in np.round(np.arange(0.0, 1.0 + 1e-9, paso), 6):
        pr = procesa(s, u)
        filas.append({"umbral": float(u), "recall_hueso": float(pr[p].mean()),
                      "rechazo_vacios": float((~pr[~p]).mean()) if (~p).any() else float("nan"),
                      "cortes_procesados": int(pr.sum())})
    tabla = pd.DataFrame(filas)
    validos = tabla[tabla["recall_hueso"] >= recall_min]
    return (float(validos["umbral"].max()) if len(validos) else 0.0), tabla


def metricas_cortes_vacios(scores, tiene_pelvis, umbral):
    """Con un umbral dado: recall de pelvis, fracción de cortes vacíos que pasan (falsos positivos) y fracción
    de cortes vacíos con alguna caja (score > 0, sin importar el umbral)."""
    s, p = _validar(scores, tiene_pelvis)
    pr = procesa(s, umbral)
    n_p, n_v = int(p.sum()), int((~p).sum())
    return {"umbral": float(umbral), "n_pelvis": n_p, "n_vacios": n_v,
            "recall_pelvis": float(pr[p].mean()) if n_p else float("nan"),
            "fpr_vacios": float(pr[~p].mean()) if n_v else float("nan"),
            "vacios_que_pasan": int(pr[~p].sum()),
            "vacios_con_caja": float((s[~p] > 0).mean()) if n_v else float("nan")}


def resumen_cortes_vacios(scores, tiene_pelvis, recall_min=0.99, umbral_fijo=0.45):
    """Dos lecturas complementarias, con nombres planos (para historiales CSV):
      *_cal  : umbral recalibrado para este modelo con la regla de recall_min (lo que se usaría en la práctica).
      *_fijo : el umbral_corte vigente del proyecto (umbral_corte.json), para comparar modelos con la misma vara."""
    u_cal, _ = calibrar_umbral(scores, tiene_pelvis, recall_min=recall_min)
    cal = metricas_cortes_vacios(scores, tiene_pelvis, u_cal)
    fijo = metricas_cortes_vacios(scores, tiene_pelvis, umbral_fijo)
    return {"umbral_cal": cal["umbral"], "recall_pelvis_cal": cal["recall_pelvis"], "fpr_vacios_cal": cal["fpr_vacios"],
            "umbral_fijo": fijo["umbral"], "recall_pelvis_fijo": fijo["recall_pelvis"],
            "fpr_vacios_fijo": fijo["fpr_vacios"], "vacios_con_caja": cal["vacios_con_caja"],
            "n_pelvis": cal["n_pelvis"], "n_vacios": cal["n_vacios"]}


# --------------------------------------------------------------------------------------------------
# 3. Regla de selección (escrita antes de entrenar)
# --------------------------------------------------------------------------------------------------
CRITERIO = {
    "metrica": "fpr_vacios_cal",      # menos cortes vacíos que pasan, con el umbral recalibrado (recall de pelvis >= 0,99)
    "tol_map50": 0.01,                # la mAP@0.50 en cortes positivos de val no puede bajar más de 0,01
    "tol_dice": 0.01,                 # el Dice con cajas predichas no puede bajar más de 0,01
    "recall_min": 0.99,               # con el umbral recalibrado se debe conservar al menos el 99 % de la pelvis
}


def criterio_seleccion(metricas, base, criterio=CRITERIO, prefijo="val_"):
    """metricas, base: dicts con val_det_map50, val_seg_dice_pred, val_fpr_vacios_cal y val_recall_pelvis_cal
    (prefijo configurable). Devuelve (cumple, puntaje, detalle). Mayor puntaje = mejor.
      cumple  -> puntaje = 1 - fpr (en [0, 1]).
      no cumple -> puntaje = -(peor incumplimiento) (siempre < 0): nunca le gana a una época que cumple."""
    k_map, k_dice = f"{prefijo}det_map50", f"{prefijo}seg_dice_pred"
    k_fpr, k_rec = f"{prefijo}{criterio['metrica']}", f"{prefijo}recall_pelvis_cal"
    for k in (k_map, k_dice, k_fpr):
        if k not in metricas or k not in base:
            raise KeyError(f"Falta {k} en las métricas o en la línea base")
    if k_rec not in metricas:
        raise KeyError(f"Falta {k_rec} en las métricas")
    caida_map = float(base[k_map]) - float(metricas[k_map])
    caida_dice = float(base[k_dice]) - float(metricas[k_dice])
    falta_recall = float(criterio["recall_min"]) - float(metricas[k_rec])
    exceso = max(caida_map - criterio["tol_map50"], caida_dice - criterio["tol_dice"], falta_recall, 0.0)
    fpr = float(metricas[k_fpr])
    cumple = bool(exceso <= 1e-12 and np.isfinite(fpr))
    detalle = {"caida_map50": caida_map, "caida_dice": caida_dice, "recall_pelvis": float(metricas[k_rec]),
               "fpr": fpr, "mejora_fpr": float(base[k_fpr]) - fpr}
    if cumple:
        return True, 1.0 - fpr, detalle
    return False, -max(exceso, 1e-9), detalle


# --------------------------------------------------------------------------------------------------
# 4. Filtro opcional de tramo principal en z
# --------------------------------------------------------------------------------------------------
def tramo_principal(activos, max_hueco=0):
    """activos: bool por corte (el corte pasa el umbral). Une rachas separadas por huecos de <= max_hueco cortes
    y conserva solo el grupo con más cortes activos (la pelvis ocupa un tramo continuo de z). Devuelve el bool
    filtrado (nunca activa cortes que no lo estaban). Empate: gana el grupo con menor índice z."""
    a = np.asarray(activos, dtype=bool).reshape(-1)
    if max_hueco < 0:
        raise ValueError("max_hueco debe ser >= 0")
    if not a.any():
        return a.copy()
    idx = np.flatnonzero(a)
    grupos = [[idx[0]]]
    for z in idx[1:]:
        if z - grupos[-1][-1] - 1 <= max_hueco:
            grupos[-1].append(z)
        else:
            grupos.append([z])
    mejor = max(range(len(grupos)), key=lambda k: (len(grupos[k]), -k))
    salida = np.zeros_like(a)
    salida[np.asarray(grupos[mejor])] = True
    return salida


def evaluar_tramo_principal(tabla_scores, umbral, huecos=(0, 2, 5, 10)):
    """tabla_scores: case_id, z, tiene_hueso (o tiene_pelvis), score_max (formato de scores_cortes_val.csv).
    Para cada max_hueco: falsos positivos que quita y cortes con pelvis que pierde (sumados sobre los casos)."""
    col = "tiene_pelvis" if "tiene_pelvis" in tabla_scores.columns else "tiene_hueso"
    filas = []
    for h in huecos:
        fp_antes = fp_despues = pelvis_perdida = n_pelvis = 0
        for _, g in tabla_scores.sort_values(["case_id", "z"]).groupby("case_id"):
            p = g[col].to_numpy(bool)
            act = procesa(g["score_max"].to_numpy(), umbral)
            filt = tramo_principal(act, h)
            fp_antes += int((act & ~p).sum())
            fp_despues += int((filt & ~p).sum())
            pelvis_perdida += int((act & ~filt & p).sum())
            n_pelvis += int(p.sum())
        filas.append({"max_hueco": h, "vacios_que_pasan_antes": fp_antes, "vacios_que_pasan_despues": fp_despues,
                      "cortes_pelvis_perdidos": pelvis_perdida, "n_cortes_pelvis": n_pelvis})
    return pd.DataFrame(filas)
