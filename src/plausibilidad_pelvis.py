"""plausibilidad_pelvis.py -- ¿La pelvis encontrada se parece a una pelvis humana adulta como las de PENGWIN?

POR QUÉ EXISTE
  control_pelvis.py decide si HAY una pelvis; no si es una pelvis esperable. Una pelvis de animal, o un estudio muy
  distinto de los de entrenamiento, puede pasar ese control y el modelo la procesa igual. Aquí se mide la pelvis que
  encontró el modelo y se compara con la de los 75 pacientes de entrenamiento. Si algo queda fuera de lo visto, la app
  muestra una ADVERTENCIA (no bloquea nada): "esta tomografía se sale de lo esperado para una pelvis humana adulta".

QUÉ MIDE (en el volumen canónico LPS: x hacia la izquierda del paciente, y hacia atrás, z hacia arriba)
  - ancho, profundidad y alto de la pelvis en mm, con el espaciado físico del archivo. Se usan los percentiles 0.5 y
    99.5 de la posición del hueso en cada eje, no el mínimo y el máximo, para que un vóxel suelto lejos de la pelvis
    no infle la medida.
  - volumen total de hueso (cm³) y fracción que es sacro.
  - que estén los tres huesos y en orden: coxal derecho, sacro y coxal izquierdo de derecha a izquierda del paciente.
  - la simetría entre los dos coxales (volumen menor / mayor) se mide y se guarda, pero NO dispara el aviso: no
    distingue una pelvis animal (también es simétrica) y sí salta cuando el modelo segmenta mal un coxal. En la
    verificación fue la única causa de una falsa alarma (caso 012 de prueba, modelo axial: 0.71).

REGLA (dos niveles)
  Con 75 pacientes no se ven todos los extremos de una pelvis humana: una pelvis real puede quedar fuera del rango de
  entrenamiento. Por eso hay dos niveles por medida:
  - Nota (informativa): la medida queda fuera del rango observado en los pacientes de entrenamiento. Es poco común,
    pero puede ser una pelvis humana.
  - Advertencia: la medida queda fuera de un INTERVALO DE TOLERANCIA amplio, el rango donde cae el 99.9 % de las
    pelvis como las de entrenamiento, con un 99 % de confianza (intervalo normal en escala logarítmica, factor de
    Howe, k ≈ 4 con 75 casos). Su ancho depende de cuánto varía cada medida. También hay advertencia si falta un
    hueso o si los huesos están en otro orden.
  Se eligió una cobertura tan alta porque las pelvis de PENGWIN varían poco entre sí: con el 99 % y 95 % de confianza
  (k ≈ 3) el intervalo quedaba más angosto que el rango observado ampliado en un 15 %, la regla anterior.
  Los rangos están en resultados/metricas/plausibilidad/referencia_pelvis.json y se calculan con las anotaciones de
  entrenamiento:  python -m src.plausibilidad_pelvis referencia

LIMITACIONES
  No puede decir que algo "no es humano": nunca vio una pelvis animal. Detecta lo que se sale de lo visto. Una pelvis
  de tamaño y forma parecidos a la humana (p. ej. de un primate grande) no se detectaría. Tampoco se pudo probar con
  tomografías de animales reales: se prueba con casos simulados (tamaño escalado, lados invertidos, hueso faltante) y
  con que los pacientes humanos de validación y prueba no la disparen.

Todo es NumPy: se prueba sin GPU (tests/test_plausibilidad_pelvis.py).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rutas import DATASET_CSV, METRICAS_DIR  # noqa: E402

REFERENCIA_JSON = METRICAS_DIR / "plausibilidad" / "referencia_pelvis.json"
COBERTURA, CONFIANZA = 0.999, 0.99   # límite de la advertencia: 99.9 % de las pelvis, con 99 % de confianza
MARGEN = 0.15                   # solo para referencias antiguas sin intervalo de tolerancia
PERCENTIL = 0.5                 # extremos que se ignoran en cada eje al medir el tamaño (%)
SACRO, COXAL_IZQ, COXAL_DER = 1, 2, 3
NOMBRE_REGION = {SACRO: "sacro", COXAL_IZQ: "coxal izquierdo", COXAL_DER: "coxal derecho"}
REGION_DE_NOMBRE = {"sacro": SACRO, "coxal_izquierdo": COXAL_IZQ, "coxal_derecho": COXAL_DER}
MEDIDAS = {                     # clave -> (descripción para el aviso, unidad)
    "ancho_mm": ("ancho de la pelvis", "mm"),
    "profundidad_mm": ("profundidad de la pelvis", "mm"),
    "alto_mm": ("altura de la pelvis", "mm"),
    "volumen_cm3": ("volumen de hueso", "cm³"),
    "fraccion_sacro": ("fracción del hueso que es sacro", ""),
    "simetria_coxales": ("simetría entre los coxales (menor / mayor)", ""),
}
CONTROLADAS = ("ancho_mm", "profundidad_mm", "alto_mm", "volumen_cm3", "fraccion_sacro")   # las que disparan el aviso
MENSAJE = ("Esta tomografía se sale de lo esperado para una pelvis humana adulta como las del dataset de "
           "entrenamiento. Los resultados pueden no ser válidos")


# ------------------------------------------------------------------ medidas
def regiones_desde_etiqueta(label):
    """Etiqueta de PENGWIN (1-10 sacro, 11-20 coxal izq., 21-30 coxal der.) -> volumen de regiones 0-3."""
    label = np.asarray(label)
    out = np.zeros(label.shape, dtype=np.uint8)
    out[(label >= 1) & (label <= 10)] = SACRO
    out[(label >= 11) & (label <= 20)] = COXAL_IZQ
    out[(label >= 21) & (label <= 30)] = COXAL_DER
    return out


def regiones_desde_instancias(instancias, region_by_id):
    """Volumen de instancias predichas + {id: 'sacro' | 'coxal_izquierdo' | 'coxal_derecho'} -> regiones 0-3."""
    instancias = np.asarray(instancias)
    lut = np.zeros(int(instancias.max()) + 1 if instancias.size else 1, dtype=np.uint8)
    for fid, region in region_by_id.items():
        if 0 < int(fid) < len(lut):
            lut[int(fid)] = REGION_DE_NOMBRE[region]
    return lut[instancias]


def _rango_percentil(conteos, spacing_mm, p=PERCENTIL):
    """Largo (mm) entre los percentiles p y 100-p de la posición, a partir de los vóxeles por índice de un eje."""
    total = conteos.sum()
    if total == 0:
        return 0.0
    acumulado = np.cumsum(conteos) / total
    i0 = int(np.searchsorted(acumulado, p / 100.0))
    i1 = int(np.searchsorted(acumulado, 1.0 - p / 100.0))
    return float((i1 - i0 + 1) * spacing_mm)


def medidas_pelvis(regiones, spacing_xyz):
    """Medidas de una pelvis a partir de un volumen de regiones 0-3 (z, y, x) canónico y el espaciado (x, y, z)."""
    regiones = np.asarray(regiones)
    sx, sy, sz = (float(v) for v in spacing_xyz)
    vox_cm3 = sx * sy * sz / 1000.0
    hueso = regiones > 0
    m = {"ancho_mm": _rango_percentil(hueso.sum(axis=(0, 1)), sx),
         "profundidad_mm": _rango_percentil(hueso.sum(axis=(0, 2)), sy),
         "alto_mm": _rango_percentil(hueso.sum(axis=(1, 2)), sz)}
    volumen, centro_x = {}, {}
    xs = np.arange(regiones.shape[2])
    for r in (SACRO, COXAL_IZQ, COXAL_DER):
        por_x = (regiones == r).sum(axis=(0, 1))
        n = int(por_x.sum())
        volumen[r] = n * vox_cm3
        centro_x[r] = float((por_x * xs).sum() / n) * sx if n else None
    total = sum(volumen.values())
    m["volumen_cm3"] = total
    m["fraccion_sacro"] = volumen[SACRO] / total if total else 0.0
    mayor = max(volumen[COXAL_IZQ], volumen[COXAL_DER])
    m["simetria_coxales"] = min(volumen[COXAL_IZQ], volumen[COXAL_DER]) / mayor if mayor else 0.0
    m["faltan"] = [NOMBRE_REGION[r] for r in (SACRO, COXAL_IZQ, COXAL_DER) if volumen[r] == 0]
    m["orden_lateral_ok"] = (not m["faltan"]
                             and centro_x[COXAL_DER] < centro_x[SACRO] < centro_x[COXAL_IZQ])
    return m


# ------------------------------------------------------------------ evaluación
def cargar_referencia(ruta=REFERENCIA_JSON):
    ruta = Path(ruta)
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.is_file() else None


def _fmt(valor, unidad):
    # mm con un decimal: con enteros, 163.1 frente a un máximo de 162.7 se leería "163 fuera de 117 a 163"
    return f"{valor:.1f} {unidad}" if unidad == "mm" else f"{valor:.0f} {unidad}" if unidad else f"{valor:.2f}"


def factor_tolerancia(n, cobertura=COBERTURA, confianza=CONFIANZA):
    """Factor k del intervalo de tolerancia normal bilateral (aproximación de Howe): media ± k·desviación cubre la
    fracción `cobertura` de la población con la `confianza` dada, estimando media y desviación con n casos."""
    from scipy.stats import chi2, norm
    z = norm.ppf((1 + cobertura) / 2)
    return float(z * np.sqrt((n - 1) * (1 + 1 / n) / chi2.ppf(1 - confianza, n - 1)))


def intervalo_tolerancia(valores, cobertura=COBERTURA, confianza=CONFIANZA):
    """(mínimo, máximo) del intervalo de tolerancia, calculado en escala logarítmica (medidas positivas)."""
    logv = np.log(np.asarray(valores, dtype=float))
    k = factor_tolerancia(len(logv), cobertura, confianza)
    return float(np.exp(logv.mean() - k * logv.std(ddof=1))), float(np.exp(logv.mean() + k * logv.std(ddof=1)))


def evaluar_niveles(medidas, referencia, margen=MARGEN):
    """{"avisos": [...], "notas": [...]}: avisos si la pelvis se sale de lo esperable para una pelvis humana adulta
    (intervalo de tolerancia); notas si solo se sale del rango de los pacientes de entrenamiento."""
    avisos, notas = [], []
    if medidas["faltan"]:
        avisos.append(f"no se encontró: {', '.join(medidas['faltan'])}")
    elif not medidas["orden_lateral_ok"]:
        avisos.append("los huesos no están en el orden esperado (coxal derecho, sacro y coxal izquierdo de derecha a "
                      "izquierda del paciente)")
    for clave in CONTROLADAS:
        descripcion, unidad = MEDIDAS[clave]
        rango = referencia["rangos"].get(clave)
        if rango is None:
            continue
        valor, lo, hi = medidas[clave], rango["min"], rango["max"]
        # referencias antiguas sin intervalo de tolerancia: rango observado ampliado en un margen fijo
        t_lo, t_hi = rango.get("tolerancia_min", lo * (1 - margen)), rango.get("tolerancia_max", hi * (1 + margen))
        if valor < t_lo or valor > t_hi:
            avisos.append(f"{descripcion}: {_fmt(valor, unidad)}; lo esperable para una pelvis humana adulta va de "
                          f"{_fmt(t_lo, unidad)} a {_fmt(t_hi, unidad)}")
        elif valor < lo or valor > hi:
            notas.append(f"{descripcion}: {_fmt(valor, unidad)}, fuera del rango de los pacientes de entrenamiento "
                         f"({_fmt(lo, unidad)} a {_fmt(hi, unidad)}) pero dentro de lo esperable para una pelvis "
                         f"humana adulta ({_fmt(t_lo, unidad)} a {_fmt(t_hi, unidad)})")
    return {"avisos": avisos, "notas": notas}


def evaluar(medidas, referencia, margen=MARGEN):
    """Solo los avisos (lista vacía si la pelvis es esperable). Ver evaluar_niveles para las notas."""
    return evaluar_niveles(medidas, referencia, margen)["avisos"]


def mensaje(motivos):
    return f"{MENSAJE}: " + "; ".join(motivos) + "." if motivos else ""


def mensaje_nota(notas):
    return ("Medida poco común, pero compatible con una pelvis humana adulta: " + "; ".join(notas) + "."
            if notas else "")


def escalar(medidas, factor):
    """Las medidas que tendría la misma pelvis si fuera `factor` veces más grande (simulación de otro tamaño)."""
    m = dict(medidas)
    for clave in ("ancho_mm", "profundidad_mm", "alto_mm"):
        m[clave] = medidas[clave] * factor
    m["volumen_cm3"] = medidas["volumen_cm3"] * factor ** 3
    return m


# ------------------------------------------------------------------ comandos
def _casos(split):
    import pandas as pd
    df = pd.read_csv(DATASET_CSV, dtype={"case_id": str})
    return df.loc[df["split"] == split, "case_id"].tolist()


def _medidas_de_etiquetas(casos):
    sys.path.insert(0, str(ROOT / "app"))
    import inferencia
    filas = []
    for caso in casos:
        label, spacing = inferencia.leer_canonico(inferencia.ETIQUETAS_DIR / f"{caso}.mha", es_etiqueta=True)
        filas.append({"caso": caso, **medidas_pelvis(regiones_desde_etiqueta(label), spacing)})
        print(f"  {caso}: ancho {filas[-1]['ancho_mm']:.0f} mm | alto {filas[-1]['alto_mm']:.0f} mm | "
              f"{filas[-1]['volumen_cm3']:.0f} cm³", flush=True)
    return filas


def calcular_referencia():
    """Rangos de cada medida en las anotaciones de los pacientes de entrenamiento."""
    casos = _casos("train")
    print(f"Referencia con las anotaciones de {len(casos)} pacientes de entrenamiento", flush=True)
    filas = _medidas_de_etiquetas(casos)
    import pandas as pd
    REFERENCIA_JSON.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).drop(columns=["faltan"]).to_csv(REFERENCIA_JSON.parent / "medidas_entrenamiento.csv",
                                                        index=False)
    rangos = {}
    for clave in MEDIDAS:
        v = np.array([f[clave] for f in filas], dtype=float)
        t_lo, t_hi = intervalo_tolerancia(v)
        rangos[clave] = {"min": float(v.min()), "max": float(v.max()), "p5": float(np.percentile(v, 5)),
                         "mediana": float(np.median(v)), "p95": float(np.percentile(v, 95)),
                         "tolerancia_min": t_lo, "tolerancia_max": t_hi}
    ref = {"fuente": "anotaciones de los pacientes de entrenamiento (splits/dataset.csv, split train)",
           "n_casos": len(filas), "percentil_tamano": PERCENTIL,
           "intervalo_tolerancia": {"cobertura": COBERTURA, "confianza": CONFIANZA, "escala": "logaritmica",
                                    "factor_k": factor_tolerancia(len(filas))},
           "orden_lateral_ok": int(sum(f["orden_lateral_ok"] for f in filas)), "rangos": rangos}
    REFERENCIA_JSON.write_text(json.dumps(ref, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado en {REFERENCIA_JSON.relative_to(ROOT)}")
    for clave, r in rangos.items():
        print(f"  {clave:18s} entrenamiento {r['min']:9.2f} – {r['max']:9.2f} | tolerancia "
              f"{r['tolerancia_min']:9.2f} – {r['tolerancia_max']:9.2f}")


def verificar(predicciones=False):
    """Falsas alarmas en pacientes humanos no usados para la referencia (val y test) y casos simulados."""
    import pandas as pd
    ref = cargar_referencia()
    if ref is None:
        raise SystemExit("Falta la referencia: python -m src.plausibilidad_pelvis referencia")
    casos = _casos("val") + _casos("test")
    filas = []
    print(f"Anotaciones de {len(casos)} pacientes de validación y prueba", flush=True)
    for f in _medidas_de_etiquetas(casos):
        filas.append({"caso": f["caso"], "fuente": "anotacion", **{k: f[k] for k in MEDIDAS},
                      "motivos": " | ".join(evaluar(f, ref)),
                      "notas": " | ".join(evaluar_niveles(f, ref)["notas"])})
    if predicciones:
        sys.path.insert(0, str(ROOT / "app"))
        import inferencia
        from src.rutas import DATA_MHA_DIR
        pipeline = inferencia.cargar_pipeline()
        print(f"Predicciones de los tres modelos ({pipeline['dispositivo']})", flush=True)
        for caso in casos:
            volume, spacing = inferencia.leer_canonico(next(DATA_MHA_DIR.glob(f"*images*/{caso}.mha")))
            for plano in inferencia.PLANOS:
                res = inferencia.predecir_plano(pipeline, plano, volume, spacing, case_id=caso)
                m = medidas_pelvis(regiones_desde_instancias(res["instance_volume"], res["region_by_id"]), spacing)
                filas.append({"caso": caso, "fuente": f"prediccion_{plano}", **{k: m[k] for k in MEDIDAS},
                              "motivos": " | ".join(evaluar(m, ref)),
                              "notas": " | ".join(evaluar_niveles(m, ref)["notas"])})
                print(f"  {caso} {plano:8s} {'OK' if not filas[-1]['motivos'] else 'AVISO: ' + filas[-1]['motivos']}",
                      flush=True)
            del volume
    tabla = pd.DataFrame(filas)

    # simulaciones sobre las medidas reales: otro tamaño, lados invertidos y un hueso faltante
    sim = []
    for _, f in tabla[tabla["fuente"] == "anotacion"].iterrows():
        base = {k: f[k] for k in MEDIDAS} | {"faltan": [], "orden_lateral_ok": True}
        casos_sim = {"mitad de tamaño": escalar(base, 0.5), "dos tercios del tamaño": escalar(base, 2 / 3),
                     "1.6 veces el tamaño": escalar(base, 1.6), "lados invertidos": base | {"orden_lateral_ok": False},
                     "sin sacro": base | {"faltan": ["sacro"]}}
        for nombre, m in casos_sim.items():
            sim.append({"caso": f["caso"], "simulacion": nombre, "detectado": bool(evaluar(m, ref))})
    sim = pd.DataFrame(sim)

    salida = REFERENCIA_JSON.parent
    tabla.to_csv(salida / "verificacion_humanos.csv", index=False)
    sim.to_csv(salida / "verificacion_simulaciones.csv", index=False)
    print("\nPacientes humanos: advertencias (deberían ser 0) y notas (medidas fuera del rango de entrenamiento):")
    for fuente, g in tabla.groupby("fuente", sort=False):
        print(f"  {fuente:20s} advertencias {int((g['motivos'] != '').sum())} de {len(g)} | "
              f"notas {int((g['notas'] != '').sum())}")
    print("\nCasos simulados detectados (deberían ser todos):")
    for nombre, g in sim.groupby("simulacion", sort=False):
        print(f"  {nombre:24s} {int(g['detectado'].sum())} de {len(g)}")
    print(f"\nGuardado en {salida.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="accion", required=True)
    sub.add_parser("referencia", help="calcula los rangos con las anotaciones de entrenamiento (CPU)")
    p = sub.add_parser("verificar", help="falsas alarmas en val y test y casos simulados")
    p.add_argument("--predicciones", action="store_true",
                   help="además, con las predicciones de los tres modelos (GPU recomendada)")
    args = parser.parse_args()
    if args.accion == "referencia":
        calcular_referencia()
    else:
        verificar(args.predicciones)


if __name__ == "__main__":
    main()
