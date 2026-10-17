"""Corre sin supervisión todo lo pendiente de la evaluación final y la ablación, en orden y con registro.

Pasos (cada uno es un comando independiente; ver --listar):
  1. Exportar los modelos de la etapa B de los tres planos (best_B_decoder.pth -> modelo_multitarea_{plano}_B.pt).
  2. Entrenar la cabeza de pertenencia (etapa D, ~1 min) sobre cada tronco B: el evaluador exige una cabeza
     entrenada sobre el mismo tronco.
  3. Evaluar en test los modelos en uso (etapa C) con el evaluador actual y SAM: son las cifras finales.
  4. Evaluar en test los modelos B (sin SAM), para la comparación C vs B.
  4b. Verificación en val del control de pelvis y de los conminutos (src/verificar_planos_val.py). Solo lee modelos
     y datos; escribe en resultados/metricas/verificacion_planos/. Su resumen queda al final de
     resultados/metricas/corrida_pendientes/logs/verificar_planos_val.log.
  5. Preentrenar FundidoraPC en Oxford-IIIT Pet (transfer learning).
  6. Prueba corta de cada variante de la ablación (minutos; detecta errores antes de las corridas largas).
  7. Entrenar cada variante (plano axial, etapas A y B): sin CBAM, con TL, sin CBAM con TL.
  8. Cabeza de pertenencia y evaluación en test de cada variante.
  9. Tablas con intervalos de confianza: C vs B y ablación (src/resumen_ablacion.py).

Comportamiento
- Retoma: un paso cuya salida ya existe se omite. Se puede cortar (Ctrl+C) y volver a lanzar.
- Si un paso falla se registra y se sigue con los que no dependen de él.
- Registro: resultados/metricas/corrida_pendientes/ (corrida.log, estado.json y un log por paso en logs/).
- Mientras corre, Windows no entra en suspensión (no cambia la configuración de energía; al terminar se restablece).

Desde la raíz del repositorio, con el .venv activo:

    python -m src.correr_pendientes --listar      # qué se haría y qué ya está hecho
    python -m src.correr_pendientes               # todo
    python -m src.correr_pendientes --rehacer eval_test_C
"""
import argparse
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ablacion import PESOS_PETS, VARIANTES, modelo_b_en_uso, modelo_de  # noqa: E402
from src.rutas import METRICAS_DIR, MODELOS_DIR  # noqa: E402

REGISTRO = METRICAS_DIR / "corrida_pendientes"
EVAL = METRICAS_DIR / "evaluacion"
CABEZAS = MODELOS_DIR / "nms_pertenencia"
SAM = MODELOS_DIR / "modelos_sam" / "sam_vit_b_01ec64.pth"
PLANOS = ("axial", "coronal", "sagital")
PY = [sys.executable, "-u", "-m"]


# ------------------------------------------------------------------ comprobaciones de "ya está hecho"
def _sha16(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def cabeza_lista(cabeza, tronco):
    meta = cabeza.with_name(f"{cabeza.stem}_meta.json")
    return (cabeza.is_file() and meta.is_file() and Path(tronco).is_file()
            and json.loads(meta.read_text(encoding="utf-8")).get("tronco_sha256_16") == _sha16(tronco))


def evaluacion_lista(etiqueta, split="test", con_sam=False):
    """Hecha con el evaluador actual: métricas de pertenencia y columna n_gt_fragments (y SAM si se pide)."""
    m, c = EVAL / etiqueta / f"metricas_{split}.json", EVAL / etiqueta / f"cortes_{split}.csv"
    if not (m.is_file() and c.is_file()):
        return False
    j = json.loads(m.read_text(encoding="utf-8"))
    nuevo = "pertenencia_f1_macro" in j and "n_gt_fragments" in c.open(encoding="utf-8").readline()
    return nuevo and (not con_sam or j.get("sam") == "evaluated")


def pets_listo():
    meta = PESOS_PETS.with_name(f"{PESOS_PETS.stem}_meta.json")
    return PESOS_PETS.is_file() and meta.is_file() and json.loads(meta.read_text(encoding="utf-8")).get("epochs", 0) >= 60


def variante_lista(v):
    return modelo_de(v).is_file() and (modelo_de(v).parent.parent.parent / "metricas" /
                                        modelo_de(v).parent.name / "resumen_plano.json").is_file()


def rel(p):
    return str(Path(p).relative_to(ROOT))


# ------------------------------------------------------------------ plan
def plan():
    pasos = []

    def paso(id_, descripcion, cmd, hecho, depende=(), minutos=1):
        pasos.append({"id": id_, "descripcion": descripcion, "cmd": cmd, "hecho": hecho, "depende": list(depende),
                      "minutos": minutos})

    for p in PLANOS:
        paso(f"exportar_B_{p}", f"Exportar el modelo B del plano {p}", PY + ["src.ablacion", "exportar-b", "--plano", p],
             lambda p=p: modelo_b_en_uso(p).is_file())
    for p in PLANOS:
        cab = CABEZAS / f"cabeza_pertenencia_{p}_B.pt"
        paso(f"cabeza_B_{p}", f"Cabeza de pertenencia sobre el tronco B ({p})",
             PY + ["src.entrenar_pertenencia_planos", "--plano", p, "--device", "cuda",
                   "--checkpoint", rel(modelo_b_en_uso(p)), "--salida", rel(cab)],
             lambda p=p, cab=cab: cabeza_lista(cab, modelo_b_en_uso(p)), [f"exportar_B_{p}"], 2)
    paso("eval_test_C", "Evaluar en test los modelos en uso (etapa C) con SAM: cifras finales",
         PY + ["src.evaluacion", "--plano", "todos", "--split", "test", "--sam-checkpoint", rel(SAM), "--device", "cuda"],
         lambda: all(evaluacion_lista(p, con_sam=True) for p in PLANOS), minutos=100)
    for p in PLANOS:
        paso(f"eval_test_B_{p}", f"Evaluar en test el modelo B ({p}), sin SAM",
             PY + ["src.evaluacion", "--plano", p, "--split", "test", "--checkpoint", rel(modelo_b_en_uso(p)),
                   "--cabeza", rel(CABEZAS / f"cabeza_pertenencia_{p}_B.pt"), "--etiqueta", f"{p}_B",
                   "--latency-runs", "0", "--device", "cuda"],
             lambda p=p: evaluacion_lista(f"{p}_B"), [f"cabeza_B_{p}"], 10)
    verif = METRICAS_DIR / "verificacion_planos"
    paso("verificar_planos_val", "Verificación en val del control de pelvis y de los conminutos (src/verificar_planos_val.py)",
         PY + ["src.verificar_planos_val"],
         lambda: (verif / "control_pelvis_val.csv").is_file() and (verif / "resumen_conminutos_val.csv").is_file(),
         minutos=40)
    paso("preentrenar_pets", "Preentrenar FundidoraPC en Oxford-IIIT Pet (transfer learning)",
         PY + ["src.preentrenar_backbone_pets", "--epochs", "60", "--rehacer"], pets_listo, minutos=8)
    for v, cfg in VARIANTES.items():
        dep = ["preentrenar_pets"] if cfg["transfer_learning"] else []
        paso(f"prueba_{v}", f"Prueba corta de la ablación {v}", PY + ["src.ablacion", "entrenar", "--variante", v, "--prueba"],
             lambda v=v: variante_lista(v), dep, 2)
    for v, cfg in VARIANTES.items():
        paso(f"entrenar_{v}", f"Entrenar la variante {v} (axial, etapas A y B)",
             PY + ["src.ablacion", "entrenar", "--variante", v], lambda v=v: variante_lista(v), [f"prueba_{v}"], 95)
    for v in VARIANTES:
        cab = CABEZAS / f"cabeza_pertenencia_axial_ablacion_{v}.pt"
        paso(f"cabeza_{v}", f"Cabeza de pertenencia de la variante {v}",
             PY + ["src.entrenar_pertenencia_planos", "--plano", "axial", "--device", "cuda",
                   "--checkpoint", rel(modelo_de(v)), "--salida", rel(cab)],
             lambda v=v, cab=cab: cabeza_lista(cab, modelo_de(v)), [f"entrenar_{v}"], 2)
        paso(f"eval_test_{v}", f"Evaluar en test la variante {v}, sin SAM",
             PY + ["src.evaluacion", "--plano", "axial", "--split", "test", "--checkpoint", rel(modelo_de(v)),
                   "--cabeza", rel(cab), "--etiqueta", f"ablacion_axial_{v}", "--latency-runs", "0", "--device", "cuda"],
             lambda v=v: evaluacion_lista(f"ablacion_axial_{v}"), [f"cabeza_{v}"], 10)
    paso("resumen", "Tablas con intervalos de confianza (C vs B y ablación)",
         PY + ["src.resumen_ablacion", "--split", "test"], lambda: False, minutos=3)
    return pasos


# ------------------------------------------------------------------ ejecución
def log(msg):
    linea = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linea, flush=True)
    with open(REGISTRO / "corrida.log", "a", encoding="utf-8") as fh:
        fh.write(linea + "\n")


def no_suspender(activo):
    """Windows: pide que el equipo no se suspenda mientras corre este proceso (SetThreadExecutionState)."""
    if platform.system() != "Windows":
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if activo else 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--listar", action="store_true", help="muestra el plan y qué está hecho, sin ejecutar nada")
    parser.add_argument("--rehacer", nargs="*", default=[], help="ids de pasos que se repiten aunque estén hechos")
    parser.add_argument("--solo", nargs="*", default=None, help="ids de pasos a correr (los demás se omiten)")
    args = parser.parse_args()

    pasos = plan()
    ids = {p["id"] for p in pasos}
    for i in args.rehacer + (args.solo or []):
        if i not in ids:
            raise SystemExit(f"Paso desconocido: {i}. Ver --listar.")
    if args.listar:
        total = 0
        for p in pasos:
            hecho = p["hecho"]() and p["id"] not in args.rehacer
            total += 0 if hecho else p["minutos"]
            print(f"  {'[hecho]' if hecho else '[  -  ]'} {p['id']:28s} ~{p['minutos']:>3} min  {p['descripcion']}")
        print(f"\nTiempo estimado de lo pendiente: ~{total / 60:.1f} h")
        return

    REGISTRO.mkdir(parents=True, exist_ok=True)
    (REGISTRO / "logs").mkdir(exist_ok=True)
    estado_path = REGISTRO / "estado.json"
    estado = json.loads(estado_path.read_text(encoding="utf-8")) if estado_path.is_file() else {}
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"}
    fallidos, t_total = set(), time.time()
    no_suspender(True)
    log(f"INICIO | {len(pasos)} pasos | {sys.executable}")
    try:
        for p in pasos:
            if args.solo is not None and p["id"] not in args.solo:
                continue
            if p["hecho"]() and p["id"] not in args.rehacer:
                log(f"[{p['id']}] ya está hecho: se omite")
                estado.setdefault(p["id"], {"estado": "hecho"})
                continue
            bloqueo = [d for d in p["depende"] if d in fallidos]
            if bloqueo:
                log(f"[{p['id']}] OMITIDO: depende de {bloqueo}, que falló")
                fallidos.add(p["id"])
                estado[p["id"]] = {"estado": "omitido", "por": bloqueo}
                continue
            archivo = REGISTRO / "logs" / f"{p['id']}.log"
            log(f"[{p['id']}] {p['descripcion']} (~{p['minutos']} min) -> {rel(archivo)}")
            t0 = time.time()
            with open(archivo, "w", encoding="utf-8") as fh:
                fh.write(" ".join(p["cmd"]) + "\n\n")
                fh.flush()
                codigo = subprocess.run(p["cmd"], cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
            minutos = round((time.time() - t0) / 60, 1)
            ok = codigo == 0 and (p["id"] == "resumen" or p["id"].startswith("prueba_") or p["hecho"]())
            if not ok:
                fallidos.add(p["id"])
            estado[p["id"]] = {"estado": "ok" if ok else f"falló (código {codigo})", "minutos": minutos,
                               "fin": f"{datetime.now():%Y-%m-%d %H:%M}", "log": rel(archivo)}
            estado_path.write_text(json.dumps(estado, indent=2, ensure_ascii=False), encoding="utf-8")
            log(f"[{p['id']}] {'OK' if ok else 'FALLÓ: ver ' + rel(archivo)} en {minutos} min")
    finally:
        no_suspender(False)
        estado_path.write_text(json.dumps(estado, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"FIN | {(time.time() - t_total) / 3600:.1f} h | fallidos: {sorted(fallidos) or 'ninguno'}")


if __name__ == "__main__":
    main()
