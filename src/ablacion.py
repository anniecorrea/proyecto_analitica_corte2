"""Ablación con/sin CBAM y con/sin transfer learning (consigna, informe técnico) y exportación de los modelos de la etapa B.

Ejecuta las celdas de notebooks/Entrenamiento_multitarea_planos.ipynb (menos la última, la que entrena los tres planos)
en un espacio de nombres propio y cambia solo lo que define cada variante. Todo lo demás (datos, cachés, semilla,
pérdidas, épocas, selección de checkpoint) es exactamente el entrenamiento de los modelos en uso.

- Solo etapas A y B (10 + 10 épocas): en validación la etapa C casi no mejora a B (+0.004 mAP@0.50 y +0.004 Dice en
  el axial), así que la ablación se compara al final de B. La base de la comparación es el best_B_decoder.pth del modelo
  axial en uso (con CBAM y sin transfer learning): mismo protocolo, no hace falta volver a entrenarla.
- Solo el plano axial: es el de menor costo (unos 95 min por corrida hasta B). CBAM y el backbone son iguales en los
  tres planos.
- Transfer learning: el backbone arranca con los pesos de src/preentrenar_backbone_pets.py; detección, decoder y
  pertenencia se entrenan desde cero.
- Cada variante escribe en entrenamiento_multitarea_axial_ablacion_{variante}/ (métricas, modelos y figuras): los
  modelos que usa la app no se tocan.

Desde la raíz del repositorio:

    python -m src.ablacion entrenar --variante sin_cbam --prueba     # prueba corta (minutos) para detectar errores
    python -m src.ablacion entrenar --variante sin_cbam
    python -m src.ablacion exportar-b --plano axial                  # modelo_multitarea_axial_B.pt del modelo en uso
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")            # sin ventanas: las figuras solo se guardan

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rutas import MODELOS_DIR, NOTEBOOKS_DIR  # noqa: E402

CUADERNO = NOTEBOOKS_DIR / "Entrenamiento_multitarea_planos.ipynb"
PLANO = "axial"
PESOS_PETS = MODELOS_DIR / "backbone_pets" / "fundidora_plus_pets.pt"
VARIANTES = {                       # la base (CBAM, sin TL) es el modelo axial en uso al final de la etapa B
    "sin_cbam": {"USE_CBAM": False, "transfer_learning": False},
    "con_tl": {"USE_CBAM": True, "transfer_learning": True},
    "sin_cbam_con_tl": {"USE_CBAM": False, "transfer_learning": True},
}


def etapa_de(variante, plano=PLANO):
    return f"entrenamiento_multitarea_{plano}_ablacion_{variante}"


def modelo_de(variante, plano=PLANO):
    """Modelo exportado de una variante (formato de inferencia, etapa B)."""
    return MODELOS_DIR / etapa_de(variante, plano) / f"modelo_multitarea_{plano}.pt"


def modelo_b_en_uso(plano):
    return MODELOS_DIR / f"entrenamiento_multitarea_{plano}" / f"modelo_multitarea_{plano}_B.pt"


def _celdas_de_codigo():
    nb = json.loads(CUADERNO.read_text(encoding="utf-8"))
    return [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]


def cargar_cuaderno(ajustes=None, reemplazos=None):
    """Ejecuta las celdas de definición del cuaderno. ajustes: valores que se cambian después de la celda de
    configuración. reemplazos: {celda: [(texto, nuevo)]} para cambiar una definición puntual."""
    os.chdir(ROOT)                                      # el cuaderno busca la raíz desde el directorio actual
    ns = {"__name__": "entrenamiento_planos"}
    for n, (i, fuente) in enumerate(_celdas_de_codigo()):
        if "correr_todo()" in fuente and "RESUMEN" in fuente:
            continue                                     # la celda que entrena los tres planos
        for viejo, nuevo in (reemplazos or {}).get(i, []):
            if fuente.count(viejo) < 1:
                raise RuntimeError(f"El cuaderno cambió: no encuentro {viejo!r} en la celda {i}")
            fuente = fuente.replace(viejo, nuevo)
        exec(compile(fuente, f"{CUADERNO.name} [celda {i}]", "exec"), ns)
        if n == 0 and ajustes:
            ns.update(ajustes)
    return ns


def _celda_con(texto):
    return next(i for i, f in _celdas_de_codigo() if texto in f)


def exportar_b(plano):
    """best_B_decoder.pth del modelo en uso -> modelo_multitarea_{plano}_B.pt (mismo formato que el modelo final)."""
    origen = MODELOS_DIR / f"entrenamiento_multitarea_{plano}" / "best_B_decoder.pth"
    destino = modelo_b_en_uso(plano)
    if destino.is_file():
        print(f"Ya existe {destino.relative_to(ROOT)}: se omite.")
        return destino
    if not origen.is_file() or origen.stat().st_size < 10_000:
        raise FileNotFoundError(f"Falta {origen} (o es un puntero de Git LFS: `git lfs pull`).")
    ns = cargar_cuaderno()
    payload = ns["export_inference_checkpoint"](origen, destino)
    print(f"{destino.relative_to(ROOT)}: etapa {payload['selected_from']['stage']}, época {payload['selected_from']['epoch']}")
    return destino


def entrenar(variante, prueba=False):
    cfg = VARIANTES[variante]
    if cfg["transfer_learning"] and not PESOS_PETS.is_file():
        raise FileNotFoundError(f"Faltan los pesos preentrenados {PESOS_PETS} (python -m src.preentrenar_backbone_pets).")
    celda_export, celda_plano = _celda_con("def exportar_y_comprobar"), _celda_con("def entrenar_plano")
    ns = cargar_cuaderno(
        ajustes={"USE_CBAM": cfg["USE_CBAM"], "PLANOS_A_ENTRENAR": [PLANO], "MODO": "completo",
                 # prueba corta: la del cuaderno (64 cortes, 1 época) da tan pocos pasos que OneCycleLR divide por
                 # cero (es el error_prueba_*.txt del cuaderno); con 256 cortes y 2 épocas alcanza
                 "AJUSTES_PRUEBA": {"MAX_TRAIN_SLICES": 256, "MAX_VAL_SLICES": 32, "MAX_NEG_SLICES": 32,
                                    "INSTANCE_EVAL_N": 8, "BATCH_SIZE": 8, "VAL_BATCH_SIZE": 8, "epochs": 2}},
        reemplazos={
            # el modelo exportado y su comprobación salen del mejor checkpoint de la etapa B
            celda_export: [("best_C_conjunto.pth", "best_B_decoder.pth")],
            # la época elegida del resumen se toma de la etapa B
            celda_plano: [('history["stage"] == "C_conjunto"', 'history["stage"] == "B_decoder"')],
        })
    ns["STAGES_BASE"] = [s for s in ns["STAGES_BASE"] if s["name"] in ("A_deteccion", "B_decoder")]
    assert [s["name"] for s in ns["STAGES_BASE"]] == ["A_deteccion", "B_decoder"]

    configurar_original = ns["configurar_plano"]

    def configurar_plano(plane, prueba_):
        configurar_original(plane, prueba_)            # canales, tamaños y semilla del plano, igual que siempre
        ns["ETAPA"] = etapa_de(variante, plane) + ("_smoke" if prueba_ else "")
        ns["OUT_DIR"], ns["MODEL_DIR"], ns["FIG_DIR"] = (ns["METRICAS_DIR"] / ns["ETAPA"], ns["MODELOS_DIR"] / ns["ETAPA"],
                                                         ns["FIGURAS_DIR"] / ns["ETAPA"])
        for d in (ns["OUT_DIR"], ns["MODEL_DIR"], ns["FIG_DIR"]):
            if prueba_:
                shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
        ns["LOG_PATH"] = ns["OUT_DIR"] / "log_corrida.txt"
    ns["configurar_plano"] = configurar_plano

    construir_original = ns["construir_modelo"]

    def construir_modelo(verbose=True):
        model = construir_original(verbose)              # mismas semillas: cabezas inicializadas igual que en la base
        if cfg["transfer_learning"]:
            import torch
            estado = torch.load(PESOS_PETS, map_location="cpu", weights_only=True)
            model.backbone.load_state_dict(estado, strict=True)
            ns["log"](f"backbone con transfer learning: {PESOS_PETS.name} (Oxford-IIIT Pet); cabezas desde cero")
        return model
    ns["construir_modelo"] = construir_modelo

    # registro propio desde la primera línea (el del cuaderno es el de los modelos en uso)
    carpeta_log = ns["METRICAS_DIR"] / (etapa_de(variante) + ("_smoke" if prueba else ""))
    carpeta_log.mkdir(parents=True, exist_ok=True)
    ns["LOG_PATH"] = carpeta_log / "log_corrida.txt"
    ns["log"](f"ABLACIÓN {variante} | CBAM {cfg['USE_CBAM']} | transfer learning {cfg['transfer_learning']} | "
              f"etapas {[s['name'] for s in ns['STAGES_BASE']]} | plano {PLANO} | {'prueba corta' if prueba else 'completo'}")
    fila = ns["entrenar_plano"](PLANO, prueba=prueba)
    fila.update({"variante": variante, "use_cbam": cfg["USE_CBAM"], "transfer_learning": cfg["transfer_learning"]})
    if not prueba:
        (ns["OUT_DIR"] / "resumen_plano.json").write_text(json.dumps(fila, indent=2, default=float), encoding="utf-8")
    print(json.dumps(fila, indent=2, default=float))
    return fila


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="accion", required=True)
    p = sub.add_parser("entrenar")
    p.add_argument("--variante", choices=list(VARIANTES), required=True)
    p.add_argument("--prueba", action="store_true", help="prueba corta: pocos cortes y 1 época por etapa")
    p = sub.add_parser("exportar-b")
    p.add_argument("--plano", choices=("axial", "coronal", "sagital"), required=True)
    args = parser.parse_args()
    if args.accion == "entrenar":
        entrenar(args.variante, args.prueba)
    else:
        exportar_b(args.plano)


if __name__ == "__main__":
    main()
