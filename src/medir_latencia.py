"""Latencia de inferencia en GPU y CPU (consigna 4.3), con el mismo pipeline del frontend (app/inferencia.py).

Mide dos cosas por dispositivo:
- por imagen: un corte axial a la vez (batch 1) por preprocesamiento + backbone/CBAM + detección + NMS + decoder;
- por volumen: la tomografía completa como en el frontend: los tres modelos (axial, coronal y sagital), cada uno con
  todos sus cortes, asociación 3D, limpieza y distancias.

Desde la raíz del repositorio, con los datos en data_mha/:

    python -m src.medir_latencia                 # caso 035 de val, GPU (si hay) y CPU
    python -m src.medir_latencia --caso 037 --cortes 60
    python -m src.medir_latencia --split test    # por imagen en los 20 casos de test; por volumen en 3 de ellos

Guarda resultados/metricas/latencia/latencia.json (un caso) o latencia_{split}.json (con --split), que lee el
dashboard. Con --split, la latencia por imagen es el promedio de los casos y la desviación es entre casos; la del
volumen completo se mide en los primeros --casos-volumen casos, porque en CPU cada tomografía tarda unos 2 minutos.
"""
import argparse
import copy
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import numpy as np
import pandas as pd
import torch

import inferencia
from rutas import DATA_MHA_DIR, DATASET_CSV, METRICAS_DIR

SALIDA_DIR = METRICAS_DIR / "latencia"


def nombre_cpu():
    if platform.system() == "Windows":
        try:
            import winreg
            clave = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(clave, "ProcessorNameString")[0].strip()
        except OSError:
            return os.environ.get("PROCESSOR_IDENTIFIER", platform.processor())
    try:
        for linea in Path("/proc/cpuinfo").read_text().splitlines():
            if linea.startswith("model name"):
                return linea.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def en_dispositivo(pipeline, device, batch_size=None):
    """Copia del pipeline con los tres modelos en `device` (AMP solo en GPU) y, si se pide, otro tamaño de lote."""
    cfg = {**pipeline["ns"]["CFG_SLICE"], **({"batch_size": batch_size} if batch_size else {})}
    ns = {**pipeline["ns"], "DEVICE": device, "AMP": device.type == "cuda", "CFG_SLICE": cfg}
    modelos = {}
    for plano, comp in pipeline["modelos"].items():
        modelos[plano] = {k: (copy.deepcopy(v).to(device).eval() if isinstance(v, torch.nn.Module) else v)
                          for k, v in comp.items()}
    return {**pipeline, "ns": ns, "modelos": modelos}


def cortes_centrales(volume_hu, n_cortes):
    centro = volume_hu.shape[0] // 2
    return list(range(max(1, centro - n_cortes // 2), min(volume_hu.shape[0] - 1, centro + n_cortes // 2)))


def ms_por_imagen(uno_a_uno, volume_hu, spacing, z_cortes, repeticiones):
    """Un corte axial a la vez (batch 1): ms por imagen en cada repetición y cortes por repetición."""
    sub = volume_hu[z_cortes[0] - 1:z_cortes[-1] + 2]           # con un corte vecino a cada lado (entrada 2.5D)
    inferencia.inferir_plano(uno_a_uno, "axial", sub[:6], spacing)       # calentamiento
    tiempos = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        inferencia.inferir_plano(uno_a_uno, "axial", sub, spacing)
        tiempos.append(1000 * (time.perf_counter() - t0) / sub.shape[0])
    return tiempos, int(sub.shape[0])


def s_por_volumen(completo, volume_hu, spacing):
    """Volumen completo, igual que el frontend: los tres planos con el lote del pipeline, asociación 3D, limpieza y
    distancias."""
    res = {p: inferencia.predecir_plano(completo, p, volume_hu, spacing, case_id="latencia") for p in inferencia.PLANOS}
    return {"s": float(sum(r["segundos"] for r in res.values())), "cortes": int(sum(volume_hu.shape)),
            "s_por_plano": {p: float(r["segundos"]) for p, r in res.items()},
            "fragmentos_por_plano": {p: int(len(r["fragmentos"])) for p, r in res.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--caso", default="035", help="ID del caso en data_mha/*images*/ (por defecto 035, de val)")
    parser.add_argument("--split", choices=["train", "val", "test"],
                        help="mide sobre todos los casos del split en lugar de --caso")
    parser.add_argument("--casos-volumen", type=int, default=3,
                        help="con --split: cuántos casos se procesan completos para la latencia por volumen")
    parser.add_argument("--cortes", type=int, default=40, help="cortes consecutivos con hueso para la latencia")
    parser.add_argument("--repeticiones", type=int, default=3)
    parser.add_argument("--solo-cpu", action="store_true")
    args = parser.parse_args()

    if args.split:
        dataset = pd.read_csv(DATASET_CSV, dtype={"case_id": str})
        casos = sorted(dataset.loc[dataset["split"] == args.split, "case_id"])
        casos_volumen = casos[:args.casos_volumen]
    else:
        casos = casos_volumen = [args.caso]
    rutas = {c: next(DATA_MHA_DIR.glob(f"*images*/{c}.mha"), None) for c in casos}
    faltan = [c for c, r in rutas.items() if r is None]
    if faltan:
        raise SystemExit(f"No encuentro {faltan} en {DATA_MHA_DIR}/*images*/")

    pipeline = inferencia.cargar_pipeline()
    dispositivos = [torch.device("cpu")]
    if torch.cuda.is_available() and not args.solo_cpu:
        dispositivos.insert(0, torch.device("cuda"))
    # (batch 1 para la latencia por imagen, lote del pipeline para el volumen completo)
    pipelines = {d.type: (en_dispositivo(pipeline, d, batch_size=1), en_dispositivo(pipeline, d)) for d in dispositivos}

    por_imagen = {d.type: {} for d in dispositivos}
    por_volumen = {d.type: {} for d in dispositivos}
    imagenes_por_repeticion = []
    for caso in casos:
        volume_hu, spacing = inferencia.leer_canonico(rutas[caso])
        z_cortes = cortes_centrales(volume_hu, args.cortes)
        for device in dispositivos:
            uno_a_uno, completo = pipelines[device.type]
            tiempos, n_img = ms_por_imagen(uno_a_uno, volume_hu, spacing, z_cortes, args.repeticiones)
            por_imagen[device.type][caso] = tiempos
            imagenes_por_repeticion.append(n_img)
            linea = f"  {caso} {device.type:4s} {np.mean(tiempos):6.1f} ms por imagen"
            if caso in casos_volumen:
                v = por_volumen[device.type][caso] = s_por_volumen(completo, volume_hu, spacing)
                linea += f" | {v['s']:.1f} s por volumen ({v['cortes']} cortes en los tres planos)"
            print(linea, flush=True)
        del volume_hu

    resultados = []
    for device in dispositivos:
        medias = {c: float(np.mean(t)) for c, t in por_imagen[device.type].items()}
        vols = por_volumen[device.type]
        r = {"dispositivo": device.type,
             "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else nombre_cpu(),
             "hilos_cpu": torch.get_num_threads() if device.type == "cpu" else None,
             "precision_mixta": device.type == "cuda",
             "ms_por_imagen": float(np.mean(list(medias.values()))),
             # un caso: desviación entre repeticiones; varios casos: entre casos
             "ms_por_imagen_std": float(np.std(list(medias.values())) if len(medias) > 1
                                        else np.std(por_imagen[device.type][casos[0]])),
             "repeticiones": args.repeticiones,
             "imagenes_por_repeticion": int(np.mean(imagenes_por_repeticion)),
             "s_por_volumen": float(np.mean([v["s"] for v in vols.values()])),
             "cortes_volumen": int(np.mean([v["cortes"] for v in vols.values()])),
             "s_por_plano": {p: float(np.mean([v["s_por_plano"][p] for v in vols.values()])) for p in inferencia.PLANOS}}
        if args.split:
            r.update({"casos_imagen": len(medias), "ms_por_imagen_min": min(medias.values()),
                      "ms_por_imagen_max": max(medias.values()), "casos_volumen": list(vols),
                      "ms_por_imagen_por_caso": medias, "s_por_volumen_por_caso": {c: v["s"] for c, v in vols.items()}})
        else:
            r["fragmentos_por_plano"] = vols[casos[0]]["fragmentos_por_plano"]
        print(f"{device.type}: {r['ms_por_imagen']:.1f} ms por imagen, {r['s_por_volumen']:.1f} s por volumen", flush=True)
        resultados.append(r)

    salida = SALIDA_DIR / (f"latencia_{args.split}.json" if args.split else "latencia.json")
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps({"caso": args.split or args.caso, "split": args.split, "casos": casos,
                                  "modelo": ", ".join(r.name for r in inferencia.MODELOS.values()),
                                  "creado": time.strftime("%Y-%m-%d %H:%M"), "resultados": resultados},
                                 indent=2, ensure_ascii=False), encoding="utf-8")
    print("Guardado en", salida.relative_to(ROOT))


if __name__ == "__main__":
    main()
