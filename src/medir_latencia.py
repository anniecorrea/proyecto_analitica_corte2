"""Latencia de inferencia en GPU y CPU (consigna 4.3), con el mismo pipeline del frontend (app/inferencia.py).

Mide dos cosas por dispositivo:
- por imagen: un corte a la vez (batch 1) por preprocesamiento + backbone/CBAM + detección + NMS + decoder;
- por volumen: la tomografía completa como en el frontend (todos los cortes, asociación 3D, limpieza y distancias).

Desde la raíz del repositorio, con los datos en data_mha/:

    python -m src.medir_latencia                 # caso 035 de val, GPU (si hay) y CPU
    python -m src.medir_latencia --caso 037 --cortes 60

Guarda resultados/metricas/latencia/latencia.json, que lee el dashboard.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import numpy as np
import torch

import inferencia
from rutas import DATA_MHA_DIR, METRICAS_DIR

SALIDA = METRICAS_DIR / "latencia" / "latencia.json"


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


def medir_dispositivo(pipeline, device, volume_hu, image_itk, z_cortes, repeticiones):
    ns = pipeline["ns"]
    comp, _ = ns["build_components_from_export"](inferencia.MODELO_PATH, device=device)
    amp = device.type == "cuda"
    cfg = {**ns["CFG_SLICE"], "batch_size": 1}
    sub = volume_hu[z_cortes[0] - 1:z_cortes[-1] + 2]           # con un corte vecino a cada lado (entrada 2.5D)

    with torch.no_grad():
        ns["infer_case"](comp, "calentamiento", cfg=cfg, device=device, amp=amp, volume=sub[:6], image_itk=image_itk,
                         verbose=False)
        tiempos = []
        for _ in range(repeticiones):
            t0 = time.perf_counter()
            ns["infer_case"](comp, "latencia", cfg=cfg, device=device, amp=amp, volume=sub, image_itk=image_itk,
                             verbose=False)
            tiempos.append(1000 * (time.perf_counter() - t0) / sub.shape[0])

    # Volumen completo, igual que el frontend (batch del pipeline, asociación 3D, limpieza y distancias)
    infer_case = ns["infer_case"]
    ns_dispositivo = {**ns, "infer_case": lambda c, cid, **kw: infer_case(c, cid, device=device, amp=amp, **kw)}
    res = inferencia.predecir_volumen({**pipeline, "comp": comp, "ns": ns_dispositivo}, volume_hu, image_itk,
                                      case_id="latencia")

    return {"dispositivo": device.type,
            "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else nombre_cpu(),
            "hilos_cpu": torch.get_num_threads() if device.type == "cpu" else None,
            "precision_mixta": amp,
            "ms_por_imagen": float(np.mean(tiempos)), "ms_por_imagen_std": float(np.std(tiempos)),
            "repeticiones": repeticiones, "imagenes_por_repeticion": int(sub.shape[0]),
            "s_por_volumen": float(res["segundos"]), "cortes_volumen": int(volume_hu.shape[0]),
            "fragmentos_volumen": int(len(res["fragmentos"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--caso", default="035", help="ID del caso en data_mha/*images*/ (por defecto 035, de val)")
    parser.add_argument("--cortes", type=int, default=40, help="cortes consecutivos con hueso para la latencia")
    parser.add_argument("--repeticiones", type=int, default=3)
    parser.add_argument("--solo-cpu", action="store_true")
    args = parser.parse_args()

    ruta = next(DATA_MHA_DIR.glob(f"*images*/{args.caso}.mha"), None)
    if ruta is None:
        raise SystemExit(f"No encuentro {args.caso}.mha en {DATA_MHA_DIR}/*images*/")
    volume_hu, image_itk = inferencia.leer_mha(ruta)
    centro = volume_hu.shape[0] // 2
    z_cortes = list(range(max(1, centro - args.cortes // 2), min(volume_hu.shape[0] - 1, centro + args.cortes // 2)))

    pipeline = inferencia.cargar_pipeline()
    dispositivos = [torch.device("cpu")]
    if torch.cuda.is_available() and not args.solo_cpu:
        dispositivos.insert(0, torch.device("cuda"))

    resultados = []
    for device in dispositivos:
        print(f"Midiendo en {device.type}...", flush=True)
        r = medir_dispositivo(pipeline, device, volume_hu, image_itk, z_cortes, args.repeticiones)
        print(f"  {r['hardware']}: {r['ms_por_imagen']:.1f} ms por imagen, {r['s_por_volumen']:.1f} s por volumen "
              f"({r['cortes_volumen']} cortes)", flush=True)
        resultados.append(r)

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps({"caso": args.caso, "modelo": inferencia.MODELO_PATH.name,
                                  "creado": time.strftime("%Y-%m-%d %H:%M"), "resultados": resultados},
                                 indent=2, ensure_ascii=False), encoding="utf-8")
    print("Guardado en", SALIDA.relative_to(ROOT))


if __name__ == "__main__":
    main()
