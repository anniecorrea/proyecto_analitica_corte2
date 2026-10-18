"""entrenar_pertenencia_planos.py -- Etapa D (cabeza de pertenencia) sobre los modelos por plano.

USO (desde la raiz del repositorio)
  python -m src.entrenar_pertenencia_planos --plano todos --device cuda
  python -m src.entrenar_pertenencia_planos --plano sagital --device cuda --max-cortes-train 5000
Despues: python -m src.evaluacion --plano todos --split val --device cuda
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import evaluacion as ev  # noqa: E402   (reutiliza la carga de modelos y datos por plano)
from src.pertenencia import PertenenciaHead, entrenar_pertenencia, precalcular_cache_pertenencia  # noqa: E402
from src.rutas import METRICAS_DIR  # noqa: E402

HISTORIAL_DIR = METRICAS_DIR / "nms_pertenencia"


class _Pares(torch.utils.data.Dataset):
    """precalcular_cache_pertenencia espera dataset[i] -> (imagen, target); PlaneSliceDataset da un dict."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        return item["image"], item


def _submuestreo(n: int, maximo: int | None) -> list[int]:
    """Indices repartidos de forma uniforme (todos los casos quedan representados, no solo los primeros)."""
    if not maximo or maximo >= n:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, maximo).astype(int).tolist()))


def _cache_completa_existe(plano: str, split: str, man: pd.DataFrame) -> bool:
    """Replica la clave de construir_o_cargar_cache: True si la cache completa ya esta en disco y es valida."""
    c = ev._COMPONENTS
    man = c["seleccionar"](man, None)
    carpeta = Path(c["CACHE_DIR"]) / c["_etiqueta_cache"](plano, split, "pos", None)
    clave = {**c["DATA_VERSION"], "kind": "pos", "split": split, "max_slices": None,
             "cortes": hashlib.sha1(man[["case_id", "slice_index"]].to_csv(index=False).encode()).hexdigest()[:12]}
    ruta = carpeta / "key.json"
    return ruta.is_file() and json.loads(ruta.read_text()) == clave


def _dataset_del_split(plano: str, split: str, positivos: pd.DataFrame, config: dict, maximo: int | None):
    """Dataset de cortes positivos del split y los indices a usar (submuestreo uniforme si hace falta)."""
    c = ev._COMPONENTS
    man = positivos[positivos["split"] == split].sort_values(["case_id", "slice_index"]).reset_index(drop=True)
    if maximo and maximo < len(man) and not _cache_completa_existe(plano, split, man):
        man = man.iloc[_submuestreo(len(man), maximo)].reset_index(drop=True)
        print(f"[{plano}/{split}] no hay cache completa: se construye una con {len(man)} cortes")
    cache = c["construir_o_cargar_cache"](plano, split, "pos", man)
    dataset = c["PlaneSliceDataset"](
        cache, bool(config.get("USE_POSITION_CHANNEL", False)), train=False,
        val_mode=config.get("POS_VAL_MODE", "fov"), center_mode=config.get("CENTER_MODE", "interior"),
    )
    dataset.need_seg = False
    return dataset, _submuestreo(len(dataset), maximo)


def entrenar_plano(plano: str, args: argparse.Namespace) -> dict:
    # --checkpoint / --salida: otro tronco del mismo plano (modelo de la etapa B o de la ablacion)
    checkpoint = getattr(args, "checkpoint", None) or ev.checkpoint_del_plano(plano)
    salida = getattr(args, "salida", None) or ev.cabeza_del_plano(plano)
    if not checkpoint.is_file() or checkpoint.stat().st_size < 10_000:
        raise FileNotFoundError(f"Falta {checkpoint} (o es un puntero de Git LFS: correr `git lfs pull`).")
    tronco_id = hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:16]
    meta_path = salida.with_name(f"{salida.stem}_meta.json")
    if salida.is_file() and meta_path.is_file() and not args.reentrenar:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("tronco_sha256_16") == tronco_id:
            print(f"[{plano}] ya existe {salida.name} para este tronco: se omite (usar --reentrenar para repetir).")
            return meta

    print(f"\n===== Etapa D | plano {plano} | {checkpoint.name} | {ev.DEVICE} =====", flush=True)
    t0 = time.time()
    ev._load_project_components(plane_mode=True)
    info = ev._load_plane_components(plano, torch.load(checkpoint, map_location="cpu", weights_only=True))
    config = info["config"]
    model, _ = ev._load_model(checkpoint, ev.DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)          # tronco congelado: solo se entrena la cabeza
    model.eval()

    def tronco(x):
        return model.trunk(x)[-1]        # backbone + CBAM -> f4, el mismo mapa que ve la cabeza en inferencia

    positivos, _ = ev._COMPONENTS["leer_manifiestos"](plano)
    ds_train, idx_train = _dataset_del_split(plano, "train", positivos, config, args.max_cortes_train)
    ds_val, idx_val = _dataset_del_split(plano, "val", positivos, config, args.max_cortes_val)
    print(f"[{plano}] precalculando f4: {len(idx_train)} cortes de train y {len(idx_val)} de val")
    cache_train = precalcular_cache_pertenencia(tronco, _Pares(ds_train), idx_train, batch_size=args.batch_size,
                                                device=ev.DEVICE)
    cache_val = precalcular_cache_pertenencia(tronco, _Pares(ds_val), idx_val, batch_size=args.batch_size,
                                              device=ev.DEVICE)
    del model, ds_train, ds_val
    gc.collect()
    por_clase = torch.bincount(cache_train["label"], minlength=3).tolist()
    print(f"[{plano}] f4 listo en {time.time() - t0:.0f} s | instancias train {cache_train['label'].shape[0]} "
          f"(sacro, coxal izq., coxal der. = {por_clase}) | val {cache_val['label'].shape[0]}")

    torch.manual_seed(args.seed)
    head = PertenenciaHead()
    head, historial, mejor = entrenar_pertenencia(head, cache_train, cache_val, epochs=args.epochs, lr=args.lr,
                                                  batch_inst=256, paciencia=args.paciencia, seed=args.seed,
                                                  device=ev.DEVICE)
    salida.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in head.state_dict().items()}, salida)
    meta = {
        "plano": plano, "tronco": checkpoint.name, "tronco_sha256_16": tronco_id,
        "mejor_epoca": int(mejor["epoch"]), "val_f1_macro": float(mejor["val_f1_macro"]),
        "val_auc_macro_ovr": float(mejor["val_auc_macro_ovr"]),
        "nota_val": "F1/AUC con mascaras REALES (teacher forcing); el F1 del sistema completo lo da src/evaluacion.py",
        "cortes_train": len(idx_train), "cortes_val": len(idx_val),
        "instancias_train": int(cache_train["label"].shape[0]), "instancias_val": int(cache_val["label"].shape[0]),
        "linea_media_sagital": config.get("POS_VAL_MODE") if config.get("USE_POSITION_CHANNEL") else None,
        "epochs_max": args.epochs, "lr": args.lr, "paciencia": args.paciencia, "seed": args.seed,
        "device": str(ev.DEVICE), "minutos": round((time.time() - t0) / 60, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    HISTORIAL_DIR.mkdir(parents=True, exist_ok=True)
    sufijo = salida.stem.removeprefix("cabeza_pertenencia_")      # axial, axial_B, axial_ablacion_sin_cbam...
    historial.to_csv(HISTORIAL_DIR / f"historial_etapa_D_{sufijo}.csv", index=False)
    print(f"[{plano}] mejor epoca {meta['mejor_epoca']}: F1 val {meta['val_f1_macro']:.3f} | "
          f"AUC val {meta['val_auc_macro_ovr']:.3f} | {salida.name} guardada ({meta['minutos']} min)")
    del cache_train, cache_val, head
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plano", choices=(*ev.PLANOS, "todos"), default="todos")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-cortes-train", type=int, default=3000,
                        help="Cortes de train (submuestreo uniforme). 0 = todos (necesita mucha RAM: ~130 KB por corte).")
    parser.add_argument("--max-cortes-val", type=int, default=0, help="Cortes de val; 0 = todos.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--paciencia", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16, help="Cortes por lote al precalcular f4.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reentrenar", action="store_true", help="Repite aunque ya exista la cabeza de ese tronco.")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Solo con un plano: tronco distinto del modelo final (p. ej. el de la etapa B o de la ablación).")
    parser.add_argument("--salida", type=Path, default=None,
                        help="Solo con un plano: archivo de la cabeza (por defecto cabeza_pertenencia_{plano}.pt).")
    args = parser.parse_args()
    if (args.checkpoint or args.salida) and args.plano == "todos":
        parser.error("--checkpoint y --salida solo se pueden usar con un plano.")
    args.checkpoint = None if args.checkpoint is None else (args.checkpoint if args.checkpoint.is_absolute()
                                                            else ROOT / args.checkpoint)
    args.salida = None if args.salida is None else (args.salida if args.salida.is_absolute() else ROOT / args.salida)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("Se pidio --device cuda pero PyTorch no ve ninguna GPU.")
    return args


def main() -> None:
    args = parse_args()
    ev.DEVICE = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                             ("cpu" if args.device == "auto" else args.device))
    ev._COMPONENTS["DEVICE"] = ev.DEVICE
    planos = list(ev.PLANOS) if args.plano == "todos" else [args.plano]
    resumen = []
    for plano in planos:
        meta = entrenar_plano(plano, args)
        resumen.append({"plano": plano, "val_f1_macro_mascaras_reales": meta.get("val_f1_macro"),
                        "val_auc_macro_ovr": meta.get("val_auc_macro_ovr"), "mejor_epoca": meta.get("mejor_epoca")})
    print("\nResumen etapa D:")
    print(pd.DataFrame(resumen).round(3).to_string(index=False))
    print("\nSiguiente: python -m src.evaluacion --plano todos --split val --device cuda")


if __name__ == "__main__":
    main()
