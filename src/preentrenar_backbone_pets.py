"""Transfer learning del backbone (consigna 4.1): FundidoraPC "plus" preentrenada en Oxford-IIIT Pet.

Por qué así
- La consigna solo permite transfer learning en el backbone y con un dataset natural ("ImageNet u otro dataset natural",
  "dataset previamente usados"). FundidoraPC es propia, así que no hay pesos de ImageNet: se preentrena aquí, como
  clasificador de las 37 razas de Oxford-IIIT Pet (dataset ya usado en el curso).
- Lo que se transfiere son las capas que detectan bordes y texturas. El clasificador de razas se descarta; detección,
  decoder y pertenencia se entrenan desde cero sobre PENGWIN.
- Las fotos se pasan a escala de grises y se repiten en 3 canales: la entrada 2.5D del modelo (cortes z-1, z, z+1) es
  gris y de valores parecidos entre canales, sin color. Se escala a [0, 1], como las imágenes ventaneadas.
- Mismo constructor que el entrenamiento (build_backbone de Backbone.ipynb), así las claves del state_dict coinciden.

Desde la raíz del repositorio:

    python -m src.preentrenar_backbone_pets --datos C:/Users/<usuario>/Downloads/data/oxford-iiit-pet

Guarda resultados/modelos/backbone_pets/fundidora_plus_pets.pt (solo el state_dict del backbone, la mejor época por
exactitud en el split test oficial de Pets, que aquí hace de validación: no tiene nada que ver con el test de PENGWIN),
su _meta.json y resultados/metricas/backbone_pets/historial.csv.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.definiciones_notebooks import cargar_definiciones  # noqa: E402
from src.rutas import CACHE_DIR, METRICAS_DIR, MODELOS_DIR, NOTEBOOKS_DIR  # noqa: E402

PESOS = MODELOS_DIR / "backbone_pets" / "fundidora_plus_pets.pt"
HISTORIAL = METRICAS_DIR / "backbone_pets" / "historial.csv"
DATOS_POR_DEFECTO = Path.home() / "Downloads" / "data" / "oxford-iiit-pet"
LADO_CACHE = 288            # se guarda a 288 y se recorta a 256 (recorte aleatorio en train, central en validación)
LADO = 256                  # la misma resolución que los cortes de PENGWIN


def construir_backbone():
    ns = {"torch": torch, "nn": nn, "F": F}             # Backbone.ipynb los importa en otra celda
    cargar_definiciones(NOTEBOOKS_DIR / "Backbone.ipynb", {"FundidoraPC", "FundidoraPCPlus", "build_backbone"},
                        destino=ns)
    return ns["build_backbone"]("plus", in_channels=3)


class Clasificador(nn.Module):
    """backbone._encode (f4: 256 x 16 x 16) -> promedio global -> lineal. El mismo f4 que usan las cabezas de PENGWIN."""

    def __init__(self, backbone, n_clases):
        super().__init__()
        self.backbone = backbone
        self.salida = nn.Sequential(nn.Dropout(0.2), nn.Linear(256, n_clases))

    def forward(self, x):
        return self.salida(self.backbone._encode(x).mean(dim=(2, 3)))


def leer_lista(datos, nombre):
    """annotations/{trainval,test}.txt: 'Abyssinian_100 1 1 1' -> (imagen, clase 0..36)."""
    filas = []
    for linea in (datos / "annotations" / nombre).read_text().splitlines():
        if linea and not linea.startswith("#"):
            partes = linea.split()
            filas.append((partes[0], int(partes[1]) - 1))
    return filas


def cache_gris(datos, nombre):
    """Imágenes en gris, lado menor a LADO_CACHE y recorte central cuadrado: uint8 (N, 288, 288). Se calcula una vez."""
    destino = CACHE_DIR / "pets" / f"pets_gris_{LADO_CACHE}_{Path(nombre).stem}.npz"
    if destino.is_file():
        z = np.load(destino)
        return z["imagenes"], z["clases"]
    filas = leer_lista(datos, nombre)
    imagenes = np.zeros((len(filas), LADO_CACHE, LADO_CACHE), dtype=np.uint8)
    clases = np.array([c for _, c in filas], dtype=np.int64)
    t0 = time.time()
    for i, (img, _) in enumerate(filas):
        with Image.open(datos / "images" / f"{img}.jpg") as im:
            im = im.convert("L")
            escala = LADO_CACHE / min(im.size)
            im = im.resize((max(LADO_CACHE, round(im.width * escala)), max(LADO_CACHE, round(im.height * escala))),
                           Image.BILINEAR)
            x0, y0 = (im.width - LADO_CACHE) // 2, (im.height - LADO_CACHE) // 2
            imagenes[i] = np.asarray(im.crop((x0, y0, x0 + LADO_CACHE, y0 + LADO_CACHE)))
    destino.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destino, imagenes=imagenes, clases=clases)
    print(f"caché {destino.name}: {len(filas)} imágenes en {time.time() - t0:.0f} s", flush=True)
    return imagenes, clases


def lote(imagenes, idx, rng, entrenamiento, device):
    """Recorte 256 (aleatorio + espejo + brillo/contraste en train; central en validación) -> (B, 3, 256, 256) en [0, 1]."""
    m = LADO_CACHE - LADO
    if entrenamiento:
        oy, ox = rng.integers(0, m + 1, len(idx)), rng.integers(0, m + 1, len(idx))
    else:
        oy = ox = np.full(len(idx), m // 2)
    x = np.stack([imagenes[i, a:a + LADO, b:b + LADO] for i, a, b in zip(idx, oy, ox)])
    x = torch.from_numpy(x).to(device, non_blocking=True).float().div_(255.0).unsqueeze(1)
    if entrenamiento:
        espejo = torch.from_numpy(rng.random(len(idx)) < 0.5).to(device)
        x = torch.where(espejo[:, None, None, None], x.flip(-1), x)
        c = torch.from_numpy(rng.uniform(0.8, 1.2, (len(idx), 1, 1, 1))).to(device, torch.float32)
        b = torch.from_numpy(rng.uniform(-0.1, 0.1, (len(idx), 1, 1, 1))).to(device, torch.float32)
        x = ((x - 0.5) * c + 0.5 + b).clamp_(0, 1)
    return x.expand(-1, 3, -1, -1).contiguous()


@torch.no_grad()
def evaluar(modelo, imagenes, clases, device, batch, amp):
    modelo.eval()
    aciertos, perdida = 0, 0.0
    rng = np.random.default_rng(0)
    for i in range(0, len(clases), batch):
        idx = np.arange(i, min(i + batch, len(clases)))
        y = torch.from_numpy(clases[idx]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = modelo(lote(imagenes, idx, rng, False, device))
        perdida += float(F.cross_entropy(logits.float(), y, reduction="sum"))
        aciertos += int((logits.argmax(1) == y).sum())
    return aciertos / len(clases), perdida / len(clases)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datos", type=Path, default=DATOS_POR_DEFECTO,
                        help="carpeta oxford-iiit-pet con images/ y annotations/")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--rehacer", action="store_true", help="repite aunque ya existan los pesos")
    args = parser.parse_args()

    if PESOS.is_file() and not args.rehacer:
        print(f"Ya existe {PESOS.relative_to(ROOT)}: se omite (--rehacer para repetir).")
        return
    if not (args.datos / "annotations" / "trainval.txt").is_file():
        raise SystemExit(f"No encuentro Oxford-IIIT Pet en {args.datos} (falta annotations/trainval.txt).")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          ("cpu" if args.device == "auto" else args.device))
    amp = device.type == "cuda"
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    x_tr, y_tr = cache_gris(args.datos, "trainval.txt")
    x_va, y_va = cache_gris(args.datos, "test.txt")
    n_clases = int(y_tr.max()) + 1
    print(f"Oxford-IIIT Pet: {len(y_tr)} de entrenamiento, {len(y_va)} de validación, {n_clases} clases | {device}",
          flush=True)

    modelo = Clasificador(construir_backbone(), n_clases).to(device)
    pasos = math.ceil(len(y_tr) / args.batch_size)
    opt = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * pasos, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    rng = np.random.default_rng(args.seed)
    filas, mejor, t0 = [], -1.0, time.time()
    PESOS.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        modelo.train()
        orden, suma, n = rng.permutation(len(y_tr)), 0.0, 0
        for i in range(0, len(orden), args.batch_size):
            idx = orden[i:i + args.batch_size]
            y = torch.from_numpy(y_tr[idx]).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                perdida = F.cross_entropy(modelo(lote(x_tr, idx, rng, True, device)).float(), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(perdida).backward()
            scaler.step(opt); scaler.update(); sched.step()
            suma += float(perdida.detach()) * len(idx); n += len(idx)
        acc, perdida_va = evaluar(modelo, x_va, y_va, device, 128, amp)
        filas.append({"epoca": ep, "perdida_train": suma / n, "perdida_val": perdida_va, "exactitud_val": acc})
        es_mejor = acc > mejor
        if es_mejor:
            mejor = acc
            torch.save({k: v.cpu() for k, v in modelo.backbone.state_dict().items()}, PESOS)
        print(f"época {ep:02d}/{args.epochs} | pérdida {suma / n:.3f} | val {perdida_va:.3f} | exactitud {acc:.3f}"
              + (" *" if es_mejor else ""), flush=True)

    HISTORIAL.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(HISTORIAL, index=False)
    meta = {"backbone": "FundidoraPCPlus (build_backbone('plus', in_channels=3))", "dataset": "Oxford-IIIT Pet",
            "tarea": f"clasificación de {n_clases} razas", "imagenes_train": int(len(y_tr)),
            "imagenes_val": int(len(y_va)), "entrada": "gris repetido en 3 canales, [0, 1], recorte 256 de 288",
            "mejor_exactitud_val": mejor, "mejor_epoca": int(max(filas, key=lambda f: f["exactitud_val"])["epoca"]),
            "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "seed": args.seed,
            "minutos": round((time.time() - t0) / 60, 1), "nota": "solo se guarda el backbone; el clasificador se descarta"}
    PESOS.with_name(f"{PESOS.stem}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Listo: exactitud {mejor:.3f} en {meta['minutos']} min -> {PESOS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
