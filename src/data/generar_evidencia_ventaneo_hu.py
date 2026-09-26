from pathlib import Path
import os
import sys

import numpy as np
import SimpleITK as sitk


PROJECT_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(PROJECT_DIR / "src"))

from utils.project_paths import DATA_MHA, EVIDENCE


OUTPUT_DIR = EVIDENCE / "entrega1" / "ventaneo_hu"


def to_repo_relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def find_image_path(case_id: str) -> Path:
    for folder_name in [
        "PENGWIN_CT_train_images_part1",
        "PENGWIN_CT_train_images_part2",
    ]:
        image_path = DATA_MHA / folder_name / f"{case_id}.mha"
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"No se encontro la imagen del caso {case_id}")


def load_ct(case_id: str):
    image_path = find_image_path(case_id)
    image = sitk.ReadImage(str(image_path))
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    return image_path, volume, image.GetSpacing()


def apply_hu_window(volume: np.ndarray, center: float = 400, width: float = 1800):
    low = center - width / 2
    high = center + width / 2
    windowed = np.clip(volume, low, high)
    return (windowed - low) / (high - low), low, high


def save_histograms(case_id: str, volume: np.ndarray, windowed: np.ndarray, low: float, high: float):
    sample = volume.ravel()
    if sample.size > 2_000_000:
        rng = np.random.default_rng(42)
        sample = rng.choice(sample, size=2_000_000, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(sample, bins=256, color="#4C78A8", alpha=0.9)
    axes[0].axvline(low, color="#E15759", linestyle="--", label=f"low={low:.0f} HU")
    axes[0].axvline(high, color="#E15759", linestyle="--", label=f"high={high:.0f} HU")
    axes[0].axvline(300, color="#59A14F", linestyle=":", label="umbral hueso 300 HU")
    axes[0].set_title(f"Histograma HU crudo caso {case_id}")
    axes[0].set_xlabel("HU")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_xlim(-1200, 2200)
    axes[0].legend()

    axes[1].hist(windowed.ravel(), bins=128, color="#F28E2B", alpha=0.9)
    axes[1].set_title("Distribucion despues de ventaneo oseo")
    axes[1].set_xlabel("Intensidad normalizada 0-1")
    axes[1].set_ylabel("Frecuencia")

    fig.tight_layout()
    out = OUTPUT_DIR / f"histograma_ventaneo_hu_caso_{case_id}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def save_slice_comparison(case_id: str, volume: np.ndarray, windowed: np.ndarray):
    bone_score = np.count_nonzero(volume >= 300, axis=(1, 2))
    z = int(np.argmax(bone_score))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(volume[z], cmap="gray")
    axes[0].set_title("HU crudo")
    axes[0].axis("off")

    axes[1].imshow(windowed[z], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ventana osea")
    axes[1].axis("off")

    axes[2].imshow(volume[z] >= 300, cmap="gray")
    axes[2].set_title("Umbral HU >= 300")
    axes[2].axis("off")

    fig.suptitle(f"Caso {case_id} corte {z}: comparacion de preprocesamiento")
    fig.tight_layout()
    out = OUTPUT_DIR / f"comparacion_ventaneo_umbral_caso_{case_id}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out, z


def save_summary(case_id: str, image_path: Path, spacing, volume: np.ndarray, low: float, high: float, z: int):
    bone_ratio = float(np.count_nonzero(volume >= 300) / volume.size)
    summary = f"""# Evidencia de ventaneo HU caso {case_id}

- Imagen: `{image_path}`
- Shape z_y_x: {volume.shape}
- Spacing x_y_z mm: {spacing}
- Rango HU original: {float(volume.min()):.1f} a {float(volume.max()):.1f}
- Ventana osea usada: centro 400 HU, ancho 1800 HU
- Limites de ventana: {low:.1f} HU a {high:.1f} HU
- Umbral de hueso exploratorio: HU >= 300
- Porcentaje de voxeles sobre 300 HU: {bone_ratio * 100:.2f}%
- Corte usado para comparacion visual: {z}

Esta evidencia permite justificar el preprocesamiento con histograma y no solo con inspeccion visual.
"""
    summary = summary.replace(str(image_path), to_repo_relative(image_path))
    out = OUTPUT_DIR / f"resumen_ventaneo_hu_caso_{case_id}.md"
    out.write_text(summary, encoding="utf-8")
    return out


def main(case_id: str = "001"):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_path, volume, spacing = load_ct(case_id)
    windowed, low, high = apply_hu_window(volume)
    hist_path = save_histograms(case_id, volume, windowed, low, high)
    comparison_path, z = save_slice_comparison(case_id, volume, windowed)
    summary_path = save_summary(case_id, image_path, spacing, volume, low, high, z)

    print("Histograma guardado en:", hist_path)
    print("Comparacion guardada en:", comparison_path)
    print("Resumen guardado en:", summary_path)


if __name__ == "__main__":
    main(case_id="001")
