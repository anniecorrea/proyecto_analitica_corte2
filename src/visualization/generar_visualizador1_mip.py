from pathlib import Path
import sys

from PIL import Image, ImageDraw
import SimpleITK as sitk
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR / "src"))

from utils.project_paths import DATA_MHA, EVIDENCE


OUTPUT_DIR = EVIDENCE / "entrega1"


def find_image_path(case_id: str) -> Path:
    for folder_name in [
        "PENGWIN_CT_train_images_part1",
        "PENGWIN_CT_train_images_part2",
    ]:
        image_path = DATA_MHA / folder_name / f"{case_id}.mha"
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"No se encontro la imagen del caso {case_id}")


def load_ct_volume(case_id: str):
    image_path = find_image_path(case_id)
    image = sitk.ReadImage(str(image_path))
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing_xyz = image.GetSpacing()
    return image_path, volume, spacing_xyz


def compute_bone_mip(volume: np.ndarray, axis: int, hu_threshold: float = 300):
    bone_only = np.where(volume >= hu_threshold, volume, 0)
    return bone_only.max(axis=axis)


def mip_to_image(mip: np.ndarray, title: str) -> Image.Image:
    high = np.percentile(mip[mip > 0], 99) if np.any(mip > 0) else 1
    normalized = np.clip(mip / high, 0, 1)
    image_array = (normalized * 255).astype(np.uint8)
    image = Image.fromarray(image_array).convert("L")
    image.thumbnail((520, 520))

    canvas = Image.new("RGB", (560, 600), "white")
    canvas.paste(image.convert("RGB"), ((560 - image.width) // 2, 55))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 20), title, fill=(0, 0, 0))
    return canvas


def save_visualizer_1(case_id: str = "001", hu_threshold: float = 300):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_path, volume, spacing_xyz = load_ct_volume(case_id)

    views = {
        "axial": compute_bone_mip(volume, axis=0, hu_threshold=hu_threshold),
        "coronal": compute_bone_mip(volume, axis=1, hu_threshold=hu_threshold),
        "sagital": compute_bone_mip(volume, axis=2, hu_threshold=hu_threshold),
    }

    output_png = OUTPUT_DIR / f"visualizador1_mip_raw_caso_{case_id}.png"
    panels = [mip_to_image(mip, f"MIP {name}") for name, mip in views.items()]
    final = Image.new("RGB", (1680, 680), "white")
    draw = ImageDraw.Draw(final)
    draw.text(
        (20, 20),
        f"Visualizador 1 MIP raw caso {case_id} umbral HU >= {hu_threshold}",
        fill=(0, 0, 0),
    )
    for index, panel in enumerate(panels):
        final.paste(panel, (index * 560, 70))
    final.save(output_png)

    summary_path = OUTPUT_DIR / f"visualizador1_mip_raw_caso_{case_id}_resumen.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"Caso: {case_id}",
                f"Imagen: {image_path}",
                f"Shape array z_y_x: {volume.shape}",
                f"Spacing x_y_z mm: {spacing_xyz}",
                f"Rango HU original: min={float(volume.min())}, max={float(volume.max())}",
                f"Umbral hueso HU: {hu_threshold}",
                f"Salida PNG: {output_png}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Evidencia guardada en: {output_png}")
    print(f"Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    save_visualizer_1(case_id="001", hu_threshold=300)
