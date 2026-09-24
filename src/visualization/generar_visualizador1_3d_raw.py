from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
import SimpleITK as sitk


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


def build_bone_point_cloud(
    volume: np.ndarray,
    spacing_xyz,
    hu_threshold: float = 300,
    stride: int = 4,
    max_points: int = 120_000,
    seed: int = 42,
):
    sampled = volume[::stride, ::stride, ::stride]
    zyx = np.argwhere(sampled >= hu_threshold)

    if len(zyx) > max_points:
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(zyx), size=max_points, replace=False)
        zyx = zyx[selected]

    z = zyx[:, 0] * stride * spacing_xyz[2]
    y = zyx[:, 1] * stride * spacing_xyz[1]
    x = zyx[:, 2] * stride * spacing_xyz[0]
    intensities = sampled[zyx[:, 0], zyx[:, 1], zyx[:, 2]]
    return x, y, z, intensities


def save_visualizer_3d(
    case_id: str = "001",
    hu_threshold: float = 300,
    stride: int = 4,
    max_points: int = 120_000,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_path, volume, spacing_xyz = load_ct_volume(case_id)
    x, y, z, intensities = build_bone_point_cloud(
        volume=volume,
        spacing_xyz=spacing_xyz,
        hu_threshold=hu_threshold,
        stride=stride,
        max_points=max_points,
    )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker={
                    "size": 1.6,
                    "color": intensities,
                    "colorscale": "Gray",
                    "opacity": 0.55,
                    "showscale": True,
                    "colorbar": {"title": "HU"},
                },
                name="Hueso umbralizado",
            )
        ]
    )
    fig.update_layout(
        title=(
            f"Visualizador 1 3D raw caso {case_id} "
            f"umbral HU >= {hu_threshold}"
        ),
        scene={
            "xaxis_title": "x mm",
            "yaxis_title": "y mm",
            "zaxis_title": "z mm",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )

    output_html = OUTPUT_DIR / f"visualizador1_3d_raw_caso_{case_id}.html"
    fig.write_html(output_html, include_plotlyjs="cdn")

    summary_path = OUTPUT_DIR / f"visualizador1_3d_raw_caso_{case_id}_resumen.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"Caso: {case_id}",
                f"Imagen: {image_path}",
                f"Shape array z_y_x: {volume.shape}",
                f"Spacing x_y_z mm: {spacing_xyz}",
                f"Rango HU original: min={float(volume.min())}, max={float(volume.max())}",
                f"Umbral hueso HU: {hu_threshold}",
                f"Stride de muestreo: {stride}",
                f"Puntos 3D renderizados: {len(x)}",
                f"Maximo de puntos permitido: {max_points}",
                f"Salida HTML: {output_html}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Visualizador 3D guardado en: {output_html}")
    print(f"Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    save_visualizer_3d(case_id="001", hu_threshold=300, stride=4)
