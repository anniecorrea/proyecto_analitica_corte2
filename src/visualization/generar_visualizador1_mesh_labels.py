from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
import SimpleITK as sitk
from skimage import measure


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR / "src"))

from utils.project_paths import EVIDENCE, LABELS


OUTPUT_DIR = EVIDENCE / "entrega1"


REGIONS = {
    "sacro": {"labels": range(1, 11), "color": "#D95F5F"},
    "coxal_izquierdo": {"labels": range(11, 21), "color": "#4C78A8"},
    "coxal_derecho": {"labels": range(21, 31), "color": "#59A14F"},
}


def to_repo_relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def label_path_for_case(case_id: str) -> Path:
    path = LABELS / f"{case_id}.mha"
    if not path.exists():
        raise FileNotFoundError(f"No se encontro label para el caso {case_id}")
    return path


def load_label(case_id: str):
    path = label_path_for_case(case_id)
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    return path, array, image.GetSpacing()


def crop_to_nonzero(mask: np.ndarray, margin: int = 8):
    zyx = np.argwhere(mask)
    if len(zyx) == 0:
        return None
    lower = np.maximum(zyx.min(axis=0) - margin, 0)
    upper = np.minimum(zyx.max(axis=0) + margin + 1, mask.shape)
    z0, y0, x0 = lower
    z1, y1, x1 = upper
    return mask[z0:z1, y0:y1, x0:x1], lower, upper


def mesh_from_mask(mask: np.ndarray, spacing_xyz, origin_zyx, stride: int = 1):
    sampled = mask[::stride, ::stride, ::stride]
    if not np.any(sampled):
        return None

    spacing_zyx = (
        spacing_xyz[2] * stride,
        spacing_xyz[1] * stride,
        spacing_xyz[0] * stride,
    )
    verts_zyx, faces, _, _ = measure.marching_cubes(
        sampled.astype(np.uint8),
        level=0.5,
        spacing=spacing_zyx,
    )
    z = verts_zyx[:, 0] + origin_zyx[0] * spacing_xyz[2]
    y = verts_zyx[:, 1] + origin_zyx[1] * spacing_xyz[1]
    x = verts_zyx[:, 2] + origin_zyx[2] * spacing_xyz[0]
    return x, y, z, faces


def save_label_mesh_visualizer(case_id: str = "001", stride: int = 1):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    label_path, label_array, spacing_xyz = load_label(case_id)
    traces = []
    summary_lines = [
        f"Caso: {case_id}",
        f"Label: {to_repo_relative(label_path)}",
        f"Shape label z_y_x: {label_array.shape}",
        f"Spacing x_y_z mm: {spacing_xyz}",
        f"Stride malla: {stride}",
    ]

    for region_name, config in REGIONS.items():
        region_mask = np.isin(label_array, list(config["labels"]))
        cropped = crop_to_nonzero(region_mask, margin=8)
        if cropped is None:
            continue

        cropped_mask, lower, upper = cropped
        result = mesh_from_mask(
            cropped_mask,
            spacing_xyz=spacing_xyz,
            origin_zyx=lower,
            stride=stride,
        )
        if result is None:
            continue

        x, y, z, faces = result
        traces.append(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=faces[:, 2],
                j=faces[:, 1],
                k=faces[:, 0],
                color=config["color"],
                opacity=0.92,
                flatshading=False,
                lighting={
                    "ambient": 0.42,
                    "diffuse": 0.82,
                    "roughness": 0.5,
                    "specular": 0.18,
                },
                name=region_name.replace("_", " "),
            )
        )
        summary_lines.extend(
            [
                f"",
                f"Region: {region_name}",
                f"ROI z_y_x desde {lower.tolist()} hasta {upper.tolist()}",
                f"Vertices: {len(x)}",
                f"Caras: {len(faces)}",
            ]
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Visualizador mesh labels caso {case_id}: sacro y coxales",
        scene={
            "xaxis_title": "x mm",
            "yaxis_title": "y mm",
            "zaxis_title": "z mm",
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.45, "y": 1.4, "z": 0.85}},
        },
        legend={"orientation": "h", "y": 0.02},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        dragmode="orbit",
    )

    output_html = OUTPUT_DIR / f"visualizador1_mesh_labels_caso_{case_id}.html"
    fig.write_html(
        output_html,
        include_plotlyjs=True,
        config={"scrollZoom": True, "displaylogo": False, "responsive": True},
    )

    summary_lines.append(f"Salida HTML: {to_repo_relative(output_html)}")
    summary_path = OUTPUT_DIR / f"visualizador1_mesh_labels_caso_{case_id}_resumen.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Visualizador mesh labels guardado en: {output_html}")
    print(f"Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    save_label_mesh_visualizer(case_id="001", stride=1)
