from pathlib import Path
import sys

import numpy as np
import plotly.graph_objects as go
from scipy import ndimage as ndi
import SimpleITK as sitk
from skimage import measure


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR / "src"))

from utils.project_paths import DATA_MHA, EVIDENCE
from utils.project_paths import LABELS


OUTPUT_DIR = EVIDENCE / "entrega1"


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


def find_label_path(case_id: str) -> Path:
    label_path = LABELS / f"{case_id}.mha"
    if label_path.exists():
        return label_path
    raise FileNotFoundError(f"No se encontro el label del caso {case_id}")


def load_ct_volume(case_id: str):
    image_path = find_image_path(case_id)
    image = sitk.ReadImage(str(image_path))
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing_xyz = image.GetSpacing()
    return image_path, volume, spacing_xyz


def load_label_volume(case_id: str):
    label_path = find_label_path(case_id)
    label = sitk.ReadImage(str(label_path))
    label_array = sitk.GetArrayFromImage(label)
    return label_path, label_array


def crop_to_label_roi(volume: np.ndarray, label_array: np.ndarray, margin: int = 18):
    zyx = np.argwhere(label_array > 0)
    if len(zyx) == 0:
        raise SystemExit("El label no tiene voxeles positivos para construir ROI.")

    lower = np.maximum(zyx.min(axis=0) - margin, 0)
    upper = np.minimum(zyx.max(axis=0) + margin + 1, volume.shape)
    z0, y0, x0 = lower
    z1, y1, x1 = upper
    return volume[z0:z1, y0:y1, x0:x1], lower, upper


def build_bone_mask_from_ct(
    volume: np.ndarray,
    hu_threshold: float = 300,
    stride: int = 2,
    min_component_voxels: int = 250,
) -> np.ndarray:
    sampled = volume[::stride, ::stride, ::stride]
    mask = sampled >= hu_threshold

    structure = np.ones((3, 3, 3), dtype=bool)
    mask = ndi.binary_closing(mask, structure=structure, iterations=1)
    mask = ndi.binary_fill_holes(mask)

    labels, n_labels = ndi.label(mask)
    if n_labels > 0:
        counts = np.bincount(labels.ravel())
        keep = counts >= min_component_voxels
        keep[0] = False
        mask = keep[labels]

    return mask


def marching_cubes_mesh(mask: np.ndarray, spacing_xyz, origin_zyx, stride: int = 2):
    # El array esta en orden z, y, x; spacing debe corresponder a z, y, x.
    spacing_zyx = (
        spacing_xyz[2] * stride,
        spacing_xyz[1] * stride,
        spacing_xyz[0] * stride,
    )
    verts_zyx, faces, _, _ = measure.marching_cubes(
        mask.astype(np.uint8),
        level=0.5,
        spacing=spacing_zyx,
    )

    z = verts_zyx[:, 0] + origin_zyx[0] * spacing_xyz[2]
    y = verts_zyx[:, 1] + origin_zyx[1] * spacing_xyz[1]
    x = verts_zyx[:, 2] + origin_zyx[2] * spacing_xyz[0]
    return x, y, z, faces


def decimate_faces(faces: np.ndarray, max_faces: int = 80_000, seed: int = 42) -> np.ndarray:
    if len(faces) <= max_faces:
        return faces
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(faces), size=max_faces, replace=False)
    return faces[np.sort(selected)]


def save_mesh_visualizer(
    case_id: str = "001",
    hu_threshold: float = 300,
    stride: int = 2,
    max_faces: int = 80_000,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_path, volume, spacing_xyz = load_ct_volume(case_id)
    label_path, label_array = load_label_volume(case_id)
    cropped_volume, crop_lower_zyx, crop_upper_zyx = crop_to_label_roi(
        volume=volume,
        label_array=label_array,
        margin=18,
    )
    mask = build_bone_mask_from_ct(
        cropped_volume,
        hu_threshold=hu_threshold,
        stride=stride,
    )

    if not np.any(mask):
        raise SystemExit("La mascara quedo vacia; revisa el umbral HU.")

    x, y, z, faces = marching_cubes_mesh(
        mask,
        spacing_xyz=spacing_xyz,
        origin_zyx=crop_lower_zyx,
        stride=stride,
    )
    faces = decimate_faces(faces, max_faces=max_faces)

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=faces[:, 2],
                j=faces[:, 1],
                k=faces[:, 0],
                color="#F0E5D8",
                opacity=0.96,
                flatshading=False,
                lighting={
                    "ambient": 0.45,
                    "diffuse": 0.8,
                    "roughness": 0.55,
                    "specular": 0.18,
                },
                lightposition={"x": 100, "y": 200, "z": 300},
                name="Malla de hueso en ROI pelvis",
            )
        ]
    )
    fig.update_layout(
        title=(
            f"Visualizador 1 mesh raw caso {case_id} "
            f"HU >= {hu_threshold}, ROI pelvis, stride={stride}"
        ),
        scene={
            "xaxis_title": "x mm",
            "yaxis_title": "y mm",
            "zaxis_title": "z mm",
            "aspectmode": "data",
            "camera": {
                "eye": {"x": 1.6, "y": 1.4, "z": 0.9},
                "center": {"x": 0, "y": 0, "z": 0},
            },
        },
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        dragmode="orbit",
    )

    output_html = OUTPUT_DIR / f"visualizador1_mesh_raw_caso_{case_id}.html"
    fig.write_html(
        output_html,
        include_plotlyjs=True,
        config={"scrollZoom": True, "displaylogo": False, "responsive": True},
    )

    summary_path = OUTPUT_DIR / f"visualizador1_mesh_raw_caso_{case_id}_resumen.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"Caso: {case_id}",
                f"Imagen: {to_repo_relative(image_path)}",
                f"Label usado para ROI: {to_repo_relative(label_path)}",
                f"Shape array z_y_x: {volume.shape}",
                f"ROI z_y_x desde {crop_lower_zyx.tolist()} hasta {crop_upper_zyx.tolist()}",
                f"Shape ROI z_y_x: {cropped_volume.shape}",
                f"Spacing x_y_z mm: {spacing_xyz}",
                f"Rango HU original: min={float(volume.min())}, max={float(volume.max())}",
                f"Umbral hueso HU: {hu_threshold}",
                f"Stride malla: {stride}",
                f"Voxeles positivos tras limpieza: {int(np.count_nonzero(mask))}",
                f"Vertices de malla: {len(x)}",
                f"Caras renderizadas: {len(faces)}",
                f"Maximo de caras permitido: {max_faces}",
                f"Salida HTML: {to_repo_relative(output_html)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Visualizador mesh guardado en: {output_html}")
    print(f"Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    save_mesh_visualizer(case_id="001", hu_threshold=300, stride=2)
