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


def load_ct_volume(case_id: str):
    image_path = find_image_path(case_id)
    image = sitk.ReadImage(str(image_path))
    volume = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing_xyz = image.GetSpacing()
    return image_path, volume, spacing_xyz


def keep_largest_components(mask: np.ndarray, n_components: int = 3) -> np.ndarray:
    labels, n_labels = ndi.label(mask)
    if n_labels == 0:
        return mask

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    selected = np.argsort(counts)[-n_components:]
    return np.isin(labels, selected)


def build_bone_mask(
    volume: np.ndarray,
    hu_threshold: float = 300,
    stride: int = 2,
    min_component_voxels: int = 2_000,
) -> np.ndarray:
    sampled = volume[::stride, ::stride, ::stride]
    mask = sampled >= hu_threshold

    structure = np.ones((3, 3, 3), dtype=bool)
    mask = ndi.binary_opening(mask, structure=structure, iterations=1)
    mask = ndi.binary_closing(mask, structure=structure, iterations=1)
    mask = ndi.binary_fill_holes(mask)

    labels, n_labels = ndi.label(mask)
    if n_labels > 0:
        counts = np.bincount(labels.ravel())
        keep = counts >= min_component_voxels
        keep[0] = False
        mask = keep[labels]
        mask = keep_largest_components(mask, n_components=3)

    return mask


def marching_cubes_mesh(mask: np.ndarray, spacing_xyz, stride: int = 2):
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

    z = verts_zyx[:, 0]
    y = verts_zyx[:, 1]
    x = verts_zyx[:, 2]
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
    mask = build_bone_mask(volume, hu_threshold=hu_threshold, stride=stride)

    if not np.any(mask):
        raise SystemExit("La mascara quedo vacia; revisa el umbral HU.")

    x, y, z, faces = marching_cubes_mesh(mask, spacing_xyz=spacing_xyz, stride=stride)
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
                color="#D9D0C7",
                opacity=0.88,
                flatshading=False,
                lighting={
                    "ambient": 0.45,
                    "diffuse": 0.8,
                    "roughness": 0.55,
                    "specular": 0.18,
                },
                lightposition={"x": 100, "y": 200, "z": 300},
                name="Malla de hueso",
            )
        ]
    )
    fig.update_layout(
        title=(
            f"Visualizador 1 mesh raw caso {case_id} "
            f"HU >= {hu_threshold}, stride={stride}"
        ),
        scene={
            "xaxis_title": "x mm",
            "yaxis_title": "y mm",
            "zaxis_title": "z mm",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )

    output_html = OUTPUT_DIR / f"visualizador1_mesh_raw_caso_{case_id}.html"
    fig.write_html(output_html, include_plotlyjs="cdn")

    summary_path = OUTPUT_DIR / f"visualizador1_mesh_raw_caso_{case_id}_resumen.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"Caso: {case_id}",
                f"Imagen: {to_repo_relative(image_path)}",
                f"Shape array z_y_x: {volume.shape}",
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
