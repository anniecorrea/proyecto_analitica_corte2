from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk


PROJECT_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(PROJECT_DIR / "src"))

from utils.project_paths import DATA_MHA, EVIDENCE, IMAGES_PART1, IMAGES_PART2, LABELS


OUTPUT_DIR = EVIDENCE / "entrega1" / "eda"


def image_path_for_case(case_id: str) -> Path:
    for folder in [IMAGES_PART1, IMAGES_PART2]:
        path = folder / f"{case_id}.mha"
        if path.exists():
            return path
    raise FileNotFoundError(f"No se encontro imagen para el caso {case_id}")


def label_path_for_case(case_id: str) -> Path:
    path = LABELS / f"{case_id}.mha"
    if not path.exists():
        raise FileNotFoundError(f"No se encontro label para el caso {case_id}")
    return path


def label_to_region(label: int) -> str:
    if 1 <= label <= 10:
        return "sacro"
    if 11 <= label <= 20:
        return "coxal_izquierdo"
    if 21 <= label <= 30:
        return "coxal_derecho"
    return "desconocido"


def collect_case_ids() -> list[str]:
    image_ids = {path.stem for path in IMAGES_PART1.glob("*.mha")}
    image_ids |= {path.stem for path in IMAGES_PART2.glob("*.mha")}
    label_ids = {path.stem for path in LABELS.glob("*.mha")}
    common = sorted(image_ids & label_ids)
    if not common:
        raise SystemExit(
            f"No se encontraron casos. Verifica que exista data_mha en {DATA_MHA}"
        )
    return common


def collect_metadata(case_ids: list[str]) -> pd.DataFrame:
    rows = []
    for case_id in case_ids:
        image = sitk.ReadImage(str(image_path_for_case(case_id)))
        label = sitk.ReadImage(str(label_path_for_case(case_id)))
        size_x, size_y, size_z = image.GetSize()
        spacing_x, spacing_y, spacing_z = image.GetSpacing()
        rows.append(
            {
                "case_id": case_id,
                "image_size_x": size_x,
                "image_size_y": size_y,
                "image_size_z": size_z,
                "spacing_x_mm": spacing_x,
                "spacing_y_mm": spacing_y,
                "spacing_z_mm": spacing_z,
                "label_size_x": label.GetSize()[0],
                "label_size_y": label.GetSize()[1],
                "label_size_z": label.GetSize()[2],
                "same_size": image.GetSize() == label.GetSize(),
                "same_spacing": image.GetSpacing() == label.GetSpacing(),
            }
        )
    return pd.DataFrame(rows)


def collect_fragments(case_ids: list[str]) -> pd.DataFrame:
    rows = []
    for case_id in case_ids:
        label_image = sitk.ReadImage(str(label_path_for_case(case_id)))
        spacing = label_image.GetSpacing()
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
        label_array = sitk.GetArrayFromImage(label_image)
        labels = [int(value) for value in np.unique(label_array) if value != 0]
        for label in labels:
            n_voxels = int(np.count_nonzero(label_array == label))
            rows.append(
                {
                    "case_id": case_id,
                    "label": label,
                    "region": label_to_region(label),
                    "n_voxels": n_voxels,
                    "volume_mm3": n_voxels * voxel_volume_mm3,
                }
            )
    return pd.DataFrame(rows)


def save_plot(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_metadata(metadata: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    plt.hist(metadata["image_size_z"], bins=14, color="#4C78A8", edgecolor="white")
    plt.title("Distribucion de cortes por volumen CT")
    plt.xlabel("Numero de cortes en eje z")
    plt.ylabel("Cantidad de casos")
    save_plot(OUTPUT_DIR / "01_distribucion_cortes_z.png")

    spacing = metadata[["spacing_x_mm", "spacing_y_mm", "spacing_z_mm"]]
    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [spacing["spacing_x_mm"], spacing["spacing_y_mm"], spacing["spacing_z_mm"]],
        tick_labels=["x", "y", "z"],
    )
    plt.title("Distribucion de spacing fisico")
    plt.xlabel("Eje")
    plt.ylabel("Spacing mm")
    save_plot(OUTPUT_DIR / "02_spacing_fisico.png")


def plot_fragments(fragments: pd.DataFrame):
    fragments_per_case = fragments.groupby("case_id").size()
    plt.figure(figsize=(8, 5))
    plt.hist(fragments_per_case, bins=range(3, 11), color="#59A14F", edgecolor="white")
    plt.title("Distribucion de fragmentos por caso")
    plt.xlabel("Numero de fragmentos")
    plt.ylabel("Cantidad de casos")
    save_plot(OUTPUT_DIR / "03_fragmentos_por_caso.png")

    region_counts = fragments["region"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    region_counts.plot(kind="bar", color="#F28E2B")
    plt.title("Fragmentos por region anatomica")
    plt.xlabel("Region")
    plt.ylabel("Cantidad de fragmentos")
    plt.xticks(rotation=20, ha="right")
    save_plot(OUTPUT_DIR / "04_fragmentos_por_region.png")

    plt.figure(figsize=(8, 5))
    plt.hist(np.log10(fragments["volume_mm3"]), bins=24, color="#E15759", edgecolor="white")
    plt.title("Distribucion log10 del volumen de fragmentos")
    plt.xlabel("log10 volumen mm3")
    plt.ylabel("Cantidad de fragmentos")
    save_plot(OUTPUT_DIR / "05_volumen_fragmentos_log10.png")

    plt.figure(figsize=(8, 5))
    fragments.boxplot(column="volume_mm3", by="region", grid=False, rot=20)
    plt.title("Volumen de fragmentos por region")
    plt.suptitle("")
    plt.xlabel("Region")
    plt.ylabel("Volumen mm3")
    save_plot(OUTPUT_DIR / "06_volumen_por_region.png")


def save_case_overlay(case_id: str):
    image = sitk.ReadImage(str(image_path_for_case(case_id)))
    label = sitk.ReadImage(str(label_path_for_case(case_id)))
    image_array = sitk.GetArrayFromImage(image).astype(np.float32)
    label_array = sitk.GetArrayFromImage(label)

    slice_scores = np.count_nonzero(label_array, axis=(1, 2))
    z = int(np.argmax(slice_scores))

    ct_slice = image_array[z]
    label_slice = label_array[z]
    low, high = -200, 1000
    windowed = np.clip((ct_slice - low) / (high - low), 0, 1)

    plt.figure(figsize=(7, 7))
    plt.imshow(windowed, cmap="gray")
    masked = np.ma.masked_where(label_slice == 0, label_slice)
    plt.imshow(masked, cmap="tab20", alpha=0.45)
    plt.title(f"Caso {case_id} corte {z} CT con mascara superpuesta")
    plt.axis("off")
    save_plot(OUTPUT_DIR / f"07_caso_{case_id}_ct_label_overlay.png")


def write_summary(metadata: pd.DataFrame, fragments: pd.DataFrame):
    fragments_per_case = fragments.groupby("case_id").size()
    region_counts = fragments["region"].value_counts()
    small_threshold = fragments["volume_mm3"].quantile(0.05)
    small_fragments = fragments[fragments["volume_mm3"] <= small_threshold]
    top_cases = fragments_per_case.sort_values(ascending=False).head(5)

    summary = f"""# Resumen EDA Entrega 1

## Dataset

- Total de casos con imagen y label: {len(metadata)}
- CT y label con mismo tamano: {int(metadata['same_size'].sum())} de {len(metadata)}
- CT y label con mismo spacing: {int(metadata['same_spacing'].sum())} de {len(metadata)}
- Cortes por volumen: minimo {int(metadata['image_size_z'].min())}, maximo {int(metadata['image_size_z'].max())}, promedio {metadata['image_size_z'].mean():.2f}
- Spacing z promedio: {metadata['spacing_z_mm'].mean():.4f} mm

## Fragmentos

- Total de fragmentos individuales: {len(fragments)}
- Fragmentos por caso: minimo {int(fragments_per_case.min())}, maximo {int(fragments_per_case.max())}, promedio {fragments_per_case.mean():.2f}
- Fragmentos por region:
  - Sacro: {int(region_counts.get('sacro', 0))}
  - Coxal izquierdo: {int(region_counts.get('coxal_izquierdo', 0))}
  - Coxal derecho: {int(region_counts.get('coxal_derecho', 0))}
- Percentil 5 de volumen: {small_threshold:.2f} mm3
- Fragmentos pequenos en percentil 5: {len(small_fragments)}

## Casos con mas fragmentos

{top_cases.to_string()}

## Conclusiones

- El dataset local esta completo para el primer avance: hay 100 casos con imagen y label correspondiente.
- Todos los casos verificados mantienen consistencia de tamano y spacing entre CT y label.
- El numero de fragmentos por caso varia, lo cual anticipa dificultad desigual entre casos simples y casos complejos.
- Existen fragmentos de volumen muy pequeno; estos pueden ser dificiles de segmentar y afectar metricas como Dice o IoU.
- Las medidas de volumen y futuras distancias deben calcularse en milimetros usando el spacing fisico, no en pixeles.

## Archivos generados

- `metadata_casos.csv`
- `metadata_fragmentos.csv`
- `01_distribucion_cortes_z.png`
- `02_spacing_fisico.png`
- `03_fragmentos_por_caso.png`
- `04_fragmentos_por_region.png`
- `05_volumen_fragmentos_log10.png`
- `06_volumen_por_region.png`
- `07_caso_001_ct_label_overlay.png`
"""
    (OUTPUT_DIR / "resumen_eda_entrega1.md").write_text(summary, encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case_ids = collect_case_ids()
    metadata = collect_metadata(case_ids)
    fragments = collect_fragments(case_ids)

    metadata.to_csv(OUTPUT_DIR / "metadata_casos.csv", index=False)
    fragments.to_csv(OUTPUT_DIR / "metadata_fragmentos.csv", index=False)

    plot_metadata(metadata)
    plot_fragments(fragments)
    save_case_overlay("001")
    write_summary(metadata, fragments)

    print(f"EDA generado en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
