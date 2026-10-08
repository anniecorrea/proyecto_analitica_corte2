"""Genera proyecciones ortogonales para revisar distancias de fragmentos.

Uso desde la raíz del repositorio:
    python3 generar_evidencias_distancias.py

Las distancias del título se leen del CSV actual; los contornos son
proyecciones 2D de las máscaras MHA y no sustituyen la medición 3D.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "splits" / "dataset.csv"
DISTANCES_PATH = ROOT / "evidencias" / "distancias_referencia.csv"
OUTPUT_DIR = ROOT / "evidencias" / "distancias_revision"


def projection_views(volume: np.ndarray):
    """Return axial, coronal, and sagittal maximum-intensity projections."""
    return (
        (np.max(volume, axis=0), 0, "Axial: proyección sobre z"),
        (np.max(volume, axis=1), 1, "Coronal: proyección sobre y"),
        (np.max(volume, axis=2), 2, "Sagital: proyección sobre x"),
    )


def main() -> None:
    dataset = pd.read_csv(DATASET_PATH, dtype={"case_id": str})
    distances = pd.read_csv(DISTANCES_PATH, dtype={"case_id": str})
    candidates = distances.loc[~distances["is_main"].astype(bool)].copy()
    candidates["distance_mm"] = pd.to_numeric(candidates["distance_mm"])

    # Keep the same representative cases used for manual review, updating values
    # directly from the current distance table.
    selected = [("001", "coxal_izquierdo", 12), ("044", "coxal_derecho", 22)]
    for row in candidates.nlargest(3, "distance_mm").itertuples(index=False):
        selected.append((str(row.case_id), row.region, int(row.fragment_id)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for case_id, region, fragment_id in selected:
        match = candidates.loc[
            (candidates["case_id"] == case_id)
            & (candidates["region"] == region)
            & (candidates["fragment_id"] == fragment_id)
        ]
        if match.empty:
            print(f"Se omite {case_id}/{region}/{fragment_id}: no está en el CSV.")
            continue
        record = match.iloc[0]
        main_id = int(record["main_fragment_id"])
        dataset_row = dataset.loc[dataset["case_id"] == case_id].iloc[0]
        image = sitk.ReadImage(str(ROOT / dataset_row["image_path"]))
        label = sitk.ReadImage(str(ROOT / dataset_row["label_path"]))
        image_volume = sitk.GetArrayFromImage(image).astype(np.float32)
        labels = sitk.GetArrayFromImage(label)
        if image_volume.shape != labels.shape:
            raise ValueError(f"Imagen y máscara no coinciden para el caso {case_id}.")

        windowed = np.clip((image_volume + 200.0) / 1800.0, 0.0, 1.0)
        fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
        for axis, (image_projection, projection_axis, title) in zip(
            axes, projection_views(windowed)
        ):
            main_projection = np.any(labels == main_id, axis=projection_axis)
            fragment_projection = np.any(labels == fragment_id, axis=projection_axis)
            axis.imshow(image_projection, cmap="gray", vmin=0, vmax=1)
            if main_projection.any():
                axis.contour(main_projection, levels=[0.5], colors=["#16a779"], linewidths=1.2)
            if fragment_projection.any():
                axis.contour(fragment_projection, levels=[0.5], colors=["#ee4c9b"], linewidths=1.2)
            axis.set_title(title)
            axis.axis("off")

        fig.suptitle(
            f"Caso {case_id} | {region} | fragmento {fragment_id} vs. principal {main_id}"
            f" | distancia 3D borde a borde = {float(record['distance_mm']):.3f} mm\n"
            "Proyecciones MIP 2D de máscaras voxelizadas; no representan distancia 2D",
            fontsize=13,
        )
        fig.legend(
            [
                plt.Line2D([0], [0], color="#16a779", lw=1.5),
                plt.Line2D([0], [0], color="#ee4c9b", lw=1.5),
            ],
            [f"Principal {main_id}", f"Fragmento {fragment_id}"],
            loc="lower center",
            ncol=2,
            frameon=False,
        )
        output_path = OUTPUT_DIR / f"distancia_caso_{case_id}_frag_{fragment_id}.png"
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"{output_path.relative_to(ROOT)}: {float(record['distance_mm']):.3f} mm")


if __name__ == "__main__":
    main()
