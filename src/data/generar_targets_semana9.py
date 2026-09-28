"""Genera manifiesto y evidencia visual de los targets de deteccion."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import SimpleITK as sitk


sys.path.append(str(PROJECT_DIR / "src"))

from data.targets_detection import (  # noqa: E402
    CLASS_ID_TO_NAME,
    apply_hu_window,
    build_detection_target,
    image_path_for_case,
    label_path_for_case,
    letterbox_pair,
    read_split_ids,
)


COLORS = {
    1: "#f4c542",
    2: "#2b8cbe",
    3: "#e34a33",
}


def build_manifest(split: str, min_pixels: int) -> pd.DataFrame:
    rows = []
    for case_id in read_split_ids(split):
        label_volume = sitk.GetArrayFromImage(
            sitk.ReadImage(str(label_path_for_case(case_id)))
        )
        for slice_index, label_slice in enumerate(label_volume):
            target = build_detection_target(
                label_slice,
                case_id,
                slice_index,
                min_pixels=min_pixels,
            )
            if not target["labels"].size:
                continue
            rows.append(
                {
                    "split": split,
                    "case_id": case_id,
                    "slice_index": slice_index,
                    "n_regions": len(target["labels"]),
                    "regions": "|".join(target["class_names"]),
                    "n_instances": len(target["instance_ids"]),
                }
            )
    return pd.DataFrame(rows)


def choose_preview_slices(manifest: pd.DataFrame, count: int) -> list[int]:
    candidates = manifest[manifest["n_regions"] == 3]
    if candidates.empty:
        candidates = manifest
    positions = np.linspace(0, len(candidates) - 1, min(count, len(candidates))).astype(int)
    return candidates.iloc[positions]["slice_index"].astype(int).tolist()


def render_preview(
    case_id: str,
    manifest: pd.DataFrame,
    output_path: Path,
    image_size: int,
    min_pixels: int,
    count: int,
) -> None:
    image_volume = sitk.GetArrayFromImage(
        sitk.ReadImage(str(image_path_for_case(case_id)))
    ).astype(np.float32)
    label_volume = sitk.GetArrayFromImage(
        sitk.ReadImage(str(label_path_for_case(case_id)))
    ).astype(np.int16)
    slices = choose_preview_slices(manifest, count)

    columns = 3
    rows = int(np.ceil(len(slices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(13, 4.3 * rows))
    axes = np.atleast_1d(axes).reshape(-1)

    for axis, slice_index in zip(axes, slices):
        image = apply_hu_window(image_volume[slice_index])
        image, label, transform = letterbox_pair(
            image,
            label_volume[slice_index],
            image_size,
        )
        scaled_min_pixels = max(
            1,
            int(round(min_pixels * float(transform["scale"]) ** 2)),
        )
        target = build_detection_target(
            label,
            case_id,
            slice_index,
            scaled_min_pixels,
        )
        semantic = target["semantic_mask"]

        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        for class_id, color in COLORS.items():
            overlay = np.ma.masked_where(
                semantic != class_id,
                np.ones_like(semantic, dtype=np.float32),
            )
            axis.imshow(
                overlay,
                cmap=ListedColormap([color]),
                alpha=0.16,
                vmin=0,
                vmax=1,
            )
        for box, class_id in zip(target["boxes"], target["labels"]):
            x_min, y_min, x_max, y_max = box
            color = COLORS[int(class_id)]
            axis.add_patch(
                patches.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    linewidth=2,
                    edgecolor=color,
                    facecolor="none",
                )
            )
            axis.text(
                x_min,
                max(8, y_min - 4),
                CLASS_ID_TO_NAME[int(class_id)].replace("_", " "),
                color="black",
                fontsize=8,
                bbox={"facecolor": color, "alpha": 0.9, "pad": 2, "edgecolor": "none"},
            )
        axis.set_title(f"Caso {case_id} | corte z={slice_index}")
        axis.axis("off")

    for axis in axes[len(slices) :]:
        axis.axis("off")

    fig.suptitle(
        "Targets de deteccion PENGWIN: una bounding box por region anatomica",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--min-pixels", type=int, default=20)
    parser.add_argument("--preview-slices", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "evidence" / "entrega2" / "datos_targets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifests = [build_manifest(split, args.min_pixels) for split in ("train", "val", "test")]
    manifest = pd.concat(manifests, ignore_index=True)
    manifest_path = args.output_dir / "manifest_cortes_positivos.csv"
    manifest.to_csv(manifest_path, index=False)

    case_id = args.case_id or read_split_ids("train")[0]
    case_manifest = manifest[manifest["case_id"] == case_id]
    if case_manifest.empty:
        raise SystemExit(f"El caso {case_id} no tiene cortes positivos.")
    preview_path = args.output_dir / f"targets_bounding_boxes_caso_{case_id}.png"
    render_preview(
        case_id,
        case_manifest,
        preview_path,
        args.image_size,
        args.min_pixels,
        args.preview_slices,
    )

    summary = manifest.groupby("split").agg(
        casos=("case_id", "nunique"),
        cortes_positivos=("slice_index", "size"),
        promedio_regiones_por_corte=("n_regions", "mean"),
        max_instancias_por_corte=("n_instances", "max"),
    )
    print("TARGETS DE DETECCION GENERADOS")
    print(summary.round(2))
    print(f"\nManifest: {manifest_path.relative_to(PROJECT_DIR)}")
    print(f"Evidencia: {preview_path.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
