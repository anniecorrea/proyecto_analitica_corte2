from pathlib import Path
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR / "src"))

from utils.project_paths import IMAGES_PART1, IMAGES_PART2, LABELS, SPLITS


SEED = 42
TRAIN_SIZE = 75
VAL_SIZE = 5
TEST_SIZE = 20


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


def to_repo_relative(path: Path) -> str:
    return path.relative_to(PROJECT_DIR).as_posix()


def collect_case_ids() -> list[str]:
    image_ids = {path.stem for path in IMAGES_PART1.glob("*.mha")}
    image_ids |= {path.stem for path in IMAGES_PART2.glob("*.mha")}
    label_ids = {path.stem for path in LABELS.glob("*.mha")}
    case_ids = sorted(image_ids & label_ids)

    if len(case_ids) != 100:
        raise SystemExit(f"Se esperaban 100 casos y se encontraron {len(case_ids)}.")
    if image_ids != label_ids:
        raise SystemExit("Los IDs de imagenes y labels no coinciden.")

    return case_ids


def fragment_count(case_id: str) -> int:
    label = sitk.ReadImage(str(label_path_for_case(case_id)))
    label_array = sitk.GetArrayFromImage(label)
    return int(len(np.unique(label_array)) - 1)


def complexity_bin(n_fragments: int) -> str:
    if n_fragments <= 4:
        return "baja"
    if n_fragments <= 6:
        return "media"
    return "alta"


def build_metadata(case_ids: list[str]) -> pd.DataFrame:
    rows = []
    for case_id in case_ids:
        n_fragments = fragment_count(case_id)
        rows.append(
            {
                "case_id": case_id,
                "n_fragments": n_fragments,
                "complexity": complexity_bin(n_fragments),
                "image_path": to_repo_relative(image_path_for_case(case_id)),
                "label_path": to_repo_relative(label_path_for_case(case_id)),
            }
        )
    return pd.DataFrame(rows)


def split_cases(metadata: pd.DataFrame):
    train_val, test = train_test_split(
        metadata,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
        stratify=metadata["complexity"],
    )

    train, val = train_test_split(
        train_val,
        test_size=VAL_SIZE,
        random_state=SEED,
        shuffle=True,
        stratify=train_val["complexity"],
    )

    if len(train) != TRAIN_SIZE or len(val) != VAL_SIZE or len(test) != TEST_SIZE:
        raise SystemExit("Los tamanos de split no coinciden con 75/5/20.")

    return (
        train.sort_values("case_id"),
        val.sort_values("case_id"),
        test.sort_values("case_id"),
    )


def write_ids(path: Path, case_ids: pd.Series):
    path.write_text("\n".join(case_ids.tolist()) + "\n", encoding="utf-8")


def main():
    SPLITS.mkdir(parents=True, exist_ok=True)
    case_ids = collect_case_ids()
    metadata = build_metadata(case_ids)
    train, val, test = split_cases(metadata)

    split_map = {
        "train": train,
        "val": val,
        "test": test,
    }

    dataset = []
    for split_name, frame in split_map.items():
        write_ids(SPLITS / f"{split_name}.txt", frame["case_id"])
        temp = frame.copy()
        temp.insert(1, "split", split_name)
        dataset.append(temp)

    dataset = pd.concat(dataset, ignore_index=True).sort_values("case_id")
    dataset.to_csv(SPLITS / "dataset.csv", index=False)

    summary = dataset.groupby(["split", "complexity"]).size().unstack(fill_value=0)
    print("Splits generados con semilla fija:", SEED)
    print(dataset["split"].value_counts().reindex(["train", "val", "test"]))
    print("\nDistribucion por complejidad:")
    print(summary)
    print("\nArchivos actualizados en:", SPLITS)


if __name__ == "__main__":
    main()
