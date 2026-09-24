from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = project_root()
DATA_MHA = ROOT / "data_mha"
IMAGES_PART1 = DATA_MHA / "PENGWIN_CT_train_images_part1"
IMAGES_PART2 = DATA_MHA / "PENGWIN_CT_train_images_part2"
LABELS = DATA_MHA / "PENGWIN_CT_train_labels"
SPLITS = ROOT / "splits"
EVIDENCE = ROOT / "evidence"
