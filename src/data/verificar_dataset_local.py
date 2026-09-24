from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_DIR / "src"))

from utils.project_paths import IMAGES_PART1, IMAGES_PART2, LABELS


def ids_in(folder: Path) -> set[str]:
    return {path.stem for path in folder.glob("*.mha")}


def main():
    ids_part1 = ids_in(IMAGES_PART1)
    ids_part2 = ids_in(IMAGES_PART2)
    image_ids = ids_part1 | ids_part2
    label_ids = ids_in(LABELS)

    print(f"Imagenes parte 1: {len(ids_part1)}")
    print(f"Imagenes parte 2: {len(ids_part2)}")
    print(f"Imagenes totales unicas: {len(image_ids)}")
    print(f"Labels totales: {len(label_ids)}")
    print(f"Duplicados entre partes: {len(ids_part1 & ids_part2)}")
    print(f"Imagenes sin label: {sorted(image_ids - label_ids)}")
    print(f"Labels sin imagen: {sorted(label_ids - image_ids)}")

    if len(image_ids) != 100 or len(label_ids) != 100:
        raise SystemExit("Dataset incompleto: se esperaban 100 imagenes y 100 labels.")
    if image_ids != label_ids:
        raise SystemExit("Los IDs de imagenes y labels no coinciden.")

    print("Dataset local verificado correctamente.")


if __name__ == "__main__":
    main()
