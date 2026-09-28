"""Carga 2D y construccion de targets de deteccion para PENGWIN.

Las mascaras PENGWIN codifican instancias por decenas:
1-10 sacro, 11-20 coxal izquierdo y 21-30 coxal derecho. Para la cabeza
de deteccion se consolida cada region presente en el corte en una sola caja.
Las coordenadas se expresan como ``[x_min, y_min, x_max, y_max)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import SimpleITK as sitk

from utils.project_paths import IMAGES_PART1, IMAGES_PART2, LABELS, SPLITS


CLASS_ID_TO_NAME = {
    1: "sacro",
    2: "coxal_izquierdo",
    3: "coxal_derecho",
}


@dataclass(frozen=True)
class SliceRecord:
    case_id: str
    slice_index: int


def anatomical_class(instance_id: int) -> int:
    """Convierte el ID de instancia PENGWIN en una clase anatomica."""
    if 1 <= instance_id <= 10:
        return 1
    if 11 <= instance_id <= 20:
        return 2
    if 21 <= instance_id <= 30:
        return 3
    if instance_id == 0:
        return 0
    raise ValueError(f"Etiqueta PENGWIN fuera de la taxonomia esperada: {instance_id}")


def instance_to_semantic(label_slice: np.ndarray) -> np.ndarray:
    """Transforma una mascara de instancias en tres clases anatomicas."""
    label_slice = np.asarray(label_slice)
    unknown = np.unique(label_slice[(label_slice < 0) | (label_slice > 30)])
    if unknown.size:
        raise ValueError(f"Etiquetas fuera de rango encontradas: {unknown.tolist()}")

    semantic = np.zeros(label_slice.shape, dtype=np.uint8)
    semantic[(label_slice >= 1) & (label_slice <= 10)] = 1
    semantic[(label_slice >= 11) & (label_slice <= 20)] = 2
    semantic[(label_slice >= 21) & (label_slice <= 30)] = 3
    return semantic


def boxes_from_semantic(
    semantic_mask: np.ndarray,
    min_pixels: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrae una caja por region anatomica presente en el corte."""
    boxes: list[list[float]] = []
    labels: list[int] = []
    areas: list[float] = []

    for class_id in CLASS_ID_TO_NAME:
        ys, xs = np.nonzero(semantic_mask == class_id)
        if xs.size < min_pixels:
            continue
        x_min, x_max = int(xs.min()), int(xs.max()) + 1
        y_min, y_max = int(ys.min()), int(ys.max()) + 1
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(class_id)
        areas.append(float((x_max - x_min) * (y_max - y_min)))

    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(labels, dtype=np.int64),
        np.asarray(areas, dtype=np.float32),
    )


def build_detection_target(
    label_slice: np.ndarray,
    case_id: str,
    slice_index: int,
    min_pixels: int = 20,
) -> dict[str, Any]:
    """Construye el contrato comun entre el cargador y la cabeza de deteccion."""
    semantic = instance_to_semantic(label_slice)
    boxes, labels, areas = boxes_from_semantic(semantic, min_pixels=min_pixels)
    instance_ids = np.unique(label_slice)
    instance_ids = instance_ids[instance_ids != 0].astype(np.int64)

    return {
        "case_id": case_id,
        "slice_index": int(slice_index),
        "boxes": boxes,
        "labels": labels,
        "areas": areas,
        "class_names": [CLASS_ID_TO_NAME[int(label)] for label in labels],
        "semantic_mask": semantic,
        "instance_mask": np.asarray(label_slice, dtype=np.int16),
        "instance_ids": instance_ids,
    }


def apply_hu_window(
    image: np.ndarray,
    center: float = 400.0,
    width: float = 1800.0,
) -> np.ndarray:
    """Aplica ventana osea y normaliza el corte al intervalo [0, 1]."""
    if width <= 0:
        raise ValueError("El ancho de ventana HU debe ser positivo.")
    lower = center - width / 2.0
    upper = center + width / 2.0
    clipped = np.clip(np.asarray(image, dtype=np.float32), lower, upper)
    return ((clipped - lower) / (upper - lower)).astype(np.float32)


def letterbox_pair(
    image: np.ndarray,
    label: np.ndarray,
    output_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Redimensiona sin deformar y rellena hasta un lienzo cuadrado."""
    if image.shape != label.shape or image.ndim != 2:
        raise ValueError("Imagen y label deben ser matrices 2D con la misma forma.")
    if output_size <= 0:
        raise ValueError("output_size debe ser positivo.")

    height, width = image.shape
    scale = min(output_size / width, output_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    offset_x = (output_size - new_width) // 2
    offset_y = (output_size - new_height) // 2

    image_resized = np.asarray(
        Image.fromarray(image.astype(np.float32), mode="F").resize(
            (new_width, new_height), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    label_resized = np.asarray(
        Image.fromarray(label.astype(np.int32), mode="I").resize(
            (new_width, new_height), Image.Resampling.NEAREST
        ),
        dtype=np.int16,
    )

    image_out = np.zeros((output_size, output_size), dtype=np.float32)
    label_out = np.zeros((output_size, output_size), dtype=np.int16)
    y_end, x_end = offset_y + new_height, offset_x + new_width
    image_out[offset_y:y_end, offset_x:x_end] = image_resized
    label_out[offset_y:y_end, offset_x:x_end] = label_resized

    metadata: dict[str, float | int] = {
        "scale": float(scale),
        "offset_x": offset_x,
        "offset_y": offset_y,
        "original_height": height,
        "original_width": width,
        "output_size": output_size,
    }
    return image_out, label_out, metadata


def image_path_for_case(case_id: str) -> Path:
    for directory in (IMAGES_PART1, IMAGES_PART2):
        candidate = directory / f"{case_id}.mha"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No se encontro CT para el caso {case_id}.")


def label_path_for_case(case_id: str) -> Path:
    candidate = LABELS / f"{case_id}.mha"
    if not candidate.exists():
        raise FileNotFoundError(f"No se encontro label para el caso {case_id}.")
    return candidate


def load_case(case_id: str) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    """Carga CT y label como arreglos con orden (z, y, x)."""
    image_itk = sitk.ReadImage(str(image_path_for_case(case_id)))
    label_itk = sitk.ReadImage(str(label_path_for_case(case_id)))
    if image_itk.GetSize() != label_itk.GetSize():
        raise ValueError(f"CT y label del caso {case_id} no tienen el mismo tamano.")
    if not np.allclose(image_itk.GetSpacing(), label_itk.GetSpacing()):
        raise ValueError(f"CT y label del caso {case_id} no tienen el mismo spacing.")

    image = sitk.GetArrayFromImage(image_itk).astype(np.float32)
    label = sitk.GetArrayFromImage(label_itk).astype(np.int16)
    return image, label, tuple(float(v) for v in image_itk.GetSpacing())


def read_split_ids(split: str) -> list[str]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split debe ser train, val o test.")
    path = SPLITS / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Genere primero los splits fijos.")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def positive_slice_indices(label_volume: np.ndarray, min_pixels: int = 20) -> list[int]:
    """Devuelve cortes que contienen al menos una region util para deteccion."""
    indices = []
    for index, label_slice in enumerate(label_volume):
        semantic = instance_to_semantic(label_slice)
        _, labels, _ = boxes_from_semantic(semantic, min_pixels=min_pixels)
        if labels.size:
            indices.append(index)
    return indices


class PengwinDetectionDataset:
    """Dataset corte a corte con targets genericos o tensores PyTorch.

    El indice incluye solo cortes positivos por defecto. ``as_torch=False``
    permite validar los datos sin instalar ni importar PyTorch.
    """

    def __init__(
        self,
        split: str = "train",
        image_size: int = 256,
        min_pixels: int = 20,
        include_empty: bool = False,
        channels: int = 1,
        as_torch: bool = False,
    ) -> None:
        if channels not in {1, 3}:
            raise ValueError("channels debe ser 1 o 3.")
        self.split = split
        self.image_size = image_size
        self.min_pixels = min_pixels
        self.include_empty = include_empty
        self.channels = channels
        self.as_torch = as_torch
        self.records = self._build_index()
        self._cached_case_id: str | None = None
        self._cached_image: np.ndarray | None = None
        self._cached_label: np.ndarray | None = None
        self._cached_spacing: tuple[float, ...] | None = None

    def _build_index(self) -> list[SliceRecord]:
        records = []
        for case_id in read_split_ids(self.split):
            label_itk = sitk.ReadImage(str(label_path_for_case(case_id)))
            label_volume = sitk.GetArrayFromImage(label_itk)
            if self.include_empty:
                indices = range(label_volume.shape[0])
            else:
                indices = positive_slice_indices(label_volume, self.min_pixels)
            records.extend(SliceRecord(case_id, int(index)) for index in indices)
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _get_case(self, case_id: str) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
        if case_id != self._cached_case_id:
            image, label, spacing = load_case(case_id)
            self._cached_case_id = case_id
            self._cached_image = image
            self._cached_label = label
            self._cached_spacing = spacing
        assert self._cached_image is not None
        assert self._cached_label is not None
        assert self._cached_spacing is not None
        return self._cached_image, self._cached_label, self._cached_spacing

    def __getitem__(self, index: int):
        record = self.records[index]
        image_volume, label_volume, spacing = self._get_case(record.case_id)
        image_slice = apply_hu_window(image_volume[record.slice_index])
        image_slice, label_slice, transform = letterbox_pair(
            image_slice,
            label_volume[record.slice_index],
            output_size=self.image_size,
        )
        scaled_min_pixels = max(
            1,
            int(round(self.min_pixels * float(transform["scale"]) ** 2)),
        )
        target = build_detection_target(
            label_slice,
            record.case_id,
            record.slice_index,
            min_pixels=scaled_min_pixels,
        )
        target["spacing_xyz_mm"] = spacing
        target["transform"] = transform
        target["min_pixels_after_resize"] = scaled_min_pixels

        image_chw = image_slice[None, ...]
        if self.channels == 3:
            image_chw = np.repeat(image_chw, 3, axis=0)

        if not self.as_torch:
            return image_chw.astype(np.float32), target

        try:
            import torch
        except ImportError as error:
            raise ImportError("Instale PyTorch para usar as_torch=True.") from error

        torch_target = {
            **target,
            "boxes": torch.from_numpy(target["boxes"]),
            "labels": torch.from_numpy(target["labels"]),
            "areas": torch.from_numpy(target["areas"]),
            "semantic_mask": torch.from_numpy(target["semantic_mask"].astype(np.int64)),
            "instance_mask": torch.from_numpy(target["instance_mask"].astype(np.int64)),
            "instance_ids": torch.from_numpy(target["instance_ids"]),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        return torch.from_numpy(image_chw.astype(np.float32)), torch_target


def detection_collate(batch):
    """Collate para cajas con cantidad variable por imagen."""
    images, targets = zip(*batch)
    return list(images), list(targets)
