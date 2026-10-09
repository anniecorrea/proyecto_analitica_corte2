
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_otsu
from sklearn.metrics import f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
BASE_DIR = ROOT
IMAGE_SIZE = 256
GRID_SIZE = 16
MAX_TRAIN_SLICES = None
MAX_VAL_SLICES = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = ROOT / "splits" / "dataset.csv"
MANIFEST_PATH = ROOT / "evidencias" / "datos_targets" / "manifest_cortes_positivos.csv"
CACHE_DIR = ROOT / "evidencias" / "cache_entrenamiento"
OUT_DIR = ROOT / "evidencias" / "evaluacion_hoyos"
DEFAULT_CHECKPOINT = ROOT / "evidencias" / "entrenamiento_multitarea" / "modelo_multitarea_final.pt"
CLASS_IDS = (1, 2, 3)
N_WORST_EXAMPLES = 5   # el informe pide una galeria de los cinco peores casos
N_BEST_EXAMPLES = 3
REGION_COLORS = {
    1: np.array([0.95, 0.60, 0.10], dtype=np.float32),
    2: np.array([0.20, 0.45, 0.95], dtype=np.float32),
    3: np.array([0.15, 0.75, 0.40], dtype=np.float32),
}
_COMPONENTS: dict[str, Any] = {
    "__builtins__": __builtins__,
    "__name__": __name__,
    "Path": Path,
    "ROOT": ROOT,
    "BASE_DIR": BASE_DIR,
    "IMAGE_SIZE": IMAGE_SIZE,
    "GRID_SIZE": GRID_SIZE,
    "MAX_TRAIN_SLICES": MAX_TRAIN_SLICES,
    "MAX_VAL_SLICES": MAX_VAL_SLICES,
    "DEVICE": DEVICE,
    "np": np,
    "pd": pd,
    "torch": torch,
    "nn": nn,
    "F": F,
    "sitk": sitk,
    "Image": Image,
    "gc": gc,
    "hashlib": hashlib,
    "json": json,
    "math": math,
    "time": time,
    "defaultdict": defaultdict,
    "maximum_filter": maximum_filter,
    "equalize_adapthist": equalize_adapthist,
    "threshold_otsu": threshold_otsu,
    "DATASET_PATH": DATASET_PATH,
    "MANIFEST_PATH": MANIFEST_PATH,
    "CACHE_DIR": CACHE_DIR,
}


def _target_names(node: ast.AST) -> set[str]:
    targets = node.targets if isinstance(node, ast.Assign) else (
        [node.target] if isinstance(node, ast.AnnAssign) else []
    )
    names = set()
    for target in targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.update(item.id for item in target.elts if isinstance(item, ast.Name))
    return names


def _load_notebook_definitions(
    notebook_path: Path,
    definition_names: set[str],
    value_names: set[str] | None = None,
) -> None:
    if not notebook_path.is_file():
        raise FileNotFoundError(f"Falta el notebook requerido: {notebook_path}")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    wanted_values = value_names or set()
    snippets: list[str] = []
    found_defs: set[str] = set()
    found_values: set[str] = set()
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            name = getattr(node, "name", None)
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and name in definition_names
                and name not in found_defs
            ):
                snippets.append(ast.get_source_segment(source, node) or "")
                found_defs.add(name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                selected = _target_names(node) & wanted_values
                if selected and not selected.issubset(found_values):
                    snippets.append(ast.get_source_segment(source, node) or "")
                    found_values.update(selected)

    missing_defs = definition_names - found_defs
    missing_values = wanted_values - found_values
    if missing_defs or missing_values:
        raise RuntimeError(
            f"Faltan definiciones en {notebook_path.name}: "
            f"funciones/clases={sorted(missing_defs)}, constantes={sorted(missing_values)}"
        )
    exec("\n\n".join(snippets), _COMPONENTS)


def _sha1(path: Path, length: int = 12) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def _load_project_components() -> None:
    global df, slice_manifest, DATA_VERSION
    for path in (DATASET_PATH, MANIFEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Falta el archivo requerido: {path}")
    df = pd.read_csv(DATASET_PATH, dtype={"case_id": str})
    slice_manifest = pd.read_csv(MANIFEST_PATH, dtype={"case_id": str})
    DATA_VERSION = {
        "dataset_csv": _sha1(DATASET_PATH),
        "manifest": _sha1(MANIFEST_PATH),
        "image_size": IMAGE_SIZE,
        "channels": 1,
    }
    _COMPONENTS.update({
        "df": df,
        "slice_manifest": slice_manifest,
        "DATA_VERSION": DATA_VERSION,
    })

    _load_notebook_definitions(
        ROOT / "proyecto_pengwin.ipynb",
        {
            "resolve_repo_path", "apply_hu_window", "enhance_bone_contrast",
            "instance_to_semantic", "boxes_from_semantic", "letterbox_pair",
            "build_detection_target", "PengwinDetectionDataset", "FundidoraPC",
            "FundidoraPCPlus", "build_backbone", "build_instance_segmentation_targets",
            "build_detection_region_priors", "SegmentationConvBlock",
            "backbone_feature_pyramid", "SkipFusionBlock", "CenterOffsetSkipDecoder",
            "decode_center_offset_instances",
        },
        {"CLASS_ID_TO_NAME", "TARGET_COLORS"},
    )
    _load_notebook_definitions(
        ROOT / "cbam.ipynb",
        {"ChannelAttention", "SpatialAttention", "CBAM", "CompuertaResidualCBAM"},
    )
    _load_notebook_definitions(
        ROOT / "entrenamiento_multitarea.ipynb",
        {
            "_VolumeSlices", "LeanPengwinDetectionDataset", "build_or_load_cache",
            "_read_cache", "pad_boxes", "unpad_boxes", "CachedSliceDataset",
            "collate_slices", "MultiTaskNet",
        },
    )
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    try:
        from deteccion_grid import CLASS_NAMES, GridDetectionHead, postprocess
    except ModuleNotFoundError as exc:
        raise FileNotFoundError(f"No se pudo importar deteccion_grid.py desde {ROOT}") from exc
    _COMPONENTS["GridDetectionHead"] = GridDetectionHead
    _COMPONENTS["postprocess"] = postprocess
    _COMPONENTS["CLASS_NAMES"] = CLASS_NAMES


def _resolve_repo_path(path: Path, description: str) -> Path:
    candidate = path if path.is_absolute() else ROOT / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"La ruta de {description} debe estar dentro de la raiz del repositorio.") from exc
    return resolved


def _verify_class_convention() -> None:
    expected = {
        1: "sacro",
        2: "coxal_izquierdo",
        3: "coxal_derecho",
    }
    actual = {
        "targets": _COMPONENTS["CLASS_ID_TO_NAME"],
        "detection_head": _COMPONENTS["CLASS_NAMES"],
    }
    for source, mapping in actual.items():
        if mapping != expected:
            raise ValueError(
                f"Convencion de clases inesperada en {source}: {mapping!r}; "
                f"se requiere {expected!r}."
            )


def _load_callable(specification: str) -> Callable[..., Any]:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("El NMS debe indicarse como 'modulo:funcion'.")
    module = importlib.import_module(module_name)
    candidate = getattr(module, attribute, None)
    if not callable(candidate):
        raise TypeError(f"{specification!r} no identifica una funcion NMS invocable.")
    return candidate


def _load_sam(checkpoint: Path, model_type: str, device: torch.device) -> Any:
    try:
        segment_anything = importlib.import_module("segment_anything")
    except ModuleNotFoundError as exc:
        if exc.name != "segment_anything":
            raise
        raise ImportError(
            "Para evaluar SAM instala el paquete Segment Anything en el ambiente activo."
        ) from exc
    sam_model_registry = segment_anything.sam_model_registry
    if model_type not in sam_model_registry:
        raise ValueError(f"Tipo SAM no reconocido: {model_type}")
    predictor = segment_anything.SamPredictor(
        sam_model_registry[model_type](checkpoint=str(checkpoint))
    )
    predictor.model.to(device=device)
    predictor.model.eval()
    return predictor


def _load_model(checkpoint: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"No existe el checkpoint: {checkpoint}")
    torch_version_type = getattr(getattr(torch, "torch_version", None), "TorchVersion", None)
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    if torch_version_type is None or safe_globals is None:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
    else:
        with safe_globals([torch_version_type]):
            payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("El checkpoint debe contener un diccionario de entrenamiento.")
    selected = payload.get("selected_from")
    if not isinstance(selected, dict):
        selected = {}
    stage = selected.get("stage", payload.get("stage"))
    if stage != "C_conjunto":
        raise ValueError(
            f"Se requiere el checkpoint completo de la etapa C_conjunto; el archivo indica {stage!r}."
        )
    architecture = payload.get("architecture")
    if architecture is None:
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ValueError("El checkpoint no contiene arquitectura ni configuracion de entrenamiento.")
        architecture = {
            "backbone_name": config["BACKBONE_NAME"],
            "use_cbam": config["USE_CBAM"],
            "use_gate": config.get("USE_CBAM_GATE", True),
            "use_skips": config.get("USE_SKIPS", True),
        }
    if architecture.get("image_size", IMAGE_SIZE) != IMAGE_SIZE:
        raise ValueError(f"El checkpoint usa image_size={architecture['image_size']}, se esperaba {IMAGE_SIZE}.")
    if architecture.get("grid_size", GRID_SIZE) != GRID_SIZE:
        raise ValueError(f"El checkpoint usa grid_size={architecture['grid_size']}, se esperaba {GRID_SIZE}.")
    model = _COMPONENTS["MultiTaskNet"](
        architecture["backbone_name"],
        architecture["use_cbam"],
        architecture.get("use_gate", True),
        architecture.get("use_skips", True),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_box = max(box[2] - box[0], 0) * max(box[3] - box[1], 0)
    area_boxes = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    return intersection / np.maximum(area_box + area_boxes - intersection, 1e-9)


def _average_precision(
    predictions: list[dict[str, np.ndarray]],
    targets: list[dict[str, np.ndarray]],
    class_id: int,
    iou_threshold: float,
) -> float | None:
    ground_truth: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    detections: list[tuple[float, int, np.ndarray]] = []
    total_ground_truth = 0
    for image_index, (prediction, target) in enumerate(zip(predictions, targets)):
        labels = target["labels"].astype(int)
        boxes = target["boxes"].reshape(-1, 4)
        boxes = boxes[labels == class_id]
        ground_truth[image_index] = (boxes, np.zeros(len(boxes), dtype=bool))
        total_ground_truth += len(boxes)
        keep = (prediction["labels"] == class_id)
        detections.extend(
            (float(score), image_index, box)
            for score, box in zip(prediction["scores"][keep], prediction["boxes"][keep])
        )
    if total_ground_truth == 0:
        return None
    detections.sort(key=lambda item: -item[0])
    true_positive = np.zeros(len(detections))
    false_positive = np.zeros(len(detections))
    for index, (_, image_index, box) in enumerate(detections):
        boxes, used = ground_truth[image_index]
        if not len(boxes):
            false_positive[index] = 1
            continue
        ious = _box_iou(box, boxes)
        best = int(ious.argmax())
        if ious[best] >= iou_threshold and not used[best]:
            used[best] = True
            true_positive[index] = 1
        else:
            false_positive[index] = 1
    true_positive = np.cumsum(true_positive)
    false_positive = np.cumsum(false_positive)
    recall = true_positive / total_ground_truth
    precision = true_positive / np.maximum(true_positive + false_positive, 1e-9)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    changes = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[changes + 1] - mrec[changes]) * mpre[changes + 1]))


def _detection_metrics(
    predictions: list[dict[str, np.ndarray]],
    targets: list[dict[str, np.ndarray]],
) -> dict[str, float]:
    # IoU promedio: para cada caja GT se usa la prediccion de MAYOR SCORE de su clase (no mira la
    # geometria del GT para elegirla). La variante "mejor candidata" es optimista y se reporta aparte.
    matched_ious, best_candidate_ious = [], []
    for prediction, target in zip(predictions, targets):
        for box, label in zip(target["boxes"], target["labels"]):
            same_class = np.flatnonzero(prediction["labels"] == label)
            if not len(same_class):
                matched_ious.append(0.0)
                best_candidate_ious.append(0.0)
                continue
            ious = _box_iou(box, prediction["boxes"][same_class])
            top_score = int(np.argmax(prediction["scores"][same_class]))
            matched_ious.append(float(ious[top_score]))
            best_candidate_ious.append(float(ious.max()))
    ap50_values = [
        value for class_id in CLASS_IDS
        if (value := _average_precision(predictions, targets, class_id, 0.50)) is not None
    ]
    ap_values = [
        value
        for threshold in np.arange(0.50, 0.951, 0.05)
        for class_id in CLASS_IDS
        if (value := _average_precision(predictions, targets, class_id, float(threshold))) is not None
    ]
    return {
        "bbox_mean_iou": float(np.mean(matched_ious)) if matched_ious else math.nan,
        "bbox_mean_iou_best_candidate": (
            float(np.mean(best_candidate_ious)) if best_candidate_ious else math.nan
        ),
        "bbox_map50": float(np.mean(ap50_values)) if ap50_values else math.nan,
        "bbox_map50_95": float(np.mean(ap_values)) if ap_values else math.nan,
    }


def _classification_metrics(
    predictions: list[dict[str, np.ndarray]],
    targets: list[dict[str, np.ndarray]],
    score_threshold: float,
) -> dict[str, float]:
    y_true = np.zeros((len(targets), len(CLASS_IDS)), dtype=np.uint8)
    y_score = np.zeros_like(y_true, dtype=np.float64)
    y_pred = np.zeros_like(y_true, dtype=np.uint8)
    for image_index, (prediction, target) in enumerate(zip(predictions, targets)):
        for class_index, class_id in enumerate(CLASS_IDS):
            y_true[image_index, class_index] = np.any(target["labels"] == class_id)
            scores = prediction["scores"][prediction["labels"] == class_id]
            if len(scores):
                y_score[image_index, class_index] = float(scores.max())
                y_pred[image_index, class_index] = scores.max() >= score_threshold
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    aucs = [
        roc_auc_score(y_true[:, index], y_score[:, index])
        for index in range(len(CLASS_IDS))
        if np.unique(y_true[:, index]).size == 2
    ]
    return {
        "classification_f1_macro": float(f1_macro),
        "classification_f1_micro": float(f1_micro),
        "classification_auc_macro": float(np.mean(aucs)) if aucs else math.nan,
    }


def _semantic_metrics(
    counts: dict[str, dict[int, int]],
    model_name: str,
) -> dict[str, float]:
    dice_values, iou_values = [], []
    for class_id in CLASS_IDS:
        values = counts[model_name][class_id]
        denominator = values["pred"] + values["gt"]
        union = denominator - values["intersection"]
        if denominator:
            dice_values.append(2 * values["intersection"] / denominator)
        if union:
            iou_values.append(values["intersection"] / union)
    return {
        f"{model_name}_dice_macro": float(np.mean(dice_values)) if dice_values else math.nan,
        f"{model_name}_iou_macro": float(np.mean(iou_values)) if iou_values else math.nan,
    }


def _fragment_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    prediction_classes: dict[int, int],
) -> tuple[float, float, int]:
    gt_ids = [int(value) for value in np.unique(ground_truth) if value != 0]
    pred_ids = [int(value) for value in np.unique(prediction) if value != 0]
    if not gt_ids:
        return 0.0, 0.0, 0
    if not pred_ids:
        return 0.0, 0.0, len(gt_ids)

    overlaps = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
    gt_areas = np.array([(ground_truth == value).sum() for value in gt_ids])
    pred_areas = np.array([(prediction == value).sum() for value in pred_ids])
    for row, gt_id in enumerate(gt_ids):
        gt_class = 1 if gt_id <= 10 else 2 if gt_id <= 20 else 3
        for column, pred_id in enumerate(pred_ids):
            if prediction_classes.get(pred_id) != gt_class:
                continue
            intersection = np.logical_and(ground_truth == gt_id, prediction == pred_id).sum()
            union = gt_areas[row] + pred_areas[column] - intersection
            overlaps[row, column] = intersection / union if union else 0.0

    matched_gt, matched_pred = linear_sum_assignment(-overlaps)
    matched_ious = {
        int(gt_index): float(overlaps[gt_index, pred_index])
        for gt_index, pred_index in zip(matched_gt, matched_pred)
    }
    ious = [matched_ious.get(index, 0.0) for index in range(len(gt_ids))]
    dices = [2 * value / (1 + value) if value else 0.0 for value in ious]
    return float(np.sum(dices)), float(np.sum(ious)), len(gt_ids)


def _semantic_from_instances(instance_mask: np.ndarray) -> np.ndarray:
    semantic = np.zeros(instance_mask.shape, dtype=np.uint8)
    semantic[(instance_mask >= 1) & (instance_mask <= 10)] = 1
    semantic[(instance_mask >= 11) & (instance_mask <= 20)] = 2
    semantic[(instance_mask >= 21) & (instance_mask <= 30)] = 3
    return semantic


def _add_semantic_counts(
    counts: dict[str, dict[int, dict[str, int]]],
    model_name: str,
    prediction: np.ndarray,
    target: np.ndarray,
) -> None:
    for class_id in CLASS_IDS:
        pred_mask, gt_mask = prediction == class_id, target == class_id
        counts[model_name][class_id]["intersection"] += int(np.logical_and(pred_mask, gt_mask).sum())
        counts[model_name][class_id]["pred"] += int(pred_mask.sum())
        counts[model_name][class_id]["gt"] += int(gt_mask.sum())


def _sam_semantic(
    predictor: Any,
    image: np.ndarray,
    detections: dict[str, torch.Tensor],
) -> np.ndarray:
    rgb = np.repeat(
        np.clip(image[0], 0, 1)[..., None] * 255, 3, axis=2
    ).astype(np.uint8)
    predictor.set_image(rgb)
    semantic = np.zeros(image.shape[-2:], dtype=np.uint8)
    best_score = np.full(image.shape[-2:], -np.inf, dtype=np.float32)
    boxes = detections["boxes"].detach().cpu().numpy()
    labels = detections["labels"].detach().cpu().numpy()
    for box, class_id in zip(boxes, labels):
        masks, scores, _ = predictor.predict(
            box=box.astype(np.float32),
            multimask_output=False,
        )
        mask = masks[0].astype(bool)
        score = float(scores[0])
        update = mask & (score > best_score)
        semantic[update] = int(class_id)
        best_score[update] = score
    return semantic


def _macro_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    scores = []
    for class_id in CLASS_IDS:
        pred_mask, gt_mask = prediction == class_id, target == class_id
        if gt_mask.any():
            scores.append(2 * np.logical_and(pred_mask, gt_mask).sum() / (pred_mask.sum() + gt_mask.sum()))
    return float(np.mean(scores)) if scores else math.nan


def _save_example(
    path: Path,
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    sam_prediction: np.ndarray | None,
    title: str,
) -> None:
    panels = [("Referencia", ground_truth), ("Modelo", prediction)]
    if sam_prediction is not None:
        panels.append(("SAM", sam_prediction))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    for axis, (panel_title, mask) in zip(np.atleast_1d(axes), panels):
        axis.imshow(image[0], cmap="gray", vmin=0, vmax=1)
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        for class_id, color in REGION_COLORS.items():
            selected = mask == class_id
            overlay[selected, :3] = color
            overlay[selected, 3] = 0.55
        axis.imshow(overlay)
        axis.set_title(panel_title)
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _run_pipeline(
    model: nn.Module,
    images: torch.Tensor,
    nms_fn: Callable[..., Any],
    args: argparse.Namespace,
    params: dict[str, Any],
):
    """Inferencia completa de un lote: deteccion -> NMS -> segmentacion -> instancias.

    Es la UNICA definicion del pipeline: la usan tanto la evaluacion como la medicion de latencia,
    para que ambas midan exactamente lo mismo.
    """
    center_threshold = params["center_threshold"]
    min_center_distance = params["min_center_distance"]
    min_instance_pixels = params["min_instance_pixels"]
    pred_box_score_min = params["pred_box_score_min"]
    with torch.inference_mode():
        features = model.trunk(images)
        logits = model.det_head(features[-1])
        detections = _COMPONENTS["postprocess"](
            logits.float(),
            score_thresh=args.score_threshold,
            nms_fn=nms_fn,
            iou_thresh=args.nms_iou,
        )
        segmentation_detections = []
        for detection in detections:
            keep = detection["scores"] >= pred_box_score_min
            segmentation_detections.append({
                key: value[keep] if key in {"boxes", "scores", "labels", "cells"} else value
                for key, value in detection.items()
            })
        priors = _COMPONENTS["build_detection_region_priors"](
            segmentation_detections,
            output_size=(IMAGE_SIZE, IMAGE_SIZE),
            device=DEVICE,
        )
        segmentation = model.seg_decoder(images, features, priors)
        semantic_predictions = segmentation["semantic_logits"].float().argmax(1).cpu().numpy()
        instance_predictions = [
            _COMPONENTS["decode_center_offset_instances"](
                {key: value[index:index + 1].float() for key, value in segmentation.items()},
                center_threshold=center_threshold,
                min_center_distance=min_center_distance,
                min_instance_pixels=min_instance_pixels,
            )
            for index in range(len(detections))
        ]
    return detections, segmentation_detections, semantic_predictions, instance_predictions


def _measure_latency(
    model: nn.Module,
    dataset: Any,
    nms_fn: Callable[..., Any],
    args: argparse.Namespace,
    params: dict[str, Any],
    sam_predictor: Any,
    warmup: int = 5,
) -> dict[str, Any]:
    """Latencia por imagen (batch=1), extremo a extremo y despues de calentar.

    Incluye tronco + cabeza de deteccion + NMS + segmentacion + decodificacion de instancias.
    No incluye lectura de disco ni la distancia de separacion. SAM se mide aparte (set_image + predict).
    """
    collate = _COMPONENTS["collate_slices"]
    total = min(args.latency_runs + warmup, len(dataset))
    warmup = min(warmup, max(total - 1, 0))
    pipeline_ms: list[float] = []
    sam_ms: list[float] = []

    def _sync() -> None:
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)

    for index in range(total):
        batch = collate([dataset[index]])
        images = batch["image"].to(DEVICE)
        _sync()
        start = time.perf_counter()
        _, segmentation_detections, _, _ = _run_pipeline(model, images, nms_fn, args, params)
        _sync()
        elapsed = (time.perf_counter() - start) * 1000
        sam_elapsed = None
        if sam_predictor is not None:
            start = time.perf_counter()
            _sam_semantic(sam_predictor, batch["image"][0].numpy(), segmentation_detections[0])
            _sync()
            sam_elapsed = (time.perf_counter() - start) * 1000
        if index >= warmup:
            pipeline_ms.append(elapsed)
            if sam_elapsed is not None:
                sam_ms.append(sam_elapsed)

    summary: dict[str, Any] = {
        "device": str(DEVICE),
        "batch_size": 1,
        "warmup_slices": warmup,
        "n_measured": len(pipeline_ms),
        "pipeline_ms_mean": float(np.mean(pipeline_ms)),
        "pipeline_ms_p50": float(np.percentile(pipeline_ms, 50)),
        "pipeline_ms_p95": float(np.percentile(pipeline_ms, 95)),
    }
    if sam_ms:
        summary.update({
            "sam_ms_mean": float(np.mean(sam_ms)),
            "sam_ms_p50": float(np.percentile(sam_ms, 50)),
            "sam_ms_p95": float(np.percentile(sam_ms, 95)),
        })
    return summary


def _check_label_convention(
    predictions: list[dict[str, np.ndarray]],
    targets: list[dict[str, np.ndarray]],
) -> None:
    """Falla de forma explicita si GT o predicciones no usan las clases 1, 2, 3.

    Si la cabeza o los targets usaran 0-2, las metricas saldrian mal SIN dar ningun error.
    """
    allowed = set(CLASS_IDS)
    gt_labels = {int(v) for target in targets for v in np.unique(target["labels"])}
    pred_labels = {int(v) for prediction in predictions for v in np.unique(prediction["labels"])}
    if (gt_labels - allowed) or (pred_labels - allowed):
        raise ValueError(
            f"Convencion de clases inconsistente: se esperan {sorted(allowed)}; "
            f"el GT trae {sorted(gt_labels)} y las predicciones {sorted(pred_labels)}. "
            "Revisa si la cabeza o los targets usan clases 0-2 en lugar de 1-3."
        )


def _keep_ranked(store: list, example: tuple, k: int, largest: bool) -> None:
    """Conserva solo los k mejores (largest=True) o peores ejemplos segun su puntaje (posicion 0)."""
    store.append(example)
    store.sort(key=lambda item: item[0], reverse=largest)
    del store[k:]


def _write_comparison_table(metrics: dict[str, Any], out_dir: Path, split: str) -> pd.DataFrame:
    """Tabla comparativa metrica | objetivo | modelo | SAM | cumple (CSV y Markdown)."""
    spec = [
        ("Clasificacion F1 macro", "classification_f1_macro", None, 0.85),
        ("Clasificacion AUC macro", "classification_auc_macro", None, 0.85),
        ("Deteccion IoU promedio", "bbox_mean_iou", None, 0.65),
        ("Deteccion mAP@0.50", "bbox_map50", None, 0.65),
        ("Deteccion mAP@[0.50:0.95]", "bbox_map50_95", None, 0.40),
        ("Segmentacion Dice por fragmento", "modelo_fragment_dice", None, 0.85),
        ("Segmentacion IoU por fragmento", "modelo_fragment_iou", None, 0.70),
        ("Dice semantico macro (modelo vs SAM)", "modelo_dice_macro", "sam_dice_macro", None),
        ("IoU semantico macro (modelo vs SAM)", "modelo_iou_macro", "sam_iou_macro", None),
    ]

    def fmt(value: Any) -> str:
        return "n/a" if value is None or not math.isfinite(float(value)) else f"{float(value):.3f}"

    rows = []
    for label, model_key, sam_key, minimum in spec:
        model_value = metrics.get(model_key)
        sam_value = metrics.get(sam_key) if sam_key else None
        met = (
            "" if minimum is None
            else ("si" if model_value is not None and math.isfinite(model_value) and model_value >= minimum else "no")
        )
        rows.append({
            "Metrica": label,
            "Objetivo": "" if minimum is None else f">= {minimum:.2f}",
            "Modelo": fmt(model_value),
            "SAM": fmt(sam_value) if sam_key else "-",
            "Cumple": met,
        })
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / f"tabla_comparativa_{split}.csv", index=False)
    header = "| " + " | ".join(table.columns) + " |\n|" + "---|" * len(table.columns) + "\n"
    body = "".join("| " + " | ".join(str(v) for v in row) + " |\n" for row in table.itertuples(index=False))
    (out_dir / f"tabla_comparativa_{split}.md").write_text(header + body, encoding="utf-8")
    return table


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    global DEVICE
    args.checkpoint = _resolve_repo_path(args.checkpoint, "checkpoint")
    if args.sam_checkpoint is not None:
        args.sam_checkpoint = _resolve_repo_path(args.sam_checkpoint, "checkpoint SAM")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicito CUDA, pero PyTorch no detecta una GPU disponible.")
    DEVICE = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    ))
    _COMPONENTS["DEVICE"] = DEVICE
    _load_project_components()
    _verify_class_convention()
    nms_fn = _load_callable(args.nms_function)
    model, checkpoint_payload = _load_model(args.checkpoint, DEVICE)
    sam_predictor = None
    if args.sam_checkpoint is not None:
        if args.split != "test":
            raise ValueError("La comparacion con SAM debe ejecutarse exclusivamente en split=test.")
        if not args.sam_checkpoint.is_file():
            raise FileNotFoundError(f"No existe el checkpoint SAM: {args.sam_checkpoint}")
        sam_predictor = _load_sam(args.sam_checkpoint, args.sam_type, DEVICE)

    inference_defaults = checkpoint_payload.get("inference_defaults")
    required_defaults = {
        "center_threshold",
        "min_center_distance",
        "min_instance_pixels",
        "pred_box_score_min",
    }
    if not isinstance(inference_defaults, dict):
        raise ValueError("El checkpoint no contiene el diccionario inference_defaults requerido.")
    missing_defaults = required_defaults - inference_defaults.keys()
    if missing_defaults:
        raise ValueError(
            "Faltan parametros de inferencia en el checkpoint: "
            f"{sorted(missing_defaults)}. No se usaran valores por defecto externos."
        )
    center_threshold = float(inference_defaults["center_threshold"])
    min_center_distance = int(inference_defaults["min_center_distance"])
    min_instance_pixels = int(inference_defaults["min_instance_pixels"])
    pred_box_score_min = float(inference_defaults["pred_box_score_min"])
    if not 0 <= center_threshold <= 1 or not 0 <= pred_box_score_min <= 1:
        raise ValueError("Los umbrales de inferencia del checkpoint deben estar entre 0 y 1.")
    if min_center_distance < 1 or min_instance_pixels < 1:
        raise ValueError("Los parametros de instancias del checkpoint deben ser positivos.")
    pipeline_params = {
        "center_threshold": center_threshold,
        "min_center_distance": min_center_distance,
        "min_instance_pixels": min_instance_pixels,
        "pred_box_score_min": pred_box_score_min,
    }
    cache = _COMPONENTS["build_or_load_cache"](args.split)
    dataset = _COMPONENTS["CachedSliceDataset"](cache)
    if len(dataset) == 0:
        raise ValueError(f"El manifiesto no contiene cortes para split={args.split!r}.")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_COMPONENTS["collate_slices"],
        pin_memory=DEVICE.type == "cuda",
    )
    counts: dict[str, dict[int, dict[str, int]]] = {
        name: {
            class_id: {"intersection": 0, "pred": 0, "gt": 0}
            for class_id in CLASS_IDS
        }
        for name in ("modelo", "sam")
    }
    all_detections: list[dict[str, np.ndarray]] = []
    all_targets: list[dict[str, np.ndarray]] = []
    slice_rows: list[dict[str, Any]] = []
    worst_examples: list[tuple] = []
    best_examples: list[tuple] = []
    elapsed_seconds = 0.0
    fragment_dice_sum = 0.0
    fragment_iou_sum = 0.0
    fragment_count = 0

    model.eval()
    for batch in loader:
        images = batch["image"].to(DEVICE, non_blocking=True)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)
        start = time.perf_counter()
        detections, segmentation_detections, semantic_predictions, instance_predictions = _run_pipeline(
            model, images, nms_fn, args, pipeline_params
        )
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)
        elapsed_seconds += time.perf_counter() - start

        for index, detection in enumerate(detections):
            predicted = {
                key: value.detach().cpu().numpy()
                for key, value in detection.items()
                if key in {"boxes", "scores", "labels"}
            }
            target = {
                "boxes": np.asarray(batch["boxes"][index], dtype=np.float32).reshape(-1, 4),
                "labels": np.asarray(batch["labels"][index], dtype=np.int64).reshape(-1),
            }
            all_detections.append(predicted)
            all_targets.append(target)
            gt_semantic = _semantic_from_instances(batch["instance_mask"][index].numpy())
            model_semantic = semantic_predictions[index].astype(np.uint8)
            gt_instances = batch["instance_mask"][index].numpy()
            pred_instances, _, pred_classes = instance_predictions[index]
            dice_sum, iou_sum, n_fragments = _fragment_metrics(
                gt_instances,
                pred_instances,
                pred_classes,
            )
            fragment_dice_sum += dice_sum
            fragment_iou_sum += iou_sum
            fragment_count += n_fragments
            _add_semantic_counts(counts, "modelo", model_semantic, gt_semantic)
            sam_semantic = None
            if sam_predictor is not None:
                sam_semantic = _sam_semantic(
                    sam_predictor,
                    batch["image"][index].numpy(),
                    segmentation_detections[index],
                )
                _add_semantic_counts(counts, "sam", sam_semantic, gt_semantic)

            score = _macro_dice(model_semantic, gt_semantic)
            example = (
                score,
                f"{batch['case_id'][index]}_slice_{int(batch['slice_index'][index]):04d}",
                batch["image"][index].numpy().copy(),
                gt_semantic.copy(),
                model_semantic.copy(),
                None if sam_semantic is None else sam_semantic.copy(),
            )
            if math.isfinite(score):   # un corte sin region GT daria NaN y bloquearia el ranking
                _keep_ranked(worst_examples, example, N_WORST_EXAMPLES, largest=False)
                _keep_ranked(best_examples, example, N_BEST_EXAMPLES, largest=True)
            slice_rows.append({
                "case_id": batch["case_id"][index],
                "slice_index": int(batch["slice_index"][index]),
                "n_gt_boxes": len(target["labels"]),
                "n_pred_boxes": len(predicted["labels"]),
                "model_region_dice": score,
                "model_fragment_dice": dice_sum / n_fragments if n_fragments else math.nan,
                "model_fragment_iou": iou_sum / n_fragments if n_fragments else math.nan,
                "sam_dice": _macro_dice(sam_semantic, gt_semantic) if sam_semantic is not None else math.nan,
            })

    _check_label_convention(all_detections, all_targets)

    metrics: dict[str, Any] = {
        "split": args.split,
        "checkpoint": args.checkpoint.relative_to(ROOT).as_posix(),
        "checkpoint_stage": checkpoint_payload.get("selected_from", {}).get(
            "stage", checkpoint_payload.get("stage")
        ),
        "device": str(DEVICE),
        "n_slices": len(dataset),
        "n_cases": int(cache["meta"]["case_id"].nunique()),
        "mean_inference_ms_per_slice_batched": elapsed_seconds * 1000 / len(dataset),  # rendimiento, no latencia
        "cls_threshold": args.cls_threshold,
        "score_threshold": args.score_threshold,
        "nms_iou": args.nms_iou,
        "evaluated_slices_note": "solo cortes positivos del manifiesto (con al menos una region GT)",
        "n_ground_truth_fragments": fragment_count,
        "modelo_fragment_dice": fragment_dice_sum / fragment_count if fragment_count else math.nan,
        "modelo_fragment_iou": fragment_iou_sum / fragment_count if fragment_count else math.nan,
        "nms_function": args.nms_function,
        "sam": "evaluated" if sam_predictor is not None else "not_requested",
    }
    metrics.update(_classification_metrics(all_detections, all_targets, args.cls_threshold))
    metrics.update(_detection_metrics(all_detections, all_targets))
    metrics.update(_semantic_metrics(counts, "modelo"))
    if sam_predictor is not None:
        metrics.update(_semantic_metrics(counts, "sam"))
        metrics["sam_checkpoint"] = args.sam_checkpoint.relative_to(ROOT).as_posix()
        metrics["sam_model_type"] = args.sam_type
    metrics["targets"] = {
        "classification_f1_macro": {"minimum": 0.85, "met": metrics["classification_f1_macro"] >= 0.85},
        "classification_auc_macro": {
            "minimum": 0.85,
            "met": math.isfinite(metrics["classification_auc_macro"])
            and metrics["classification_auc_macro"] >= 0.85,
        },
        "bbox_mean_iou": {"minimum": 0.65, "met": metrics["bbox_mean_iou"] >= 0.65},
        "bbox_map50": {"minimum": 0.65, "met": metrics["bbox_map50"] >= 0.65},
        "bbox_map50_95": {"minimum": 0.40, "met": metrics["bbox_map50_95"] >= 0.40},
        "modelo_fragment_dice": {
            "minimum": 0.85,
            "met": metrics["modelo_fragment_dice"] >= 0.85,
        },
        "modelo_fragment_iou": {
            "minimum": 0.70,
            "met": metrics["modelo_fragment_iou"] >= 0.70,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.latency_runs > 0:
        latency = _measure_latency(model, dataset, nms_fn, args, pipeline_params, sam_predictor)
        metrics["latency"] = latency
        (OUT_DIR / f"latencia_{args.split}_{DEVICE.type}.json").write_text(
            json.dumps(latency, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (OUT_DIR / f"metricas_{args.split}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    pd.DataFrame([metrics]).to_csv(OUT_DIR / f"metricas_{args.split}.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(OUT_DIR / f"cortes_{args.split}.csv", index=False)
    comparison = _write_comparison_table(metrics, OUT_DIR, args.split)

    for label, ranked in (("peor", worst_examples), ("acierto", best_examples)):
        for rank, example in enumerate(ranked, start=1):
            value, name, image, gt, prediction, sam_prediction = example
            _save_example(
                OUT_DIR / f"{args.split}_{label}{rank}_{name}.png",
                image,
                gt,
                prediction,
                sam_prediction,
                f"{label} #{rank}: {name} | Dice modelo={value:.3f}",
            )

    print(f"Evaluacion terminada: {len(dataset)} cortes de {metrics['n_cases']} casos.")
    print(f"Metricas: {OUT_DIR / f'metricas_{args.split}.json'}")
    if "latency" in metrics:
        lat = metrics["latency"]
        print(f"Latencia (batch=1, {DEVICE}): media {lat['pipeline_ms_mean']:.1f} ms | "
              f"p50 {lat['pipeline_ms_p50']:.1f} | p95 {lat['pipeline_ms_p95']:.1f} ms/corte.")
    print(comparison.to_string(index=False))
    if sam_predictor is None:
        print("SAM omitido: pasa --sam-checkpoint para ejecutar la comparacion zero-shot en test.")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--nms-function",
        required=True,
        help="Funcion NMS de Fabian como modulo:funcion (cajas, scores, clases, umbral_iou)->indices.",
    )
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--sam-type", choices=("vit_b", "vit_l", "vit_h"), default="vit_b")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--score-threshold", type=float, default=0.01,
                        help="Score minimo de las cajas que salen del postproceso (bajo, para el calculo de AP).")
    parser.add_argument("--cls-threshold", type=float, default=0.5,
                        help="Umbral de score para decidir si una region esta PRESENTE (F1 de clasificacion).")
    parser.add_argument("--latency-runs", type=int, default=50,
                        help="Cortes medidos con batch=1 para la latencia (0 la desactiva).")
    parser.add_argument("--nms-iou", type=float, default=0.5)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size debe ser positivo.")
    if not all(0 <= value <= 1 for value in (args.score_threshold, args.nms_iou, args.cls_threshold)):
        parser.error("Los umbrales deben estar entre 0 y 1.")
    if args.latency_runs < 0:
        parser.error("--latency-runs no puede ser negativo.")
    return args


if __name__ == "__main__":
    run_evaluation(parse_args())