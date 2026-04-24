"""
VisionExtract 2.0 — Evaluation Metrics
Accuracy, precision, recall, IoU, and performance profiling utilities.
"""

import time
import functools
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("visionextract.metrics")


@dataclass
class SegmentationMetrics:
    """Segmentation quality metrics for a single mask."""
    iou: float = 0.0
    dice: float = 0.0
    pixel_accuracy: float = 0.0
    boundary_f1: float = 0.0


@dataclass
class DetectionMetrics:
    """Detection evaluation metrics."""
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mean_iou: float = 0.0
    mean_confidence: float = 0.0
    num_detections: int = 0

    def to_dict(self) -> Dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "mean_iou": round(self.mean_iou, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "num_detections": self.num_detections,
        }


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Compute Intersection over Union between two binary masks.
    """
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / max(union, 1))


def compute_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """
    Compute Dice coefficient (F1 for segmentation).
    """
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(2 * intersection / max(mask_a.sum() + mask_b.sum(), 1))


def compute_pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute pixel-wise accuracy."""
    correct = (pred == gt).sum()
    return float(correct / max(pred.size, 1))


def compute_boundary_f1(mask_a: np.ndarray, mask_b: np.ndarray, tolerance: int = 2) -> float:
    """
    Compute boundary F1 score between two masks.
    Measures how well the predicted boundary matches the ground truth boundary.
    """
    import cv2

    # Extract boundaries
    kernel = np.ones((3, 3), np.uint8)
    boundary_a = cv2.dilate(mask_a.astype(np.uint8), kernel) - mask_a.astype(np.uint8)
    boundary_b = cv2.dilate(mask_b.astype(np.uint8), kernel) - mask_b.astype(np.uint8)

    # Dilate boundaries by tolerance
    tol_kernel = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    boundary_a_dilated = cv2.dilate(boundary_a, tol_kernel)
    boundary_b_dilated = cv2.dilate(boundary_b, tol_kernel)

    # Precision and recall for boundaries
    if boundary_a.sum() == 0 or boundary_b.sum() == 0:
        return 0.0

    precision = float((boundary_a & boundary_b_dilated).sum() / max(boundary_a.sum(), 1))
    recall = float((boundary_b & boundary_a_dilated).sum() / max(boundary_b.sum(), 1))

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def compute_detection_metrics(
    iou_predictions: List[float],
    confidences: List[float],
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """
    Compute detection-level precision, recall, F1 from IoU predictions.
    Objects with IoU >= threshold are considered true positives.
    """
    if not iou_predictions:
        return DetectionMetrics()

    ious = np.array(iou_predictions)
    confs = np.array(confidences)

    tp = int(np.sum(ious >= iou_threshold))
    total = len(ious)

    precision = tp / max(total, 1)
    recall = tp / max(total, 1)  # Without ground truth, recall = same as precision
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1,
        mean_iou=float(ious.mean()),
        mean_confidence=float(confs.mean()),
        num_detections=total,
    )


def compare_models(
    baseline_metrics: DetectionMetrics,
    hybrid_metrics: DetectionMetrics,
) -> Dict:
    """
    Compare baseline vs hybrid model metrics.
    Returns improvement percentages.
    """
    def _pct_change(old, new):
        if old == 0:
            return 0.0
        return round(((new - old) / old) * 100, 2)

    return {
        "precision_change_pct": _pct_change(baseline_metrics.precision, hybrid_metrics.precision),
        "recall_change_pct": _pct_change(baseline_metrics.recall, hybrid_metrics.recall),
        "f1_change_pct": _pct_change(baseline_metrics.f1_score, hybrid_metrics.f1_score),
        "iou_change_pct": _pct_change(baseline_metrics.mean_iou, hybrid_metrics.mean_iou),
        "baseline": baseline_metrics.to_dict(),
        "hybrid": hybrid_metrics.to_dict(),
    }


class PerformanceTimer:
    """Context manager for timing pipeline stages."""

    def __init__(self):
        self.stages: Dict[str, float] = {}
        self._current_stage: Optional[str] = None
        self._start_time: float = 0

    def start(self, stage: str):
        self._current_stage = stage
        self._start_time = time.perf_counter()

    def stop(self):
        if self._current_stage:
            elapsed = (time.perf_counter() - self._start_time) * 1000  # ms
            self.stages[self._current_stage] = round(elapsed, 2)
            self._current_stage = None

    @property
    def total_ms(self) -> float:
        return round(sum(self.stages.values()), 2)

    def to_dict(self) -> Dict:
        return {
            "total_ms": self.total_ms,
            **{f"{k}_ms": v for k, v in self.stages.items()},
        }
