"""
VisionExtract 2.0 — YOLOv8 Detection Engine
Modular, device-aware object detection with configurable thresholds.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger("visionextract.detection")


@dataclass
class DetectionResult:
    """Structured detection output for a single image."""
    boxes: np.ndarray          # (N, 4) — x1, y1, x2, y2
    scores: np.ndarray         # (N,)   — confidence scores
    class_ids: np.ndarray      # (N,)   — integer class indices
    labels: List[str]          # (N,)   — human-readable class names
    num_objects: int = 0

    # Raw tensors for downstream SAM batching (kept on device)
    boxes_tensor: Optional[torch.Tensor] = field(default=None, repr=False)
    scores_tensor: Optional[torch.Tensor] = field(default=None, repr=False)
    class_ids_tensor: Optional[torch.Tensor] = field(default=None, repr=False)


class DetectionEngine:
    """
    YOLOv8-based object detection engine.
    Supports configurable confidence/NMS thresholds and device-aware execution.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        logger.info(f"Loading YOLOv8 model from {model_path} on {device}...")
        self.model = YOLO(model_path)

        # Extract class name mapping
        try:
            self.class_names: Dict[int, str] = self.model.model.names
        except AttributeError:
            self.class_names = self.model.names

        logger.info(f"YOLOv8 loaded — {len(self.class_names)} classes available")

    def detect(
        self,
        image_rgb: np.ndarray,
        conf_threshold: float = 0.35,
        nms_iou: float = 0.45,
    ) -> DetectionResult:
        """
        Run YOLOv8 detection on an RGB image.

        Args:
            image_rgb: Input image as RGB numpy array
            conf_threshold: Minimum confidence to keep a detection
            nms_iou: NMS IoU threshold

        Returns:
            DetectionResult with boxes, scores, labels, and raw tensors
        """
        results = self.model(image_rgb, conf=conf_threshold, iou=nms_iou)[0]

        boxes_tensor = results.boxes.xyxy
        scores_tensor = results.boxes.conf
        class_ids_tensor = results.boxes.cls

        # Filter by confidence threshold (belt-and-suspenders)
        valid_mask = scores_tensor >= conf_threshold
        boxes_tensor = boxes_tensor[valid_mask]
        scores_tensor = scores_tensor[valid_mask]
        class_ids_tensor = class_ids_tensor[valid_mask]

        # CPU numpy copies for feature extraction / rendering
        boxes_np = boxes_tensor.int().cpu().numpy()
        scores_np = scores_tensor.cpu().numpy()
        class_ids_np = class_ids_tensor.cpu().numpy().astype(int)

        labels = [self.class_names.get(int(cid), str(cid)) for cid in class_ids_np]

        return DetectionResult(
            boxes=boxes_np,
            scores=scores_np,
            class_ids=class_ids_np,
            labels=labels,
            num_objects=len(boxes_np),
            boxes_tensor=boxes_tensor,
            scores_tensor=scores_tensor,
            class_ids_tensor=class_ids_tensor,
        )

    def detect_batch(
        self,
        images: List[np.ndarray],
        conf_threshold: float = 0.35,
        nms_iou: float = 0.45,
    ) -> List[DetectionResult]:
        """Run detection on a batch of images."""
        return [self.detect(img, conf_threshold, nms_iou) for img in images]

    @property
    def available_classes(self) -> Dict[int, str]:
        """Return the full class name dictionary."""
        return dict(self.class_names)
