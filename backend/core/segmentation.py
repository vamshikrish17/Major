"""
VisionExtract 2.0 — Segmentation Engine
SAM-based segmentation with batched tensor processing and hybrid semantic-aware prompts.
Includes optional U-Net integration for biomedical precision.
"""

import logging
import uuid
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

from .detection import DetectionResult

logger = logging.getLogger("visionextract.segmentation")


@dataclass
class SegmentInfo:
    """Metadata for a single segmented object."""
    mask: np.ndarray          # (H, W) boolean mask
    box: List[int]            # [x1, y1, x2, y2]
    label: str
    score: float
    iou_prediction: float = 0.0
    segment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class SegmentationResult:
    """Complete segmentation output."""
    segments: List[SegmentInfo]
    overlay_image: np.ndarray     # BGR image with colored overlays
    num_segments: int = 0
    processing_device: str = "cpu"


# ─────────────────────────────────────────────────────────────────
# Lightweight U-Net for Biomedical Segmentation
# ─────────────────────────────────────────────────────────────────
class _DoubleConv(nn.Module):
    """Two consecutive conv-batchnorm-relu blocks."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LightweightUNet(nn.Module):
    """
    Compact U-Net (4 levels) for pixel-level segmentation.
    Can be used standalone or as a refinement layer after SAM.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        # Encoder
        self.enc1 = _DoubleConv(in_channels, 64)
        self.enc2 = _DoubleConv(64, 128)
        self.enc3 = _DoubleConv(128, 256)
        self.enc4 = _DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = _DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = _DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = _DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = _DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = _DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder path with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.final(d1))


# ─────────────────────────────────────────────────────────────────
# Main Segmentation Engine
# ─────────────────────────────────────────────────────────────────
class SegmentationEngine:
    """
    Hybrid segmentation engine combining SAM (batched tensor processing
    with semantic-aware prompts) and optional U-Net refinement.
    """

    def __init__(
        self,
        sam_checkpoint: str,
        sam_model_type: str = "vit_b",
        device: str = "cpu",
        unet_checkpoint: Optional[str] = None,
    ):
        self.device = device

        # Load SAM
        logger.info(f"Loading SAM ({sam_model_type}) on {device}...")
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)
        self.auto_mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=24,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.9,
            min_mask_region_area=100,
        )
        logger.info("SAM loaded successfully")

        # Load U-Net (optional — for bio mode)
        self.unet: Optional[LightweightUNet] = None
        if unet_checkpoint and os.path.exists(unet_checkpoint):
            logger.info(f"Loading U-Net from {unet_checkpoint}...")
            self.unet = LightweightUNet()
            self.unet.load_state_dict(torch.load(unet_checkpoint, map_location=device))
            self.unet.to(device)
            self.unet.eval()
            logger.info("U-Net loaded successfully")
        else:
            logger.info("U-Net not loaded (no checkpoint provided) — SAM-only mode")

    def segment(
        self,
        image_rgb: np.ndarray,
        detection: DetectionResult,
        use_hybrid_prompts: bool = True,
        mode: str = "general",
    ) -> SegmentationResult:
        """
        Run SAM segmentation using detection results as prompts.

        Uses Batched Tensor Processing (Kim et al. 2025) with
        Semantic-Aware Hybrid Prompts (box + center-point) for
        optimal mask quality and speed.

        Args:
            image_rgb: RGB numpy array
            detection: DetectionResult from DetectionEngine
            use_hybrid_prompts: Use box+point prompts (True) or box-only (False)

        Returns:
            SegmentationResult with masks, overlay, and segment metadata
        """
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        overlay = image_bgr.astype(np.float32)
        segments: List[SegmentInfo] = []

        if detection.num_objects == 0:
            auto_segments = self._segment_without_detections(image_rgb, overlay)
            return SegmentationResult(
                segments=auto_segments["segments"],
                overlay_image=auto_segments["overlay_image"],
                num_segments=len(auto_segments["segments"]),
                processing_device=self.device,
            )

        # Set SAM image embedding (computed once)
        self.predictor.set_image(image_rgb)

        boxes = detection.boxes_tensor
        h, w = image_rgb.shape[:2]

        # ── Batched Tensor Processing (All objects in one forward pass) ──
        batch_boxes = self.predictor.transform.apply_boxes_torch(
            boxes, image_rgb.shape[:2]
        ).to(self.device)

        if use_hybrid_prompts:
            # Semantic-Aware Hybrid Prompts: Box + Center-of-Mass Point
            points = torch.zeros((len(boxes), 1, 2), device=self.device)
            points[:, 0, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # Center X
            points[:, 0, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # Center Y
            batch_points = self.predictor.transform.apply_coords_torch(
                points, image_rgb.shape[:2]
            ).to(self.device)
            batch_labels = torch.ones((len(boxes), 1), dtype=torch.int, device=self.device)
        else:
            batch_points = None
            batch_labels = None

        # ── Execute batched prediction (zero looping across SAM instances) ──
        masks, iou_preds, _ = self.predictor.predict_torch(
            point_coords=batch_points,
            point_labels=batch_labels,
            boxes=batch_boxes,
            multimask_output=False,
        )

        # ── Offload to CPU ──
        masks_np = masks[:, 0, :, :].cpu().numpy()
        iou_preds_np = iou_preds[:, 0].cpu().numpy()
        boxes_np = detection.boxes
        scores_np = detection.scores

        unet_mask = None
        if mode == "bio" and self.unet is not None:
            unet_mask = self.segment_unet(image_rgb)

        # ── Build segment metadata + render overlay ──
        for i in range(len(masks_np)):
            mask = masks_np[i]
            if unet_mask is not None:
                mask = self._refine_mask_with_unet(mask.astype(bool), unet_mask, boxes_np[i])
            if mask.sum() < 10:
                continue

            x1, y1, x2, y2 = boxes_np[i]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            seg_info = SegmentInfo(
                mask=mask.astype(bool),
                box=[int(x1), int(y1), int(x2), int(y2)],
                label=detection.labels[i],
                score=float(scores_np[i]),
                iou_prediction=float(iou_preds_np[i]),
            )
            segments.append(seg_info)

            # Colored overlay for visualization
            color = (random.randint(60, 255), random.randint(60, 255), random.randint(60, 255))
            alpha = 0.55
            mask_bool = mask.astype(bool)
            overlay[mask_bool] = (
                (1 - alpha) * overlay[mask_bool]
                + alpha * np.array(color, dtype=np.float32)
            )

            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                overlay,
                f"{detection.labels[i]} {scores_np[i]:.2f}",
                (int(x1), max(0, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        return SegmentationResult(
            segments=segments,
            overlay_image=overlay,
            num_segments=len(segments),
            processing_device=self.device,
        )

    def _segment_without_detections(
        self,
        image_rgb: np.ndarray,
        overlay: np.ndarray,
        max_segments: int = 6,
    ) -> Dict[str, object]:
        """
        Fallback segmentation path for arbitrary uploads when detector finds nothing.
        Uses SAM automatic mask generation and returns the largest stable regions.
        """
        try:
            proposals = self.auto_mask_generator.generate(image_rgb)
        except Exception as exc:
            logger.warning(
                "Automatic SAM mask generation failed on %s, using prompt fallback: %s",
                self.device,
                exc,
            )
            return self._segment_full_image_with_prompts(image_rgb, overlay, max_segments=max_segments)

        if not proposals:
            return self._segment_full_image_with_prompts(image_rgb, overlay, max_segments=max_segments)

        proposals = sorted(proposals, key=lambda item: item.get("area", 0), reverse=True)
        segments: List[SegmentInfo] = []

        for proposal in proposals[:max_segments]:
            mask = proposal["segmentation"].astype(bool)
            if mask.sum() < 100:
                continue

            x, y, w, h = proposal["bbox"]
            x2 = x + w
            y2 = y + h
            score = float(proposal.get("predicted_iou", proposal.get("stability_score", 0.5)))
            seg_info = SegmentInfo(
                mask=mask,
                box=[int(x), int(y), int(x2), int(y2)],
                label="auto-segment",
                score=score,
                iou_prediction=score,
            )
            segments.append(seg_info)

            color = (random.randint(60, 255), random.randint(60, 255), random.randint(60, 255))
            alpha = 0.5
            overlay[mask] = (
                (1 - alpha) * overlay[mask] +
                alpha * np.array(color, dtype=np.float32)
            )
            cv2.rectangle(overlay, (int(x), int(y)), (int(x2), int(y2)), color, 2)
            cv2.putText(
                overlay,
                f"segment {score:.2f}",
                (int(x), max(0, int(y) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        return {
            "segments": segments,
            "overlay_image": np.clip(overlay, 0, 255).astype(np.uint8),
        }

    def _segment_full_image_with_prompts(
        self,
        image_rgb: np.ndarray,
        overlay: np.ndarray,
        max_segments: int = 3,
    ) -> Dict[str, object]:
        """
        Device-safe SAM fallback for arbitrary uploads.
        Uses a full-image box with a few spatial prompts instead of the
        automatic mask generator, which is unstable on some MPS setups.
        """
        self.predictor.set_image(image_rgb)
        h, w = image_rgb.shape[:2]

        box = np.array([0, 0, w - 1, h - 1], dtype=np.float32)
        point_coords = np.array(
            [
                [w * 0.5, h * 0.5],
                [w * 0.3, h * 0.3],
                [w * 0.7, h * 0.7],
            ],
            dtype=np.float32,
        )
        point_labels = np.ones((point_coords.shape[0],), dtype=np.int32)

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box[None, :],
            multimask_output=True,
        )

        segments: List[SegmentInfo] = []
        ranked = sorted(
            zip(masks, scores),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )

        for mask, score in ranked[:max_segments]:
            mask_bool = mask.astype(bool)
            if mask_bool.sum() < 100:
                continue

            ys, xs = np.where(mask_bool)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            seg_info = SegmentInfo(
                mask=mask_bool,
                box=[x1, y1, x2, y2],
                label="auto-segment",
                score=float(score),
                iou_prediction=float(score),
            )
            segments.append(seg_info)

            color = (random.randint(60, 255), random.randint(60, 255), random.randint(60, 255))
            alpha = 0.5
            overlay[mask_bool] = (
                (1 - alpha) * overlay[mask_bool] +
                alpha * np.array(color, dtype=np.float32)
            )
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                overlay,
                f"segment {score:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        return {
            "segments": segments,
            "overlay_image": np.clip(overlay, 0, 255).astype(np.uint8),
        }

    def _refine_mask_with_unet(
        self,
        sam_mask: np.ndarray,
        unet_mask: np.ndarray,
        box: np.ndarray,
    ) -> np.ndarray:
        """Fuse SAM and U-Net masks inside the detected ROI for biological mode."""
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(unet_mask.shape[1] - 1, x2)
        y2 = min(unet_mask.shape[0] - 1, y2)

        roi = np.zeros_like(sam_mask, dtype=bool)
        roi[y1:y2 + 1, x1:x2 + 1] = True
        unet_roi = unet_mask & roi

        overlap = np.logical_and(sam_mask, unet_roi).sum()
        if overlap > 0:
            return np.logical_or(sam_mask, unet_roi)
        return sam_mask

    def segment_unet(
        self,
        image_rgb: np.ndarray,
        threshold: float = 0.5,
    ) -> Optional[np.ndarray]:
        """
        Run U-Net pixel-level segmentation (for bio mode refinement).
        Returns binary mask or None if U-Net not available.
        """
        if self.unet is None:
            return None

        h, w = image_rgb.shape[:2]

        # Normalize and pad to multiple of 16
        img_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Resize to U-Net input size
        img_resized = torch.nn.functional.interpolate(
            img_tensor, size=(256, 256), mode="bilinear", align_corners=False
        )

        with torch.no_grad():
            pred = self.unet(img_resized)

        # Resize back and threshold
        pred_full = torch.nn.functional.interpolate(
            pred, size=(h, w), mode="bilinear", align_corners=False
        )
        mask = (pred_full.squeeze().cpu().numpy() > threshold).astype(bool)
        return mask


def save_segment_png(
    original_bgr: np.ndarray,
    segment: SegmentInfo,
    output_dir: str,
) -> str:
    """
    Save a single segment as a transparent PNG (object only, no background).
    Returns the filename of the saved segment.
    """
    h, w = original_bgr.shape[:2]
    x1, y1, x2, y2 = segment.box

    # Build 4-channel BGRA image
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask = segment.mask
    rgba[mask, 0:3] = original_bgr[mask]
    rgba[mask, 3] = 255

    # Crop to bounding box
    cropped = rgba[y1:y2 + 1, x1:x2 + 1]

    filename = f"{segment.segment_id}_{segment.label}.png"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, cropped)

    return filename
