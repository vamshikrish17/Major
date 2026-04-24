"""
VisionExtract 2.0 — Feature Extraction Engine
Computes scientific morphological attributes from segmentation masks.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("visionextract.features")


@dataclass
class ObjectFeatures:
    """Complete morphological feature set for a single segmented object."""
    segment_id: str
    label: str
    confidence: float

    # Spatial
    area_px: float = 0.0
    perimeter_px: float = 0.0
    bbox: List[int] = field(default_factory=list)          # [x1, y1, x2, y2]
    centroid: Tuple[float, float] = (0.0, 0.0)             # (cx, cy)
    bbox_width: float = 0.0
    bbox_height: float = 0.0

    # Shape descriptors
    circularity: float = 0.0        # 4π·Area / Perimeter² (1.0 = perfect circle)
    aspect_ratio: float = 0.0       # width / height of bounding rect
    solidity: float = 0.0           # Area / ConvexHullArea
    eccentricity: float = 0.0       # minor/major axis ratio of fitted ellipse
    convexity: float = 0.0          # ConvexHullPerimeter / Perimeter
    orientation: float = 0.0        # angle of fitted ellipse (degrees)
    equivalent_diameter: float = 0.0  # √(4·Area / π)
    extent: float = 0.0            # Area / BoundingRectArea
    compactness: float = 0.0        # Perimeter² / Area
    shape: str = "unknown"

    # Intensity (computed from original image)
    mean_intensity: float = 0.0
    std_intensity: float = 0.0
    min_intensity: float = 0.0
    max_intensity: float = 0.0

    def to_dict(self) -> Dict:
        """Serialize to JSON-compatible dictionary."""
        return {
            "segment_id": self.segment_id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "area_px": round(self.area_px, 2),
            "perimeter_px": round(self.perimeter_px, 2),
            "centroid_x": round(self.centroid[0], 2),
            "centroid_y": round(self.centroid[1], 2),
            "bbox_width": round(self.bbox_width, 2),
            "bbox_height": round(self.bbox_height, 2),
            "circularity": round(self.circularity, 4),
            "aspect_ratio": round(self.aspect_ratio, 4),
            "solidity": round(self.solidity, 4),
            "eccentricity": round(self.eccentricity, 4),
            "convexity": round(self.convexity, 4),
            "orientation": round(self.orientation, 2),
            "equivalent_diameter": round(self.equivalent_diameter, 2),
            "extent": round(self.extent, 4),
            "compactness": round(self.compactness, 2),
            "shape": self.shape,
            "mean_intensity": round(self.mean_intensity, 2),
            "std_intensity": round(self.std_intensity, 2),
            "min_intensity": round(self.min_intensity, 2),
            "max_intensity": round(self.max_intensity, 2),
        }


@dataclass
class FeatureExtractionResult:
    """Aggregate feature extraction output."""
    objects: List[ObjectFeatures]
    num_objects: int = 0
    total_area_px: float = 0.0
    mean_area_px: float = 0.0
    area_std_px: float = 0.0
    size_distribution: Dict[str, int] = field(default_factory=dict)  # small/medium/large
    label_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "num_objects": self.num_objects,
            "total_area_px": round(self.total_area_px, 2),
            "mean_area_px": round(self.mean_area_px, 2),
            "area_std_px": round(self.area_std_px, 2),
            "size_distribution": self.size_distribution,
            "label_counts": self.label_counts,
            "objects": [obj.to_dict() for obj in self.objects],
        }


class FeatureExtractionEngine:
    """
    Computes morphological and intensity features for each segmented object.
    """

    def extract_all(
        self,
        image_rgb: np.ndarray,
        segments: list,
    ) -> FeatureExtractionResult:
        """
        Extract features for all segments.

        Args:
            image_rgb: Original RGB image
            segments: List of SegmentInfo from SegmentationEngine

        Returns:
            FeatureExtractionResult with per-object and aggregate features
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        obj_features: List[ObjectFeatures] = []
        areas: List[float] = []
        label_counts: Dict[str, int] = {}

        for seg in segments:
            features = self._extract_single(gray, seg)
            obj_features.append(features)
            areas.append(features.area_px)

            label_counts[seg.label] = label_counts.get(seg.label, 0) + 1

        # Aggregate statistics
        areas_arr = np.array(areas) if areas else np.array([0.0])
        total_area = float(areas_arr.sum())
        mean_area = float(areas_arr.mean()) if len(areas) > 0 else 0.0
        std_area = float(areas_arr.std()) if len(areas) > 0 else 0.0

        # Size distribution
        if len(areas) > 0:
            q33 = np.percentile(areas_arr, 33)
            q66 = np.percentile(areas_arr, 66)
            size_dist = {
                "small": int(np.sum(areas_arr <= q33)),
                "medium": int(np.sum((areas_arr > q33) & (areas_arr <= q66))),
                "large": int(np.sum(areas_arr > q66)),
            }
        else:
            size_dist = {"small": 0, "medium": 0, "large": 0}

        return FeatureExtractionResult(
            objects=obj_features,
            num_objects=len(obj_features),
            total_area_px=total_area,
            mean_area_px=mean_area,
            area_std_px=std_area,
            size_distribution=size_dist,
            label_counts=label_counts,
        )

    def _extract_single(
        self,
        gray: np.ndarray,
        segment,
    ) -> ObjectFeatures:
        """Extract all morphological features for a single segment."""

        mask = segment.mask.astype(np.uint8) * 255
        features = ObjectFeatures(
            segment_id=segment.segment_id,
            label=segment.label,
            confidence=segment.score,
            bbox=segment.box,
        )

        # ── Contour Analysis ──
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return features

        # Use the largest contour
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)

        features.area_px = area
        features.perimeter_px = perimeter

        # ── Bounding Rectangle ──
        x, y, bw, bh = cv2.boundingRect(contour)
        features.bbox_width = bw
        features.bbox_height = bh
        features.aspect_ratio = bw / max(bh, 1)
        features.extent = area / max(bw * bh, 1)

        # ── Centroid ──
        moments = cv2.moments(contour)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            features.centroid = (cx, cy)

        # ── Shape Descriptors ──
        if perimeter > 0:
            features.circularity = (4 * math.pi * area) / (perimeter ** 2)
            features.compactness = (perimeter ** 2) / max(area, 1)

        # Convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        hull_perimeter = cv2.arcLength(hull, closed=True)
        features.solidity = area / max(hull_area, 1)
        features.convexity = hull_perimeter / max(perimeter, 1)

        # Equivalent diameter
        features.equivalent_diameter = math.sqrt(4 * area / math.pi) if area > 0 else 0

        # ── Fitted Ellipse (requires >= 5 points) ──
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (_, (minor_axis, major_axis), angle) = ellipse
                features.orientation = angle
                features.eccentricity = minor_axis / max(major_axis, 1)
            except cv2.error:
                pass

        # ── Intensity Statistics ──
        mask_bool = segment.mask
        if mask_bool.any():
            roi_pixels = gray[mask_bool]
            features.mean_intensity = float(roi_pixels.mean())
            features.std_intensity = float(roi_pixels.std())
            features.min_intensity = float(roi_pixels.min())
            features.max_intensity = float(roi_pixels.max())

        features.shape = self._infer_shape(features)

        return features

    def _infer_shape(self, features: ObjectFeatures) -> str:
        """Translate raw morphology into a human-readable shape label."""
        if features.circularity >= 0.8 and 0.8 <= features.aspect_ratio <= 1.25:
            return "round"
        if features.aspect_ratio >= 1.8 or features.aspect_ratio <= 0.55:
            return "elongated"
        if features.solidity < 0.85 or features.circularity < 0.55:
            return "irregular"
        return "compact"
