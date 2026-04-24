"""
VisionExtract 2.0 — Image Quality Analyzer
Detects common image quality issues before running the main pipeline.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import cv2
import numpy as np

logger = logging.getLogger("visionextract.quality")


@dataclass
class ImageQualityReport:
    """Structured output for image quality analysis."""
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "issues": self.issues,
            "metrics": self.metrics,
            "recommendations": self.recommendations,
        }


class ImageQualityAnalyzer:
    """
    Lightweight image quality analyzer for pre-inference diagnostics.
    Detects blur, low light, noise, and object overlap/occlusion risk.
    """

    BLUR_THRESHOLD = 120.0
    LOW_LIGHT_THRESHOLD = 65.0
    NOISE_THRESHOLD = 18.0
    OCCLUSION_IOU_THRESHOLD = 0.2

    def analyze(self, image_rgb: np.ndarray) -> ImageQualityReport:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        noise_score = self._estimate_noise(gray)
        occlusion_score = self._estimate_occlusion(gray)

        issues: List[str] = []
        recommendations: List[str] = []

        if blur_score < self.BLUR_THRESHOLD:
            issues.append("Blur Detected")
            recommendations.append("Use a sharper image or reduce motion during capture.")

        if brightness < self.LOW_LIGHT_THRESHOLD:
            issues.append("Low Light")
            recommendations.append("Increase exposure or use a brighter source image.")

        if noise_score > self.NOISE_THRESHOLD:
            issues.append("High Noise")
            recommendations.append("Apply denoising or use a cleaner source image.")

        if occlusion_score > self.OCCLUSION_IOU_THRESHOLD:
            issues.append("Occlusion / Overlap")
            recommendations.append("Capture a view with less overlap between foreground objects.")

        metrics = {
            "blur_variance": round(blur_score, 3),
            "brightness_mean": round(brightness, 3),
            "noise_score": round(noise_score, 3),
            "occlusion_score": round(occlusion_score, 3),
        }

        return ImageQualityReport(
            issues=issues,
            metrics=metrics,
            recommendations=recommendations,
        )

    def _estimate_noise(self, gray: np.ndarray) -> float:
        """Estimate image noise using residual energy after median filtering."""
        denoised = cv2.medianBlur(gray, 3)
        residual = gray.astype(np.float32) - denoised.astype(np.float32)
        return float(np.std(residual))

    def _estimate_occlusion(self, gray: np.ndarray) -> float:
        """
        Approximate overlap/occlusion by comparing merged foreground area against
        the summed area of connected components after adaptive thresholding.
        """
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            4,
        )

        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) < 2:
            return 0.0

        component_areas = [cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 25]
        if len(component_areas) < 2:
            return 0.0

        summed_area = float(sum(component_areas))
        merged_area = float(np.count_nonzero(opened))
        if summed_area <= 0:
            return 0.0

        overlap_ratio = max(0.0, min(1.0, 1.0 - (merged_area / summed_area)))
        return overlap_ratio
