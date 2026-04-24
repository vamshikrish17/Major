"""
VisionExtract 2.0 — Domain Analysis Engine
Three analysis modes: General, Biological, and Space.
Each mode extends the base detection+segmentation pipeline with domain-specific insights.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import pdist, squareform

from .feature_extraction import ObjectFeatures, FeatureExtractionResult

logger = logging.getLogger("visionextract.domain")


@dataclass
class DomainInsight:
    """Domain-specific analysis result."""
    mode: str                                       # "general", "bio", "space"
    summary: str = ""                               # Human-readable summary
    flags: List[str] = field(default_factory=list)  # Warning/info flags
    metrics: Dict = field(default_factory=dict)      # Domain-specific metrics
    heatmap: Optional[np.ndarray] = None            # Optional density/heat map
    annotations: List[Dict] = field(default_factory=list)  # Extra annotations

    def to_dict(self) -> Dict:
        return {
            "mode": self.mode,
            "summary": self.summary,
            "flags": self.flags,
            "metrics": self.metrics,
            "annotations": self.annotations,
        }


class DomainAnalysisEngine:
    """
    Multi-domain analysis engine providing specialized scientific insights
    based on the selected analysis mode.
    """

    def analyze(
        self,
        mode: str,
        image_rgb: np.ndarray,
        features: FeatureExtractionResult,
    ) -> DomainInsight:
        """
        Run domain-specific analysis.

        Args:
            mode: "general", "bio", or "space"
            image_rgb: Original RGB image
            features: FeatureExtractionResult from FeatureExtractionEngine

        Returns:
            DomainInsight with mode-specific metrics and flags
        """
        if mode == "bio":
            return self._analyze_bio(image_rgb, features)
        elif mode == "space":
            return self._analyze_space(image_rgb, features)
        else:
            return self._analyze_general(image_rgb, features)

    # ─────────────────────────────────────────────────────────
    # GENERAL MODE
    # ─────────────────────────────────────────────────────────
    def _analyze_general(
        self, image_rgb: np.ndarray, features: FeatureExtractionResult
    ) -> DomainInsight:
        """Standard object detection analysis with statistics."""
        h, w = image_rgb.shape[:2]
        total_image_area = h * w

        metrics = {
            "image_dimensions": {"width": w, "height": h},
            "total_objects": features.num_objects,
            "label_distribution": features.label_counts,
            "size_distribution": features.size_distribution,
            "coverage_percent": round(
                (features.total_area_px / total_image_area) * 100, 2
            ) if total_image_area > 0 else 0,
            "mean_confidence": round(
                float(np.mean([o.confidence for o in features.objects])), 4
            ) if features.objects else 0,
        }

        # Spatial distribution analysis
        if features.num_objects >= 2:
            centroids = np.array([o.centroid for o in features.objects])
            metrics["spatial_spread"] = round(float(centroids.std(axis=0).mean()), 2)

            # Quadrant distribution
            quadrants = {"TL": 0, "TR": 0, "BL": 0, "BR": 0}
            for cx, cy in centroids:
                q_key = ("T" if cy < h / 2 else "B") + ("L" if cx < w / 2 else "R")
                quadrants[q_key] += 1
            metrics["quadrant_distribution"] = quadrants

        flags = []
        if features.num_objects == 0:
            flags.append("No objects detected — try lowering the confidence threshold")
        elif features.num_objects > 50:
            flags.append(f"High object density: {features.num_objects} objects detected")

        summary = (
            f"Detected {features.num_objects} object(s) across "
            f"{len(features.label_counts)} class(es). "
            f"Total area coverage: {metrics['coverage_percent']}%."
        )

        return DomainInsight(
            mode="general",
            summary=summary,
            flags=flags,
            metrics=metrics,
        )

    # ─────────────────────────────────────────────────────────
    # BIOLOGICAL MODE
    # ─────────────────────────────────────────────────────────
    def _analyze_bio(
        self, image_rgb: np.ndarray, features: FeatureExtractionResult
    ) -> DomainInsight:
        """
        Biological/biomedical analysis:
        - Cell counting and size analysis
        - Abnormality detection via statistical outliers
        - Density mapping
        - Morphological classification
        """
        h, w = image_rgb.shape[:2]
        objects = features.objects

        metrics: Dict = {
            "cell_count": features.num_objects,
            "image_dimensions": {"width": w, "height": h},
        }
        flags: List[str] = []
        annotations: List[Dict] = []

        if not objects:
            return DomainInsight(
                mode="bio",
                summary="No biological structures detected.",
                flags=["No cells or structures found — verify image quality and mode selection"],
                metrics=metrics,
            )

        # ── Cell Size Analysis ──
        areas = np.array([o.area_px for o in objects])
        circularities = np.array([o.circularity for o in objects])
        intensities = np.array([o.mean_intensity for o in objects])

        metrics["area_stats"] = {
            "mean": round(float(areas.mean()), 2),
            "std": round(float(areas.std()), 2),
            "min": round(float(areas.min()), 2),
            "max": round(float(areas.max()), 2),
            "median": round(float(np.median(areas)), 2),
        }

        metrics["circularity_stats"] = {
            "mean": round(float(circularities.mean()), 4),
            "std": round(float(circularities.std()), 4),
        }

        # ── Abnormality Detection (Z-score based) ──
        abnormal_objects: List[Dict] = []
        if len(objects) >= 3:
            area_z = np.abs((areas - areas.mean()) / max(areas.std(), 1e-6))
            circ_z = np.abs((circularities - circularities.mean()) / max(circularities.std(), 1e-6))
            intensity_z = np.abs((intensities - intensities.mean()) / max(intensities.std(), 1e-6))

            z_threshold = 2.0
            for i, obj in enumerate(objects):
                anomaly_score = 0
                reasons = []

                if area_z[i] > z_threshold:
                    anomaly_score += area_z[i]
                    reasons.append(f"unusual size (z={area_z[i]:.2f})")
                if circ_z[i] > z_threshold:
                    anomaly_score += circ_z[i]
                    reasons.append(f"irregular shape (z={circ_z[i]:.2f})")
                if intensity_z[i] > z_threshold:
                    anomaly_score += intensity_z[i]
                    reasons.append(f"abnormal intensity (z={intensity_z[i]:.2f})")

                if anomaly_score > 0:
                    abnormal_objects.append({
                        "segment_id": obj.segment_id,
                        "label": obj.label,
                        "anomaly_score": round(anomaly_score, 3),
                        "reasons": reasons,
                        "centroid": list(obj.centroid),
                    })

        metrics["abnormalities"] = {
            "count": len(abnormal_objects),
            "objects": abnormal_objects,
        }

        if abnormal_objects:
            flags.append(
                f"⚠ {len(abnormal_objects)} potentially abnormal cell(s) detected — "
                "review flagged objects for irregular morphology"
            )

        # ── Density Analysis ──
        if features.num_objects >= 2:
            centroids = np.array([o.centroid for o in objects])
            density = self._compute_density(centroids, w, h)
            metrics["density_analysis"] = density

        # ── Morphological Classification ──
        round_cells = sum(1 for c in circularities if c > 0.8)
        elongated_cells = sum(1 for o in objects if o.aspect_ratio > 2.0 or o.aspect_ratio < 0.5)
        irregular_cells = sum(1 for c in circularities if c < 0.5)

        metrics["morphology_distribution"] = {
            "round": round_cells,
            "elongated": elongated_cells,
            "irregular": irregular_cells,
        }

        # ── Cluster Analysis ──
        if features.num_objects >= 3:
            clusters = self._simple_clustering(
                np.array([o.centroid for o in objects]),
                threshold=min(w, h) * 0.15,
            )
            metrics["cluster_count"] = clusters["num_clusters"]
            metrics["cluster_sizes"] = clusters["sizes"]

        summary = (
            f"Biological analysis: {features.num_objects} cell(s)/structure(s) detected. "
            f"Mean area: {metrics['area_stats']['mean']:.0f}px². "
            f"Morphology: {round_cells} round, {elongated_cells} elongated, {irregular_cells} irregular. "
            f"{len(abnormal_objects)} flagged as potentially abnormal."
        )

        return DomainInsight(
            mode="bio",
            summary=summary,
            flags=flags,
            metrics=metrics,
            annotations=annotations,
        )

    # ─────────────────────────────────────────────────────────
    # SPACE MODE
    # ─────────────────────────────────────────────────────────
    def _analyze_space(
        self, image_rgb: np.ndarray, features: FeatureExtractionResult
    ) -> DomainInsight:
        """
        Space/satellite image analysis:
        - Terrain segmentation analysis
        - Crater-like feature detection
        - Region classification by texture
        - Spectral analysis
        """
        h, w = image_rgb.shape[:2]
        objects = features.objects
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        metrics: Dict = {
            "detected_features": features.num_objects,
            "image_dimensions": {"width": w, "height": h},
        }
        flags: List[str] = []

        # ── Terrain Texture Analysis ──
        terrain = self._analyze_terrain(gray)
        metrics["terrain_analysis"] = terrain

        # ── Crater-like Feature Detection ──
        crater_candidates = []
        for obj in objects:
            if obj.circularity > 0.6 and obj.solidity > 0.7:
                crater_candidates.append({
                    "segment_id": obj.segment_id,
                    "circularity": round(obj.circularity, 4),
                    "equivalent_diameter": round(obj.equivalent_diameter, 2),
                    "centroid": list(obj.centroid),
                    "confidence": round(obj.confidence, 4),
                })

        metrics["crater_candidates"] = {
            "count": len(crater_candidates),
            "features": crater_candidates,
        }

        # ── Region Classification ──
        regions = self._classify_regions(image_rgb)
        metrics["region_classification"] = regions

        # ── Spectral Analysis ──
        spectral = self._spectral_analysis(image_rgb)
        metrics["spectral_analysis"] = spectral

        # ── Object Size in Relative Scale ──
        if objects:
            sizes = [o.equivalent_diameter for o in objects]
            metrics["feature_sizes"] = {
                "min_diameter_px": round(min(sizes), 2),
                "max_diameter_px": round(max(sizes), 2),
                "mean_diameter_px": round(float(np.mean(sizes)), 2),
            }

        if crater_candidates:
            flags.append(f"🌑 {len(crater_candidates)} circular crater-like feature(s) detected")

        summary = (
            f"Space analysis: {features.num_objects} surface feature(s) detected. "
            f"{len(crater_candidates)} crater candidate(s). "
            f"Terrain roughness: {terrain.get('roughness', 'N/A')}. "
            f"Dominant region type: {regions.get('dominant_type', 'mixed')}."
        )

        return DomainInsight(
            mode="space",
            summary=summary,
            flags=flags,
            metrics=metrics,
        )

    # ─────────────────────────────────────────────────────────
    # HELPER METHODS
    # ─────────────────────────────────────────────────────────
    def _compute_density(
        self, centroids: np.ndarray, width: int, height: int
    ) -> Dict:
        """Compute spatial density metrics from centroids."""
        if len(centroids) < 2:
            return {"density_per_1000px2": 0, "nearest_neighbor_mean": 0}

        # Pairwise distances
        dists = squareform(pdist(centroids))
        np.fill_diagonal(dists, np.inf)
        nn_dists = dists.min(axis=1)

        return {
            "density_per_1000px2": round(
                len(centroids) / (width * height) * 1000000, 4
            ),
            "nearest_neighbor_mean": round(float(nn_dists.mean()), 2),
            "nearest_neighbor_std": round(float(nn_dists.std()), 2),
            "dispersion_index": round(
                float(nn_dists.var() / max(nn_dists.mean(), 1e-6)), 4
            ),
        }

    def _simple_clustering(
        self, centroids: np.ndarray, threshold: float
    ) -> Dict:
        """Simple distance-based clustering (no sklearn dependency)."""
        n = len(centroids)
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue
            cluster = [i]
            visited[i] = True
            queue = [i]

            while queue:
                current = queue.pop(0)
                for j in range(n):
                    if not visited[j]:
                        dist = np.linalg.norm(centroids[current] - centroids[j])
                        if dist < threshold:
                            visited[j] = True
                            cluster.append(j)
                            queue.append(j)

            clusters.append(cluster)

        return {
            "num_clusters": len(clusters),
            "sizes": [len(c) for c in clusters],
        }

    def _analyze_terrain(self, gray: np.ndarray) -> Dict:
        """Analyze terrain texture from grayscale image."""
        # Laplacian variance as roughness measure
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        roughness = float(laplacian.var())

        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(edges.sum()) / max(edges.size, 1) * 100

        # Texture uniformity (entropy approximation)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / max(hist.sum(), 1)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))

        roughness_label = "smooth"
        if roughness > 1000:
            roughness_label = "very rough"
        elif roughness > 500:
            roughness_label = "rough"
        elif roughness > 100:
            roughness_label = "moderate"

        return {
            "roughness": roughness_label,
            "roughness_value": round(roughness, 2),
            "edge_density_pct": round(edge_density, 3),
            "texture_entropy": round(float(entropy), 4),
        }

    def _classify_regions(self, image_rgb: np.ndarray) -> Dict:
        """Classify image regions by dominant color/texture."""
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        # Simple color-based region classification
        total_px = h_channel.size
        dark_regions = float(np.sum(v_channel < 50)) / total_px * 100
        bright_regions = float(np.sum(v_channel > 200)) / total_px * 100
        saturated = float(np.sum(s_channel > 100)) / total_px * 100

        # Determine dominant type
        if dark_regions > 60:
            dominant = "shadow/void"
        elif bright_regions > 60:
            dominant = "bright surface"
        elif saturated > 40:
            dominant = "colored terrain"
        else:
            dominant = "mixed terrain"

        return {
            "dominant_type": dominant,
            "dark_region_pct": round(dark_regions, 2),
            "bright_region_pct": round(bright_regions, 2),
            "saturated_region_pct": round(saturated, 2),
        }

    def _spectral_analysis(self, image_rgb: np.ndarray) -> Dict:
        """Basic spectral band analysis of RGB channels."""
        r, g, b = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]

        return {
            "red_band": {
                "mean": round(float(r.mean()), 2),
                "std": round(float(r.std()), 2),
            },
            "green_band": {
                "mean": round(float(g.mean()), 2),
                "std": round(float(g.std()), 2),
            },
            "blue_band": {
                "mean": round(float(b.mean()), 2),
                "std": round(float(b.std()), 2),
            },
            "ndvi_proxy": round(
                float((r.astype(float).mean() - g.astype(float).mean()) /
                      max(r.astype(float).mean() + g.astype(float).mean(), 1)),
                4,
            ),
        }
