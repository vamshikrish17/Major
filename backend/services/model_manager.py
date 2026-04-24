"""
VisionExtract 2.0 — Model Manager
Singleton AI model loader with device fallback and warm-up.
"""

import logging
import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from ..config import settings
from ..core.detection import DetectionEngine
from ..core.segmentation import SegmentationEngine
from ..core.feature_extraction import FeatureExtractionEngine
from ..core.domain_analysis import DomainAnalysisEngine
from .image_quality_analyzer import ImageQualityAnalyzer

logger = logging.getLogger("visionextract.model_manager")


class ModelManager:
    """
    Singleton manager for all AI models.
    Loads models once on startup and provides access to pipeline engines.
    """

    _instance: Optional["ModelManager"] = None

    def __init__(self):
        self.device = settings.DEVICE
        self.detection_engines: Dict[str, DetectionEngine] = {}
        self.segmentation_engine: Optional[SegmentationEngine] = None
        self.feature_engine: FeatureExtractionEngine = FeatureExtractionEngine()
        self.domain_engine: DomainAnalysisEngine = DomainAnalysisEngine()
        self.quality_analyzer: ImageQualityAnalyzer = ImageQualityAnalyzer()
        self._loaded = False
        self._startup_time = time.time()

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_models(self):
        """Load all AI models (call once at startup)."""
        if self._loaded:
            logger.info("Models already loaded — skipping")
            return

        logger.info(f"═══ VisionExtract 2.0 Model Loading ═══")
        logger.info(f"Device: {self.device}")

        start = time.time()

        # Load domain-specific YOLOv8 models with fallback to the base checkpoint.
        for mode in ("general", "bio", "space"):
            preferred_path = settings.get_domain_yolo_path(mode)
            model_path = preferred_path if os.path.exists(preferred_path) else settings.YOLO_MODEL_PATH
            if model_path != preferred_path:
                logger.info(
                    "Domain model for %s not found at %s — falling back to %s",
                    mode,
                    preferred_path,
                    settings.YOLO_MODEL_PATH,
                )
            self.detection_engines[mode] = DetectionEngine(
                model_path=model_path,
                device=self.device,
            )

        # Load SAM
        self.segmentation_engine = SegmentationEngine(
            sam_checkpoint=settings.SAM_CHECKPOINT,
            sam_model_type=settings.SAM_MODEL_TYPE,
            device=self.device,
            unet_checkpoint=settings.BIO_UNET_CHECKPOINT,
        )

        elapsed = time.time() - start
        logger.info(f"═══ All models loaded in {elapsed:.2f}s ═══")
        self._loaded = True

        # Warm-up with dummy inference
        self._warmup()

    def _warmup(self):
        """Run a dummy inference to prime the models."""
        try:
            logger.info("Running model warm-up...")
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            det = self.get_detection_engine("general").detect(dummy_img, conf_threshold=0.9)
            logger.info("Warm-up complete")
        except Exception as e:
            logger.warning(f"Warm-up failed (non-critical): {e}")

    def get_detection_engine(self, mode: str) -> DetectionEngine:
        """Return the detector assigned to the selected domain."""
        return self.detection_engines.get(mode, self.detection_engines["general"])

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def uptime(self) -> float:
        return time.time() - self._startup_time

    @property
    def models_status(self) -> dict:
        return {
            "yolov8_general": "general" in self.detection_engines,
            "yolov8_bio": "bio" in self.detection_engines,
            "yolov8_space": "space" in self.detection_engines,
            "sam": self.segmentation_engine is not None,
            "unet": (
                self.segmentation_engine is not None
                and self.segmentation_engine.unet is not None
            ),
            "image_quality": True,
            "feature_engine": True,
            "domain_engine": True,
        }
