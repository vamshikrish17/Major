"""
VisionExtract 2.0 — Centralized Configuration
Pydantic-based settings with environment variable support and auto device detection.
"""

import os
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


# ─── Path Resolution ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT  # model weights live at project root
STATIC_DIR = PROJECT_ROOT / "frontend"
UPLOAD_DIR = PROJECT_ROOT / "static" / "uploads"
RESULT_DIR = PROJECT_ROOT / "static" / "results"
DB_DIR = PROJECT_ROOT / "data"
TRAINING_DIR = PROJECT_ROOT / "training"
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODEL_REGISTRY_DIR = PROJECT_ROOT / "models"


def _detect_device() -> str:
    """Auto-detect best available compute device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Settings:
    """Application-wide configuration."""

    # ── Compute ──
    DEVICE: str = field(default_factory=_detect_device)

    # ── Model Paths ──
    YOLO_MODEL_PATH: str = str(MODELS_DIR / "yolov8n.pt")
    SAM_CHECKPOINT: str = str(MODELS_DIR / "sam_vit_b_01ec64.pth")
    SAM_MODEL_TYPE: str = "vit_b"
    DOMAIN_YOLO_MODELS: dict = field(default_factory=lambda: {
        "general": str(MODEL_REGISTRY_DIR / "general" / "yolo" / "best.pt"),
        "bio": str(MODEL_REGISTRY_DIR / "bio" / "yolo" / "best.pt"),
        "space": str(MODEL_REGISTRY_DIR / "space" / "yolo" / "best.pt"),
    })
    BIO_UNET_CHECKPOINT: str = str(MODEL_REGISTRY_DIR / "bio" / "unet" / "best.pt")

    # ── Storage ──
    UPLOAD_DIR: str = str(UPLOAD_DIR)
    RESULT_DIR: str = str(RESULT_DIR)
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100 MB
    ALLOWED_EXTENSIONS: tuple = (
        ".png", ".jpg", ".jpeg", ".webp", ".avif",
        ".heic", ".heif", ".bmp", ".tiff", ".tif",
    )

    # ── Database ──
    DB_URL: str = f"sqlite+aiosqlite:///{DB_DIR / 'visionextract.db'}"

    # ── API ──
    API_PREFIX: str = "/api"
    CORS_ORIGINS: List[str] = field(default_factory=lambda: ["*"])
    HOST: str = "0.0.0.0"
    PORT: int = 1726

    # ── Detection Defaults ──
    DEFAULT_CONFIDENCE: float = 0.35
    DEFAULT_NMS_IOU: float = 0.45

    # ── Security ──
    RATE_LIMIT: int = 30  # requests per minute
    SECRET_KEY: str = "visionextract-2.0-secret-change-in-production"

    def __post_init__(self):
        """Ensure directories exist."""
        for d in (
            self.UPLOAD_DIR,
            self.RESULT_DIR,
            str(DB_DIR),
            str(DATASETS_DIR),
            str(MODEL_REGISTRY_DIR),
        ):
            os.makedirs(d, exist_ok=True)

        for path in self.DOMAIN_YOLO_MODELS.values():
            os.makedirs(os.path.dirname(path), exist_ok=True)
        os.makedirs(os.path.dirname(self.BIO_UNET_CHECKPOINT), exist_ok=True)

    def get_domain_yolo_path(self, mode: str) -> str:
        """Return the preferred YOLO checkpoint path for a domain."""
        return self.DOMAIN_YOLO_MODELS.get(mode, self.YOLO_MODEL_PATH)


# Singleton settings instance
settings = Settings()
