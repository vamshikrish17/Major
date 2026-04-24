"""
VisionExtract 2.0 — Storage Service
Abstract file storage with local filesystem implementation.
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import settings


class StorageBackend:
    """Abstract storage interface (local, S3, GCS compatible)."""

    def __init__(self):
        self.upload_dir = settings.UPLOAD_DIR
        self.result_dir = settings.RESULT_DIR
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

    def save_upload(self, filename: str, data: bytes) -> str:
        """Save uploaded file. Returns the saved filepath."""
        safe_name = f"{uuid.uuid4().hex}_{self._sanitize(filename)}"
        filepath = os.path.join(self.upload_dir, safe_name)
        with open(filepath, "wb") as f:
            f.write(data)
        return filepath

    def save_result_image(self, image_bgr: np.ndarray, prefix: str = "result") -> str:
        """Save a result image (BGR numpy array). Returns filepath."""
        filename = f"{prefix}_{uuid.uuid4().hex}.png"
        filepath = os.path.join(self.result_dir, filename)
        cv2.imwrite(filepath, image_bgr)
        return filepath

    def save_segment(self, image_bgra: np.ndarray, segment_id: str, label: str) -> str:
        """Save a transparent segment PNG. Returns filepath."""
        filename = f"{segment_id}_{label}.png"
        filepath = os.path.join(self.result_dir, filename)
        cv2.imwrite(filepath, image_bgra)
        return filepath

    def get_url(self, filepath: str) -> str:
        """Convert filepath to URL path for serving."""
        # Convert absolute path to a path relative to the static directory.
        static_root = os.path.dirname(self.upload_dir)
        rel = os.path.relpath(filepath, static_root)
        return f"/static/{rel.replace(os.sep, '/')}"

    def delete_file(self, filepath: str) -> bool:
        """Delete a file from storage."""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
        except Exception:
            pass
        return False

    def _sanitize(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Keep only alphanumeric, dots, dashes, underscores
        name = "".join(c for c in filename if c.isalnum() or c in ".-_")
        return name or "upload"


# Singleton storage instance
storage = StorageBackend()
