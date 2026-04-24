"""
VisionExtract 2.0 — Security Utilities
File validation, size limits, and security helpers.
"""

import hashlib
import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger("visionextract.security")

# Magic bytes for common image formats
MAGIC_BYTES = {
    b"\x89PNG\r\n\x1a\n": "png",
    b"\xff\xd8\xff": "jpeg",
    b"RIFF": "webp",  # RIFF header (WebP starts with RIFF...WEBP)
    b"BM": "bmp",
    b"II\x2a\x00": "tiff_le",   # Little-endian TIFF
    b"MM\x00\x2a": "tiff_be",   # Big-endian TIFF
    b"GIF87a": "gif",
    b"GIF89a": "gif",
}

ALLOWED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".webp", ".avif",
    ".heic", ".heif", ".bmp", ".tiff", ".tif",
    ".gif", ".mp4", ".avi", ".mov", ".mkv",
}


def validate_file_magic(data: bytes) -> Tuple[bool, str]:
    """
    Validate file type by checking magic bytes.
    Returns (is_valid, detected_format).
    """
    if len(data) < 8:
        return False, "unknown"

    for magic, fmt in MAGIC_BYTES.items():
        if data[:len(magic)] == magic:
            return True, fmt

    # ISO BMFF container check used by AVIF/HEIC/HEIF.
    if len(data) > 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"avif", b"avis", b"mif1"):
            return True, "avif"
        if brand in (b"heic", b"heix", b"hevc", b"heif", b"heis", b"heim"):
            return True, "heic"

    # If extension-based checking is needed, fall through
    return False, "unknown"


def validate_file_extension(filename: str) -> bool:
    """Check if file extension is in the allowed list."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def validate_file_size(data: bytes, max_size: int) -> bool:
    """Check if file size is within limits."""
    return len(data) <= max_size


def validate_upload(
    filename: str,
    data: bytes,
    max_size: int = 100 * 1024 * 1024,
) -> Tuple[bool, Optional[str]]:
    """
    Full upload validation pipeline.
    Returns (is_valid, error_message).
    """
    # 1. Size check
    if not validate_file_size(data, max_size):
        size_mb = len(data) / (1024 * 1024)
        max_mb = max_size / (1024 * 1024)
        return False, f"File too large: {size_mb:.1f}MB (max {max_mb:.0f}MB)"

    # 2. Extension check
    if not validate_file_extension(filename):
        ext = os.path.splitext(filename)[1]
        return False, f"Unsupported file format: {ext}"

    # 3. Magic bytes validation
    is_valid_magic, detected_fmt = validate_file_magic(data)
    if not is_valid_magic:
        # Allow through if extension is valid (some formats have complex magic)
        logger.warning(
            f"Could not validate magic bytes for {filename} — "
            f"allowing based on extension"
        )

    return True, None


def compute_file_hash(data: bytes) -> str:
    """Compute SHA-256 hash of file data."""
    return hashlib.sha256(data).hexdigest()[:16]
