"""
VisionExtract 2.0 — Multi-Format Input Pipeline
Robust image ingestion with Pillow → OpenCV → plugin fallback chain.
Handles PNG, JPG, WEBP, AVIF, HEIC/HEIF, BMP, TIFF + base64 webcam + video frames.
"""

import io
import base64
import logging
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image, ExifTags

logger = logging.getLogger("visionextract.input")

# Try to register HEIF/HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_SUPPORTED = True
    logger.info("HEIF/HEIC support enabled via pillow-heif")
except ImportError:
    HEIF_SUPPORTED = False
    logger.warning("pillow-heif not installed — HEIC/HEIF files will use OpenCV fallback")


# Maximum dimension before auto-downscale (prevents OOM on huge satellite images)
MAX_DIMENSION = 4096


def _fix_orientation(img: Image.Image) -> Image.Image:
    """Auto-rotate image based on EXIF orientation tag."""
    try:
        exif = img.getexif()
        orientation_key = None
        for k, v in ExifTags.TAGS.items():
            if v == "Orientation":
                orientation_key = k
                break
        if orientation_key and orientation_key in exif:
            orient = exif[orientation_key]
            rotations = {3: 180, 6: 270, 8: 90}
            if orient in rotations:
                img = img.rotate(rotations[orient], expand=True)
    except Exception:
        pass  # No EXIF or unreadable — skip silently
    return img


def _resize_if_needed(img_np: np.ndarray, max_dim: int = MAX_DIMENSION) -> np.ndarray:
    """Downscale image if either dimension exceeds max_dim, preserving aspect ratio."""
    h, w = img_np.shape[:2]
    if max(h, w) <= max_dim:
        return img_np
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    logger.info(f"Resizing image from {w}x{h} to {new_w}x{new_h}")
    return cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)


def decode_image_pillow(data: bytes) -> Optional[np.ndarray]:
    """Primary decoder: Pillow (handles most formats natively)."""
    try:
        img = Image.open(io.BytesIO(data))
        img = _fix_orientation(img)
        img = img.convert("RGB")
        return np.array(img)
    except Exception as e:
        logger.debug(f"Pillow decode failed: {e}")
        return None


def decode_image_opencv(data: bytes) -> Optional[np.ndarray]:
    """Fallback decoder: OpenCV imdecode."""
    try:
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return None
    except Exception as e:
        logger.debug(f"OpenCV decode failed: {e}")
        return None


def decode_image(data: bytes) -> np.ndarray:
    """
    Multi-decoder fallback pipeline: Pillow → OpenCV.
    Returns RGB numpy array. Raises ValueError if all decoders fail.
    """
    # Try Pillow first (broadest format support)
    result = decode_image_pillow(data)
    if result is not None:
        return _resize_if_needed(result)

    # Fallback to OpenCV
    result = decode_image_opencv(data)
    if result is not None:
        return _resize_if_needed(result)

    raise ValueError("Failed to decode image with all available decoders (Pillow + OpenCV)")


def decode_base64_image(base64_str: str) -> np.ndarray:
    """
    Decode a base64-encoded image (from webcam capture).
    Accepts raw base64 or data URI format (data:image/png;base64,...).
    """
    if "," in base64_str:
        # Strip data URI header
        base64_str = base64_str.split(",", 1)[1]
    data = base64.b64decode(base64_str)
    return decode_image(data)


def load_image_file(file_path: str) -> np.ndarray:
    """Load an image file from disk using the multi-decoder pipeline."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")
    data = path.read_bytes()
    return decode_image(data)


def extract_video_frames(
    video_path: str,
    max_frames: int = 30,
    interval: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Extract frames from a video file.

    Args:
        video_path: Path to video file (MP4, AVI, MOV, etc.)
        max_frames: Maximum number of frames to extract
        interval: Frame interval (None = evenly spaced across video)

    Returns:
        List of RGB numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if interval is None:
        # Evenly space frames across video
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
    else:
        frame_indices = list(range(0, total_frames, interval))[:max_frames]

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(_resize_if_needed(rgb_frame))

    cap.release()
    logger.info(f"Extracted {len(frames)}/{total_frames} frames from video")
    return frames


def image_to_bgr(rgb_array: np.ndarray) -> np.ndarray:
    """Convert RGB numpy array to BGR for OpenCV operations."""
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def image_to_bytes(rgb_array: np.ndarray, fmt: str = "png") -> bytes:
    """Encode RGB numpy array to image bytes."""
    bgr = image_to_bgr(rgb_array)
    ext = f".{fmt.lower()}"
    success, buffer = cv2.imencode(ext, bgr)
    if not success:
        raise RuntimeError(f"Failed to encode image to {fmt}")
    return buffer.tobytes()
