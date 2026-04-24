"""
VisionExtract 2.0 — Analysis Router
Core analysis endpoints: upload, analyze, batch process.
"""

import logging
import os
import uuid
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..config import settings
from ..core.input_pipeline import decode_base64_image, decode_image, image_to_bgr
from ..core.segmentation import save_segment_png
from ..models.database import AnalysisRecord, get_new_session
from ..models.schemas import (
    AnalysisOptions,
    AnalysisResponse,
    DetectionItem,
    ErrorResponse,
    TimingInfo,
)
from ..services.model_manager import ModelManager
from ..services.storage import storage
from ..utils.metrics import PerformanceTimer, compute_detection_metrics
from ..utils.security import validate_upload

logger = logging.getLogger("visionextract.api.analysis")
router = APIRouter(prefix="/api", tags=["analysis"])


def get_mode_display_name(mode: str) -> str:
    return {
        "general": "General Vision",
        "bio": "BioVision",
        "space": "Geo-Spatial Analysis",
    }.get(mode, "Analysis")


def get_domain_label(mode: str, raw_label: str) -> str:
    """
    Keep labels natural in General Vision, but avoid unrelated COCO-style names
    in domain-specific modes until dedicated trained checkpoints are available.
    """
    if mode == "bio":
        if raw_label == "auto-segment":
            return "cell region"
        return "cell structure"
    if mode == "space":
        if raw_label == "auto-segment":
            return "terrain region"
        return "geospatial feature"
    if raw_label == "auto-segment":
        return "visual segment"
    return raw_label


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: Optional[UploadFile] = File(None),
    webcam_data: Optional[str] = Form(None),
    mode: str = Form("general"),
    confidence: float = Form(0.35),
    use_hybrid_prompts: bool = Form(True),
):
    """
    Upload and analyze an image through the full VisionExtract pipeline.

    Accepts file upload or base64 webcam capture.
    Runs: Input → Detection → Segmentation → Feature Extraction → Domain Analysis.
    """
    manager = ModelManager.get_instance()
    if not manager.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    timer = PerformanceTimer()
    analysis_id = uuid.uuid4().hex[:12]

    # ── 1. Input Decoding ──
    timer.start("input")
    image_rgb = None
    original_filename = "webcam_capture.png"

    if webcam_data and webcam_data.strip():
        try:
            image_rgb = decode_base64_image(webcam_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid webcam data: {e}")
    elif file:
        original_filename = file.filename or "upload.png"
        data = await file.read()

        # Validate upload
        is_valid, error_msg = validate_upload(original_filename, data, settings.MAX_UPLOAD_SIZE)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)

        try:
            image_rgb = decode_image(data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="No image provided (file or webcam_data)")

    timer.stop()

    # Save original
    image_bgr = image_to_bgr(image_rgb)
    original_path = storage.save_result_image(image_bgr, prefix="original")
    original_url = storage.get_url(original_path)

    # ── 2. Image Quality Analysis ──
    timer.start("quality_analysis")
    quality_report = manager.quality_analyzer.analyze(image_rgb)
    timer.stop()

    # ── 3. Detection ──
    timer.start("detection")
    detection_engine = manager.get_detection_engine(mode)
    detection = detection_engine.detect(
        image_rgb,
        conf_threshold=confidence,
    )
    timer.stop()

    # ── 4. Segmentation ──
    timer.start("segmentation")
    segmentation = manager.segmentation_engine.segment(
        image_rgb,
        detection,
        use_hybrid_prompts=use_hybrid_prompts,
        mode=mode,
    )
    timer.stop()

    # Save overlay
    overlay_path = storage.save_result_image(segmentation.overlay_image, prefix="overlay")
    overlay_url = storage.get_url(overlay_path)

    # Save individual segments as transparent PNGs
    segment_urls = []
    for seg in segmentation.segments:
        filename = save_segment_png(image_bgr, seg, settings.RESULT_DIR)
        seg_url = f"/static/results/{filename}"
        display_label = get_domain_label(mode, seg.label)
        segment_urls.append({
            "url": seg_url,
            "label": display_label,
            "score": round(seg.score, 4),
            "segment_id": seg.segment_id,
        })

    # ── 5. Feature Extraction ──
    timer.start("feature_extraction")
    features = manager.feature_engine.extract_all(image_rgb, segmentation.segments)
    timer.stop()

    # ── 6. Domain Analysis ──
    timer.start("domain_analysis")
    domain_insight = manager.domain_engine.analyze(mode, image_rgb, features)
    timer.stop()

    if quality_report.issues:
        domain_insight.flags = quality_report.issues + domain_insight.flags

    # ── 7. Compute Metrics ──
    iou_preds = [seg.iou_prediction for seg in segmentation.segments]
    confs = [seg.score for seg in segmentation.segments]
    det_metrics = compute_detection_metrics(iou_preds, confs)

    # ── 8. Build Detection Items ──
    detection_items = []
    for i, seg in enumerate(segmentation.segments):
        obj_features = features.objects[i].to_dict() if i < len(features.objects) else {}
        display_label = get_domain_label(mode, seg.label)
        detection_items.append(DetectionItem(
            segment_id=seg.segment_id,
            label=display_label,
            confidence=round(seg.score, 4),
            bbox=seg.box,
            mask_url=segment_urls[i]["url"] if i < len(segment_urls) else None,
            features=obj_features,
        ))

    # ── 9. Persist to Database ──
    try:
        session = get_new_session()
        features_payload = features.to_dict()
        features_payload["quality_report"] = quality_report.to_dict()
        record = AnalysisRecord(
            id=analysis_id,
            mode=mode,
            input_filename=original_filename,
            original_path=original_path,
            overlay_path=overlay_path,
            object_count=segmentation.num_segments,
            processing_time_ms=timer.total_ms,
            device=settings.DEVICE,
        )
        record.set_features(features_payload)
        record.set_domain(domain_insight.to_dict())
        record.set_segments([s for s in segment_urls])
        session.add(record)
        session.commit()
        session.close()
    except Exception as e:
        logger.warning(f"Database save failed (non-critical): {e}")

    # ── 10. Build Response ──
    timing = TimingInfo(
        total_ms=timer.total_ms,
        quality_analysis_ms=timer.stages.get("quality_analysis", 0),
        detection_ms=timer.stages.get("detection", 0),
        segmentation_ms=timer.stages.get("segmentation", 0),
        feature_extraction_ms=timer.stages.get("feature_extraction", 0),
        domain_analysis_ms=timer.stages.get("domain_analysis", 0),
    )

    features_summary = features.to_dict()
    features_summary["detection_metrics"] = det_metrics.to_dict()
    features_summary["quality_report"] = quality_report.to_dict()

    return AnalysisResponse(
        analysis_id=analysis_id,
        timestamp=datetime.utcnow().isoformat(),
        mode=mode,
        original_url=original_url,
        overlay_url=overlay_url,
        detections=detection_items,
        num_objects=segmentation.num_segments,
        features_summary=features_summary,
        quality_report=quality_report.to_dict(),
        domain_insight=domain_insight.to_dict(),
        segment_urls=segment_urls,
        timing=timing,
        device=settings.DEVICE,
    )


@router.post("/analyze/webcam", response_model=AnalysisResponse)
async def analyze_webcam(
    webcam_data: str = Form(...),
    mode: str = Form("general"),
    confidence: float = Form(0.35),
):
    """Analyze a webcam capture (base64 encoded)."""
    return await analyze_image(
        webcam_data=webcam_data,
        mode=mode,
        confidence=confidence,
    )
