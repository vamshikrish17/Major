"""
VisionExtract 2.0 — Results Router
Endpoints for analysis history and result retrieval.
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import desc

from ..models.database import AnalysisRecord, get_new_session
from ..models.schemas import AnalysisListItem, AnalysisListResponse

logger = logging.getLogger("visionextract.api.results")
router = APIRouter(prefix="/api", tags=["results"])


@router.get("/results", response_model=AnalysisListResponse)
async def list_results(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    mode: Optional[str] = Query(None),
):
    """List all past analyses with pagination."""
    try:
        session = get_new_session()
        query = session.query(AnalysisRecord)

        if mode:
            query = query.filter(AnalysisRecord.mode == mode)

        total = query.count()
        records = (
            query.order_by(desc(AnalysisRecord.timestamp))
            .offset((page - 1) * per_page)
            .limit(per_page)
            .all()
        )

        from ..services.storage import storage

        analyses = [
            AnalysisListItem(
                analysis_id=r.id,
                timestamp=r.timestamp.isoformat() if r.timestamp else "",
                mode=r.mode or "general",
                filename=r.input_filename or "",
                num_objects=r.object_count or 0,
                processing_time_ms=r.processing_time_ms or 0.0,
                original_url=storage.get_url(r.original_path) if r.original_path else None,
                overlay_url=storage.get_url(r.overlay_path) if r.overlay_path else None,
            )
            for r in records
        ]

        session.close()

        return AnalysisListResponse(
            total=total,
            page=page,
            per_page=per_page,
            analyses=analyses,
        )
    except Exception as e:
        logger.error(f"Error listing results: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve results")


@router.get("/results/{analysis_id}")
async def get_result(analysis_id: str):
    """Retrieve a specific analysis result by ID."""
    try:
        session = get_new_session()
        record = session.query(AnalysisRecord).filter_by(id=analysis_id).first()
        session.close()

        if not record:
            raise HTTPException(status_code=404, detail="Analysis not found")

        from ..services.storage import storage
        features = record.get_features()

        return {
            "analysis_id": record.id,
            "timestamp": record.timestamp.isoformat() if record.timestamp else "",
            "mode": record.mode,
            "filename": record.input_filename,
            "original_url": storage.get_url(record.original_path) if record.original_path else None,
            "overlay_url": storage.get_url(record.overlay_path) if record.overlay_path else None,
            "num_objects": record.object_count,
            "processing_time_ms": record.processing_time_ms,
            "device": record.device,
            "features": features,
            "quality_report": features.get("quality_report", {}),
            "domain_insight": record.get_domain(),
            "segments": record.get_segments(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving result: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve result")


@router.delete("/results/{analysis_id}")
async def delete_result(analysis_id: str):
    """Delete a specific analysis result."""
    try:
        session = get_new_session()
        record = session.query(AnalysisRecord).filter_by(id=analysis_id).first()

        if not record:
            session.close()
            raise HTTPException(status_code=404, detail="Analysis not found")

        # Delete associated files
        from ..services.storage import storage
        if record.original_path:
            storage.delete_file(record.original_path)
        if record.overlay_path:
            storage.delete_file(record.overlay_path)

        session.delete(record)
        session.commit()
        session.close()

        return {"status": "deleted", "analysis_id": analysis_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting result: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete result")
