"""
VisionExtract 2.0 — Health Router
System health check endpoint.
"""

import logging
import platform

import psutil
import torch
from fastapi import APIRouter

from ..models.schemas import HealthResponse
from ..services.model_manager import ModelManager

logger = logging.getLogger("visionextract.api.health")
router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    System health status: device info, model status, memory usage, uptime.
    """
    manager = ModelManager.get_instance()

    # Memory usage
    process = psutil.Process()
    mem_mb = process.memory_info().rss / (1024 * 1024)

    # GPU info
    gpu_available = torch.cuda.is_available()
    device = manager.device

    return HealthResponse(
        status="healthy" if manager.is_loaded else "loading",
        device=device,
        gpu_available=gpu_available,
        models_loaded=manager.models_status,
        memory_usage_mb=round(mem_mb, 1),
        uptime_seconds=round(manager.uptime, 1),
    )
