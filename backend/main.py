"""
VisionExtract 2.0 — FastAPI Application Entry Point
High-performance async API with lifespan model management.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings, PROJECT_ROOT, STATIC_DIR
from .models.database import init_db
from .services.model_manager import ModelManager


# ─── Logging Setup ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("visionextract")


# ─── Application Lifespan ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    logger.info("╔════════════════════════════════════════════╗")
    logger.info("║     VisionExtract 2.0 — Starting Up       ║")
    logger.info("╚════════════════════════════════════════════╝")

    # Initialize database
    init_db(settings.DB_URL)
    logger.info("Database initialized")

    # Load AI models
    manager = ModelManager.get_instance()
    manager.load_models()

    yield  # Application runs here

    logger.info("VisionExtract 2.0 shutting down...")


# ─── FastAPI Application ───────────────────────────────────────
app = FastAPI(
    title="VisionExtract 2.0",
    description=(
        "AI-Powered Autonomous Visual Intelligence Platform — "
        "Real-time detection, segmentation, and scientific analysis "
        "across General, Biological, and Space domains."
    ),
    lifespan=lifespan,
)

# ─── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static Files ──────────────────────────────────────────────
# Serve uploaded/result images
static_dir = os.path.join(PROJECT_ROOT, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Serve frontend SPA
frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
if os.path.isdir(frontend_dir):
    app.mount("/app", StaticFiles(directory=frontend_dir, html=True), name="frontend")

# ─── API Routers ───────────────────────────────────────────────
from .routers import analysis, results, health

app.include_router(analysis.router)
app.include_router(results.router)
app.include_router(health.router)


# ─── Root Redirect ─────────────────────────────────────────────
@app.get("/")
async def root():
    """Redirect root to the frontend dashboard."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/app/index.html")


# ─── Run with uvicorn ──────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_level="info",
    )
