import uuid
import uvicorn
import asyncio

from datetime import datetime
from core.config import settings
from utils.logging import setup_logging
from core.database import init_db, get_db
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from ml_services.orchestrator import MLOrchestrator
from api import matching, datasets, models, monitoring, feedback
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks

logger = setup_logging()

ml_orchestrator = MLOrchestrator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("Starting MDM RL System...")
    await init_db()
    await ml_orchestrator.initialize()
    yield
    logger.info("Shutting down MDM RL System...")
    await ml_orchestrator.shutdown()

app = FastAPI(
    title="MDM RL System",
    description="Adaptive Data Matching Rules Management System using Multi-Paradigm RL",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(matching.router, prefix="/api/v1/matching", tags=["matching"])
app.include_router(datasets.router, prefix="/api/v1/datasets", tags=["datasets"])
app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
app.include_router(monitoring.router, prefix="/api/v1/monitoring", tags=["monitoring"])
app.include_router(feedback.router, prefix="/api/v1/feedback", tags=["feedback"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MDM RL System API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        async with get_db() as db:
            await db.execute("SELECT 1")

        ml_status = await ml_orchestrator.health_check()

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "database": "healthy",
                "ml_services": ml_status
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.post("/api/v1/upload")
async def upload_dataset(
        file: UploadFile = File(...),
        dataset_name: str = None,
        dataset_type: str = "customer",
        background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload a dataset for processing"""
    try:
        dataset_id = str(uuid.uuid4())

        file_path = f"/app/data/uploads/{dataset_id}_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        background_tasks.add_task(
            ml_orchestrator.process_dataset,
            dataset_id=dataset_id,
            file_path=file_path,
            dataset_name=dataset_name or file.filename,
            dataset_type=dataset_type
        )

        return {
            "dataset_id": dataset_id,
            "message": "Dataset uploaded successfully. Processing started.",
            "status": "processing"
        }

    except Exception as e:
        logger.error(f"Dataset upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/status/{dataset_id}")
async def get_processing_status(dataset_id: str):
    """Get processing status for a dataset"""
    try:
        status = await ml_orchestrator.get_dataset_status(dataset_id)
        if not status:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return status
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Not found", "detail": str(exc)}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal error: {exc}")
    return {"error": "Internal server error", "detail": "An unexpected error occurred"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )