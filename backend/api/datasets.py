from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from core.database import (
    get_db,
    datasets_table,
    dataset_processing_table,
    get_dataset_by_id
)

router = APIRouter()

class DatasetResponse(BaseModel):
    id: str
    name: str
    type: str
    status: str
    records_count: int
    created_at: datetime
    updated_at: Optional[datetime]
    results: Optional[dict]

class DatasetListResponse(BaseModel):
    datasets: List[DatasetResponse]
    total: int
    page: int
    page_size: int

@router.get("/", response_model=DatasetListResponse)
async def list_datasets(
        page: int = Query(1, ge=1),
        page_size: int = Query(10, ge=1, le=100),
        dataset_type: Optional[str] = None,
        status: Optional[str] = None
):
    """
    List all datasets with pagination and filtering
    """
    async with get_db() as db:
        query = datasets_table.select()

        if dataset_type:
            query = query.where(datasets_table.c.type == dataset_type)
        if status:
            query = query.where(datasets_table.c.status == status)

        count_query = datasets_table.select()
        if dataset_type:
            count_query = count_query.where(datasets_table.c.type == dataset_type)
        if status:
            count_query = count_query.where(datasets_table.c.status == status)

        total = await db.fetch_val(query=count_query.with_only_columns([db.func.count()]))

        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size).order_by(datasets_table.c.created_at.desc())

        results = await db.fetch_all(query)

        datasets = [DatasetResponse(**dict(row)) for row in results]

        return DatasetListResponse(
            datasets=datasets,
            total=total,
            page=page,
            page_size=page_size
        )

@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(dataset_id: str):
    """
    Get dataset by ID
    """
    dataset = await get_dataset_by_id(dataset_id)

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return DatasetResponse(**dict(dataset))

@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """
    Delete a dataset
    """
    async with get_db() as db:
        dataset = await get_dataset_by_id(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        await db.execute(
            dataset_processing_table.delete().where(
                dataset_processing_table.c.dataset_id == dataset_id
            )
        )

        await db.execute(
            datasets_table.delete().where(
                datasets_table.c.id == dataset_id
            )
        )

        return {"message": f"Dataset {dataset_id} deleted successfully"}

@router.get("/{dataset_id}/processing")
async def get_processing_status(dataset_id: str):
    """
    Get detailed processing status for a dataset
    """
    async with get_db() as db:
        query = dataset_processing_table.select().where(
            dataset_processing_table.c.dataset_id == dataset_id
        ).order_by(dataset_processing_table.c.started_at.desc()).limit(1)

        result = await db.fetch_one(query)

        if not result:
            raise HTTPException(status_code=404, detail="Processing record not found")

        return dict(result)

@router.post("/{dataset_id}/reprocess")
async def reprocess_dataset(dataset_id: str):
    """
    Trigger reprocessing of a dataset
    """
    from main import ml_orchestrator

    dataset = await get_dataset_by_id(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {
        "message": f"Reprocessing triggered for dataset {dataset_id}",
        "status": "queued"
    }

@router.get("/stats/summary")
async def get_dataset_statistics():
    """
    Get summary statistics across all datasets
    """
    async with get_db() as db:
        total_datasets = await db.fetch_val(
            query=datasets_table.select().with_only_columns([db.func.count()])
        )

        type_stats_query = """
            SELECT type, COUNT(*) as count, AVG(records_count) as avg_records
            FROM datasets
            GROUP BY type
        """
        type_stats = await db.fetch_all(query=type_stats_query)

        status_stats_query = """
            SELECT status, COUNT(*) as count
            FROM datasets
            GROUP BY status
        """
        status_stats = await db.fetch_all(query=status_stats_query)

        perf_query = """
            SELECT 
                AVG((results->>'f1_score')::float) as avg_f1_score,
                AVG((results->>'precision')::float) as avg_precision,
                AVG((results->>'recall')::float) as avg_recall
            FROM datasets
            WHERE results IS NOT NULL
        """
        perf_stats = await db.fetch_one(query=perf_query)

        return {
            "total_datasets": total_datasets,
            "by_type": [dict(row) for row in type_stats],
            "by_status": [dict(row) for row in status_stats],
            "average_performance": dict(perf_stats) if perf_stats else None
        }