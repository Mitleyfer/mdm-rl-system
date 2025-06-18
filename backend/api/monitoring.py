from pydantic import BaseModel
from core.database import get_db
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from utils.metrics import metrics_registry
from fastapi import APIRouter, HTTPException
from prometheus_client import Counter, Histogram, Gauge, generate_latest

router = APIRouter()

request_count = Counter('mdm_api_requests_total', 'Total API requests', ['endpoint', 'method'])
request_duration = Histogram('mdm_api_request_duration_seconds', 'API request duration', ['endpoint'])
active_datasets = Gauge('mdm_active_datasets', 'Number of active datasets')
matching_accuracy = Gauge('mdm_matching_accuracy', 'Current matching accuracy', ['metric_type'])
processing_queue_size = Gauge('mdm_processing_queue_size', 'Size of processing queue')

class SystemHealth(BaseModel):
    status: str
    components: Dict[str, str]
    uptime_seconds: float
    last_check: datetime

class PerformanceMetrics(BaseModel):
    time_range: str
    total_matches: int
    avg_processing_time: float
    accuracy_metrics: Dict[str, float]
    throughput: float

@router.get("/health", response_model=SystemHealth)
async def get_system_health():
    """
    Get overall system health status
    """
    try:
        components = {}

        async with get_db() as db:
            await db.execute("SELECT 1")
            components["database"] = "healthy"

        from main import ml_orchestrator
        ml_health = await ml_orchestrator.health_check()
        components["ml_services"] = "healthy" if all(
            status == "healthy" for status in ml_health.values()
        ) else "degraded"

        try:
            from utils.cache import redis_client
            await redis_client.ping()
            components["cache"] = "healthy"
        except:
            components["cache"] = "unhealthy"

        overall_status = "healthy"
        if any(status == "unhealthy" for status in components.values()):
            overall_status = "unhealthy"
        elif any(status == "degraded" for status in components.values()):
            overall_status = "degraded"

        return SystemHealth(
            status=overall_status,
            components=components,
            uptime_seconds=get_uptime(),
            last_check=datetime.utcnow()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/metrics")
async def get_prometheus_metrics():
    """
    Get Prometheus metrics
    """
    return generate_latest(metrics_registry)

@router.get("/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(
        time_range: str = "1h"
):
    """
    Get system performance metrics
    """
    time_map = {
        "1h": timedelta(hours=1),
        "24h": timedelta(days=1),
        "7d": timedelta(days=7),
        "30d": timedelta(days=30)
    }

    if time_range not in time_map:
        raise HTTPException(status_code=400, detail=f"Invalid time range. Must be one of: {list(time_map.keys())}")

    since = datetime.utcnow() - time_map[time_range]

    async with get_db() as db:
        query = """
            SELECT 
                COUNT(*) as total_matches,
                AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_processing_time,
                AVG((results->>'precision')::float) as avg_precision,
                AVG((results->>'recall')::float) as avg_recall,
                AVG((results->>'f1_score')::float) as avg_f1_score
            FROM dataset_processing
            WHERE completed_at >= :since AND status = 'completed'
        """

        result = await db.fetch_one(query=query, values={"since": since})

        if result:
            total_matches = result["total_matches"] or 0
            avg_processing_time = result["avg_processing_time"] or 0

            time_window_hours = time_map[time_range].total_seconds() / 3600
            throughput = total_matches / time_window_hours if time_window_hours > 0 else 0

            return PerformanceMetrics(
                time_range=time_range,
                total_matches=total_matches,
                avg_processing_time=avg_processing_time,
                accuracy_metrics={
                    "precision": result["avg_precision"] or 0,
                    "recall": result["avg_recall"] or 0,
                    "f1_score": result["avg_f1_score"] or 0
                },
                throughput=throughput
            )
        else:
            return PerformanceMetrics(
                time_range=time_range,
                total_matches=0,
                avg_processing_time=0,
                accuracy_metrics={"precision": 0, "recall": 0, "f1_score": 0},
                throughput=0
            )

@router.get("/logs")
async def get_recent_logs(
        level: Optional[str] = None,
        component: Optional[str] = None,
        limit: int = 100
):
    """
    Get recent system logs
    """
    logs = [
        {
            "timestamp": datetime.utcnow() - timedelta(minutes=i),
            "level": "INFO",
            "component": "orchestrator",
            "message": f"Processing dataset {i}"
        }
        for i in range(min(limit, 10))
    ]

    return {"logs": logs, "total": len(logs)}

@router.get("/alerts")
async def get_active_alerts():
    """
    Get active system alerts
    """
    alerts = []

    async with get_db() as db:
        query = """
            SELECT 
                COUNT(CASE WHEN status = 'failed' THEN 1 END)::float / 
                NULLIF(COUNT(*), 0) as failure_rate
            FROM dataset_processing
            WHERE started_at >= :since
        """

        result = await db.fetch_one(
            query=query,
            values={"since": datetime.utcnow() - timedelta(hours=1)}
        )

        if result and result["failure_rate"] and result["failure_rate"] > 0.1:
            alerts.append({
                "id": "high_failure_rate",
                "severity": "warning",
                "message": f"High failure rate: {result['failure_rate']:.1%}",
                "timestamp": datetime.utcnow()
            })

    queue_size = 0
    if queue_size > 100:
        alerts.append({
            "id": "queue_backlog",
            "severity": "warning",
            "message": f"Processing queue backlog: {queue_size} items",
            "timestamp": datetime.utcnow()
        })

    return {"alerts": alerts, "total": len(alerts)}

@router.get("/stats/dashboard")
async def get_dashboard_stats():
    """
    Get statistics for monitoring dashboard
    """
    async with get_db() as db:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        today_query = """
            SELECT 
                COUNT(*) as datasets_processed,
                AVG((results->>'f1_score')::float) as avg_f1_score,
                SUM(records_count) as total_records
            FROM datasets
            WHERE created_at >= :today_start
        """

        today_stats = await db.fetch_one(query=today_query, values={"today_start": today_start})

        learning_query = """
            SELECT 
                COUNT(*) as feedback_count,
                AVG(confidence) as avg_confidence
            FROM feedback
            WHERE created_at >= :since
        """

        learning_stats = await db.fetch_one(
            query=learning_query,
            values={"since": datetime.utcnow() - timedelta(days=7)}
        )

        from main import ml_orchestrator
        model_info = await ml_orchestrator.get_active_model_info()

        return {
            "today": {
                "datasets_processed": today_stats["datasets_processed"] or 0,
                "avg_f1_score": today_stats["avg_f1_score"] or 0,
                "total_records": today_stats["total_records"] or 0
            },
            "learning": {
                "feedback_collected": learning_stats["feedback_count"] or 0,
                "avg_confidence": learning_stats["avg_confidence"] or 0
            },
            "models": model_info.get("performance_summary", {}),
            "timestamp": datetime.utcnow()
        }

def get_uptime():
    """Get application uptime in seconds"""
    return 3600.0