import uuid

from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from core.database import get_db, save_feedback, get_recent_feedback

router = APIRouter()

class FeedbackSubmission(BaseModel):
    record_pair: Dict[str, Any]
    feedback_type: str = Field(..., pattern="^(match|no_match|uncertain)$")
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    dataset_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class PreferenceFeedback(BaseModel):
    state: Dict[str, Any]
    option_a: Dict[str, Any]
    option_b: Dict[str, Any]
    preference: str = Field(..., pattern="^(a|b|equal)$")
    user_id: Optional[str] = None
    reason: Optional[str] = None

class FeedbackResponse(BaseModel):
    id: int
    record_pair: Dict[str, Any]
    feedback_type: str
    confidence: float
    created_at: datetime
    dataset_id: Optional[str]
    user_id: Optional[str]
    metadata: Optional[Dict[str, Any]]

@router.post("/submit", response_model=Dict[str, Any])
async def submit_feedback(
        feedback: FeedbackSubmission,
        background_tasks: BackgroundTasks
):
    """
    Submit feedback for a matching decision
    """
    try:
        feedback_data = {
            "record_pair": feedback.record_pair,
            "feedback_type": feedback.feedback_type,
            "confidence": feedback.confidence,
            "dataset_id": feedback.dataset_id,
            "user_id": feedback.user_id or "anonymous",
            "metadata": feedback.metadata or {}
        }

        feedback_id = await save_feedback(feedback_data)

        background_tasks.add_task(
            update_rlhf_agent,
            feedback_data=feedback_data
        )

        return {
            "id": feedback_id,
            "message": "Feedback submitted successfully",
            "status": "accepted"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

@router.post("/preference")
async def submit_preference_feedback(
        preference: PreferenceFeedback,
        background_tasks: BackgroundTasks
):
    """
    Submit preference feedback between two options
    """
    try:
        preference_value = {
            "a": 0,
            "b": 1,
            "equal": 0.5
        }[preference.preference]

        feedback_data = {
            "record_pair": {
                "state": preference.state,
                "option_a": preference.option_a,
                "option_b": preference.option_b
            },
            "feedback_type": "preference",
            "confidence": preference_value,
            "user_id": preference.user_id or "anonymous",
            "metadata": {
                "preference": preference.preference,
                "reason": preference.reason
            }
        }

        feedback_id = await save_feedback(feedback_data)

        background_tasks.add_task(
            update_rlhf_preference,
            preference_data=feedback_data
        )

        return {
            "id": feedback_id,
            "message": "Preference feedback submitted successfully",
            "status": "accepted"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit preference: {str(e)}")

@router.get("/recent", response_model=List[FeedbackResponse])
async def get_recent_feedback_entries(
        limit: int = 100,
        feedback_type: Optional[str] = None,
        user_id: Optional[str] = None
):
    """
    Get recent feedback entries
    """
    feedback_entries = await get_recent_feedback(limit)

    filtered = []
    for entry in feedback_entries:
        if feedback_type and entry["feedback_type"] != feedback_type:
            continue
        if user_id and entry["user_id"] != user_id:
            continue

        filtered.append(FeedbackResponse(
            id=entry["id"],
            record_pair=entry["record_pair"],
            feedback_type=entry["feedback_type"],
            confidence=entry["confidence"],
            created_at=entry["created_at"],
            dataset_id=entry["dataset_id"],
            user_id=entry["user_id"],
            metadata=entry["metadata"]
        ))

    return filtered

@router.get("/stats")
async def get_feedback_statistics():
    """
    Get feedback statistics
    """
    async with get_db() as db:
        total_query = """
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT user_id) as unique_users,
                AVG(confidence) as avg_confidence
            FROM feedback
        """
        total_stats = await db.fetch_one(query=total_query)

        type_query = """
            SELECT 
                feedback_type,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence
            FROM feedback
            GROUP BY feedback_type
        """
        type_stats = await db.fetch_all(query=type_query)

        recent_query = """
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as count
            FROM feedback
            WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY DATE(created_at)
            ORDER BY date DESC
        """
        recent_activity = await db.fetch_all(query=recent_query)

        return {
            "total_feedback": total_stats["total"] or 0,
            "unique_users": total_stats["unique_users"] or 0,
            "avg_confidence": total_stats["avg_confidence"] or 0,
            "by_type": [dict(row) for row in type_stats],
            "recent_activity": [dict(row) for row in recent_activity]
        }

@router.post("/request_feedback")
async def request_human_feedback(
        dataset_id: str,
        sample_size: int = 10
):
    """
    Request feedback for uncertain matches in a dataset
    """

    uncertain_matches = []

    for i in range(sample_size):
        uncertain_matches.append({
            "id": str(uuid.uuid4()),
            "record1": {
                "first_name": f"John{i}",
                "last_name": "Smith",
                "address": f"{100+i} Main St"
            },
            "record2": {
                "first_name": "John",
                "last_name": f"Smith{i}",
                "address": f"{100+i} Main Street"
            },
            "current_confidence": 0.6 + (i * 0.02),
            "uncertainty_reason": "Name similarity high but address differs"
        })

    return {
        "dataset_id": dataset_id,
        "feedback_requests": uncertain_matches,
        "total": len(uncertain_matches),
        "message": "Please review these uncertain matches"
    }

async def update_rlhf_agent(feedback_data: Dict):
    """
    Update RLHF agent with new feedback
    """
    try:
        from main import ml_orchestrator

        rlhf_agent = ml_orchestrator.agents.get("rlhf")
        if rlhf_agent and hasattr(rlhf_agent, 'feedback_buffer'):
            rlhf_agent.feedback_buffer.add_feedback(
                state={"record_pair": feedback_data["record_pair"]},
                action={"decision": feedback_data["feedback_type"]},
                feedback=feedback_data["confidence"],
                metadata=feedback_data.get("metadata", {})
            )

    except Exception as e:
        print(f"Failed to update RLHF agent: {e}")

async def update_rlhf_preference(preference_data: Dict):
    """
    Update RLHF agent with preference feedback
    """
    try:
        from main import ml_orchestrator

        rlhf_agent = ml_orchestrator.agents.get("rlhf")
        if rlhf_agent and hasattr(rlhf_agent, 'feedback_buffer'):
            record_pair = preference_data["record_pair"]
            rlhf_agent.feedback_buffer.add_preference(
                state=record_pair["state"],
                action_a=record_pair["option_a"],
                action_b=record_pair["option_b"],
                preference=preference_data["confidence"]
            )

    except Exception as e:
        print(f"Failed to update RLHF preference: {e}")

@router.get("/export")
async def export_feedback(
        format: str = "json",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
):
    """
    Export feedback data for analysis
    """
    if format not in ["json", "csv"]:
        raise HTTPException(status_code=400, detail="Format must be 'json' or 'csv'")

    async with get_db() as db:
        query = "SELECT * FROM feedback WHERE 1=1"
        values = {}

        if start_date:
            query += " AND created_at >= :start_date"
            values["start_date"] = start_date

        if end_date:
            query += " AND created_at <= :end_date"
            values["end_date"] = end_date

        query += " ORDER BY created_at DESC"

        results = await db.fetch_all(query=query, values=values)

    if format == "json":
        return {
            "feedback": [dict(row) for row in results],
            "total": len(results),
            "export_date": datetime.utcnow()
        }
    else:
        import csv
        import io

        output = io.StringIO()
        if results:
            writer = csv.DictWriter(output, fieldnames=results[0].keys())
            writer.writeheader()
            for row in results:
                writer.writerow(dict(row))

        return {
            "csv_data": output.getvalue(),
            "total": len(results)
        }