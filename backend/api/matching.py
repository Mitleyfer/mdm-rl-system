import uuid

from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from core.database import get_db, get_active_rules
from ml_services.matching_engine import MatchingEngine
from utils.cache import cache_result, get_cached_result
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends

router = APIRouter()

class RecordModel(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    address_number: Optional[str] = None
    street_name: Optional[str] = None
    street_type: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    additional_fields: Optional[Dict[str, Any]] = Field(default_factory=dict)

class MatchPairRequest(BaseModel):
    record1: RecordModel
    record2: RecordModel
    use_custom_rules: bool = False
    custom_rules: Optional[Dict[str, Any]] = None

class BatchMatchRequest(BaseModel):
    query_record: RecordModel
    candidate_records: List[RecordModel]
    max_matches: int = Field(default=10, le=100)
    min_score: float = Field(default=0.7, ge=0.0, le=1.0)
    use_custom_rules: bool = False
    custom_rules: Optional[Dict[str, Any]] = None

class MatchResponse(BaseModel):
    is_match: bool
    confidence: float
    similarity_scores: Dict[str, float]
    rule_version: Optional[int] = None

class BatchMatchResponse(BaseModel):
    query_record: RecordModel
    matches: List[Dict[str, Any]]
    total_candidates: int
    processing_time: float
    rule_version: Optional[int] = None

matching_engine = MatchingEngine()

@router.post("/match_pair", response_model=MatchResponse)
async def match_pair(request: MatchPairRequest):
    """
    Match two individual records
    """
    try:
        if request.use_custom_rules and request.custom_rules:
            rules = request.custom_rules
            rule_version = None
        else:
            rules = await get_active_rules()
            if not rules:
                rules = matching_engine.config['default_thresholds']
            rule_version = rules.get('version', None)

        record1_dict = request.record1.dict(exclude_none=True)
        record2_dict = request.record2.dict(exclude_none=True)

        cache_key = f"match_pair:{hash(str(record1_dict))}:{hash(str(record2_dict))}:{hash(str(rules))}"
        cached_result = await get_cached_result(cache_key)
        if cached_result:
            return MatchResponse(**cached_result)

        is_match, similarity = matching_engine.match_pair(record1_dict, record2_dict, rules)

        response = MatchResponse(
            is_match=is_match,
            confidence=similarity.get('overall', 0.0),
            similarity_scores=similarity,
            rule_version=rule_version
        )

        await cache_result(cache_key, response.dict(), ttl=3600)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matching failed: {str(e)}")

@router.post("/batch_match", response_model=BatchMatchResponse)
async def batch_match(request: BatchMatchRequest):
    """
    Find all matches for a query record from a list of candidates
    """
    try:
        start_time = datetime.now()

        if request.use_custom_rules and request.custom_rules:
            rules = request.custom_rules
            rule_version = None
        else:
            rules = await get_active_rules()
            if not rules:
                rules = matching_engine.config['default_thresholds']
            rule_version = rules.get('version', None)

        query_dict = request.query_record.dict(exclude_none=True)
        candidates_dict = [r.dict(exclude_none=True) for r in request.candidate_records]

        matches = matching_engine.batch_match(query_dict, candidates_dict, rules)

        filtered_matches = []
        for idx, similarity in matches:
            if similarity.get('overall', 0) >= request.min_score:
                filtered_matches.append({
                    'candidate_index': idx,
                    'candidate_record': request.candidate_records[idx].dict(),
                    'confidence': similarity.get('overall', 0),
                    'similarity_scores': similarity
                })

                if len(filtered_matches) >= request.max_matches:
                    break

        processing_time = (datetime.now() - start_time).total_seconds()

        return BatchMatchResponse(
            query_record=request.query_record,
            matches=filtered_matches,
            total_candidates=len(request.candidate_records),
            processing_time=processing_time,
            rule_version=rule_version
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch matching failed: {str(e)}")

@router.post("/deduplicate")
async def find_duplicates(
        records: List[RecordModel],
        background_tasks: BackgroundTasks,
        min_confidence: float = 0.8
):
    """
    Find duplicates within a list of records
    """
    try:
        job_id = str(uuid.uuid4())

        background_tasks.add_task(
            process_deduplication,
            job_id=job_id,
            records=records,
            min_confidence=min_confidence
        )

        return {
            "job_id": job_id,
            "status": "processing",
            "message": f"Deduplication started for {len(records)} records"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deduplication failed: {str(e)}")

async def process_deduplication(job_id: str, records: List[RecordModel], min_confidence: float):
    """
    Background task for deduplication
    """
    try:
        rules = await get_active_rules() or matching_engine.config['default_thresholds']

        records_dict = [r.dict(exclude_none=True) for r in records]

        data = {
            'records': records_dict,
            'dataset_type': 'deduplication'
        }

        results = await matching_engine.match_dataset(data, rules)

        duplicate_groups = []
        processed_indices = set()

        for i, j in results['matches']:
            if results['matches'][(i, j)].get('overall', 0) >= min_confidence:
                group_found = False
                for group in duplicate_groups:
                    if i in group or j in group:
                        group.add(i)
                        group.add(j)
                        group_found = True
                        break

                if not group_found:
                    duplicate_groups.append({i, j})

                processed_indices.add(i)
                processed_indices.add(j)

        duplicate_list = [
            {
                'group_id': idx,
                'records': [records[i].dict() for i in group],
                'indices': list(group)
            }
            for idx, group in enumerate(duplicate_groups)
        ]

        await cache_result(
            f"dedup_job:{job_id}",
            {
                'status': 'completed',
                'duplicate_groups': duplicate_list,
                'total_records': len(records),
                'duplicates_found': len(processed_indices),
                'processing_time': results['metrics']['processing_time']
            },
            ttl=7200
        )

    except Exception as e:
        await cache_result(
            f"dedup_job:{job_id}",
            {
                'status': 'failed',
                'error': str(e)
            },
            ttl=3600
        )

@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """
    Get status of a matching job
    """
    result = await get_cached_result(f"dedup_job:{job_id}")

    if not result:
        raise HTTPException(status_code=404, detail="Job not found")

    return result

@router.get("/similarity_metrics")
async def get_similarity_metrics():
    """
    Get available similarity metrics and their descriptions
    """
    return {
        "metrics": [
            {
                "name": "exact",
                "description": "Exact string matching",
                "applicable_to": ["all fields"]
            },
            {
                "name": "fuzzy",
                "description": "Jaro-Winkler string similarity",
                "applicable_to": ["names", "addresses", "text fields"]
            },
            {
                "name": "phonetic",
                "description": "Soundex phonetic matching",
                "applicable_to": ["names"]
            },
            {
                "name": "numeric",
                "description": "Numeric distance and equality",
                "applicable_to": ["phone", "zip", "numeric fields"]
            },
            {
                "name": "semantic",
                "description": "Semantic similarity using embeddings",
                "applicable_to": ["descriptions", "text fields"]
            }
        ]
    }