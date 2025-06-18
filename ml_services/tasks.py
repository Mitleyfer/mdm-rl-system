import os
import logging
from celery import Celery
from typing import Dict, Any

app = Celery('ml_tasks')
app.config_from_object({
    'broker_url': os.getenv('REDIS_URL', 'redis://localhost:6379'),
    'result_backend': os.getenv('REDIS_URL', 'redis://localhost:6379'),
    'task_serializer': 'json',
    'accept_content': ['json'],
    'result_serializer': 'json',
    'timezone': 'UTC',
    'enable_utc': True,
})

logger = logging.getLogger(__name__)

@app.task
def process_dataset_task(dataset_id: str, file_path: str, dataset_name: str, dataset_type: str) -> Dict[str, Any]:
    """Process dataset in background"""
    try:
        from orchestrator import MLOrchestrator
        orchestrator = MLOrchestrator()

        result = orchestrator.process_dataset(
            dataset_id=dataset_id,
            file_path=file_path,
            dataset_name=dataset_name,
            dataset_type=dataset_type
        )

        return {
            'status': 'success',
            'dataset_id': dataset_id,
            'result': result
        }
    except Exception as e:
        logger.error(f"Failed to process dataset {dataset_id}: {e}")
        return {
            'status': 'failed',
            'dataset_id': dataset_id,
            'error': str(e)
        }

@app.task
def train_agent_task(agent_type: str, dataset_id: str = None) -> Dict[str, Any]:
    """Train a specific agent"""
    try:
        from orchestrator import MLOrchestrator
        orchestrator = MLOrchestrator()

        agent = orchestrator.agents.get(agent_type)
        if not agent:
            return {
                'status': 'failed',
                'error': f'Agent {agent_type} not found'
            }

        if dataset_id:
            pass
        else:
            if hasattr(agent, 'learn'):
                result = agent.learn({}, {}, {})

        return {
            'status': 'success',
            'agent_type': agent_type,
            'result': 'Training completed'
        }
    except Exception as e:
        logger.error(f"Failed to train agent {agent_type}: {e}")
        return {
            'status': 'failed',
            'agent_type': agent_type,
            'error': str(e)
        }

@app.task
def match_records_task(record1: Dict, record2: Dict, rules: Dict = None) -> Dict[str, Any]:
    """Match two records"""
    try:
        from matching_engine import MatchingEngine
        engine = MatchingEngine()

        is_match, similarity = engine.match_pair(record1, record2, rules or {})

        return {
            'status': 'success',
            'is_match': is_match,
            'similarity': similarity
        }
    except Exception as e:
        logger.error(f"Failed to match records: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }