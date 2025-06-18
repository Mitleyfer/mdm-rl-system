import time
import asyncio

from functools import wraps
from typing import Dict, Any, Optional
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge, Summary

metrics_registry = CollectorRegistry()

matching_operations = Counter(
    'mdm_matching_operations_total',
    'Total number of matching operations',
    ['operation_type', 'status'],
    registry=metrics_registry
)

matching_duration = Histogram(
    'mdm_matching_duration_seconds',
    'Duration of matching operations',
    ['operation_type'],
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=metrics_registry
)

dataset_size = Histogram(
    'mdm_dataset_size_records',
    'Size of processed datasets',
    ['dataset_type'],
    buckets=(100, 500, 1000, 5000, 10000, 50000, 100000, 500000),
    registry=metrics_registry
)

model_performance = Gauge(
    'mdm_model_performance',
    'Model performance metrics',
    ['model_type', 'metric_name'],
    registry=metrics_registry
)

active_processing = Gauge(
    'mdm_active_processing_jobs',
    'Number of active processing jobs',
    registry=metrics_registry
)

rule_updates = Counter(
    'mdm_rule_updates_total',
    'Total number of rule updates',
    ['update_type'],
    registry=metrics_registry
)

feedback_received = Counter(
    'mdm_feedback_received_total',
    'Total feedback received',
    ['feedback_type'],
    registry=metrics_registry
)

cache_operations = Counter(
    'mdm_cache_operations_total',
    'Cache operations',
    ['operation', 'result'],
    registry=metrics_registry
)

def track_matching_operation(operation_type: str, status: str):
    """Track matching operation"""
    matching_operations.labels(operation_type=operation_type, status=status).inc()

def track_matching_duration(operation_type: str):
    """Decorator to track operation duration"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                matching_duration.labels(operation_type=operation_type).observe(
                    time.time() - start_time
                )
                track_matching_operation(operation_type, "success")
                return result
            except Exception as e:
                track_matching_operation(operation_type, "failure")
                raise e

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                matching_duration.labels(operation_type=operation_type).observe(
                    time.time() - start_time
                )
                track_matching_operation(operation_type, "success")
                return result
            except Exception as e:
                track_matching_operation(operation_type, "failure")
                raise e

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

def track_dataset_size(dataset_type: str, size: int):
    """Track dataset size"""
    dataset_size.labels(dataset_type=dataset_type).observe(size)

def update_model_performance(model_type: str, metrics: Dict[str, float]):
    """Update model performance metrics"""
    for metric_name, value in metrics.items():
        model_performance.labels(
            model_type=model_type,
            metric_name=metric_name
        ).set(value)

def increment_active_jobs():
    """Increment active processing jobs"""
    active_processing.inc()

def decrement_active_jobs():
    """Decrement active processing jobs"""
    active_processing.dec()

def track_rule_update(update_type: str = "manual"):
    """Track rule update"""
    rule_updates.labels(update_type=update_type).inc()

def track_feedback(feedback_type: str):
    """Track feedback received"""
    feedback_received.labels(feedback_type=feedback_type).inc()

def track_cache_operation(operation: str, hit: bool):
    """Track cache operation"""
    result = "hit" if hit else "miss"
    cache_operations.labels(operation=operation, result=result).inc()

async def get_model_metrics() -> Dict[str, Any]:
    """Get current model metrics"""
    metrics = {}

    for metric in metrics_registry.collect():
        if metric.name.startswith('mdm_model_performance'):
            for sample in metric.samples:
                model_type = sample.labels.get('model_type', 'unknown')
                metric_name = sample.labels.get('metric_name', 'unknown')

                if model_type not in metrics:
                    metrics[model_type] = {}

                metrics[model_type][metric_name] = sample.value

    return metrics

def calculate_metrics(matches: list, ground_truth: list) -> Dict[str, float]:
    """Calculate precision, recall, and F1 score"""
    if not matches and not ground_truth:
        return {
            'precision': 1.0,
            'recall': 1.0,
            'f1_score': 1.0
        }

    matches_set = set(matches)
    truth_set = set(ground_truth)

    true_positives = len(matches_set & truth_set)
    false_positives = len(matches_set - truth_set)
    false_negatives = len(truth_set - matches_set)

    precision = true_positives / (true_positives + false_positives) if matches_set else 0
    recall = true_positives / (true_positives + false_negatives) if truth_set else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

class MetricsCollector:
    """Context manager for collecting metrics"""

    def __init__(self, operation_type: str):
        self.operation_type = operation_type
        self.start_time = None
        self.metrics = {}

    def __enter__(self):
        self.start_time = time.time()
        increment_active_jobs()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        decrement_active_jobs()

        if exc_type is None:
            track_matching_operation(self.operation_type, "success")
        else:
            track_matching_operation(self.operation_type, "failure")

        matching_duration.labels(operation_type=self.operation_type).observe(duration)

        return False

    def add_metric(self, name: str, value: float):
        """Add a metric to be tracked"""
        self.metrics[name] = value

__all__ = [
    'track_matching_operation',
    'track_matching_duration',
    'track_dataset_size',
    'update_model_performance',
    'increment_active_jobs',
    'decrement_active_jobs',
    'track_rule_update',
    'track_feedback',
    'track_cache_operation',
    'get_model_metrics',
    'calculate_metrics',
    'MetricsCollector',
    'metrics_registry'
]