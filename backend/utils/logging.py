import sys
import json
import logging
import structlog

from typing import Dict, Any
from datetime import datetime
from core.config import settings

def setup_logging():
    """Setup structured logging configuration"""

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ],
            ),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer() if settings.LOG_FORMAT == "json" else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        stream=sys.stdout,
    )

    logger = structlog.get_logger()

    return logger

def get_logger(name: str):
    """Get a logger instance"""
    return structlog.get_logger(name)

class RequestLogger:
    """Middleware for logging HTTP requests"""

    def __init__(self):
        self.logger = get_logger("http")

    async def log_request(self, request, call_next):
        """Log incoming request and response"""
        start_time = datetime.utcnow()

        self.logger.info(
            "request_started",
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers),
            client=request.client.host if request.client else None,
        )

        response = await call_next(request)

        duration = (datetime.utcnow() - start_time).total_seconds()

        self.logger.info(
            "request_completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            duration=duration,
        )

        return response

class MLLogger:
    """Specialized logger for ML operations"""

    def __init__(self, component: str):
        self.logger = get_logger(f"ml.{component}")
        self.component = component

    def log_training_start(self, model_type: str, dataset_info: Dict[str, Any]):
        """Log training start"""
        self.logger.info(
            "training_started",
            model_type=model_type,
            dataset_size=dataset_info.get("size", 0),
            features=dataset_info.get("features", []),
        )

    def log_training_progress(self, epoch: int, metrics: Dict[str, float]):
        """Log training progress"""
        self.logger.info(
            "training_progress",
            epoch=epoch,
            **metrics
        )

    def log_training_complete(self, final_metrics: Dict[str, float], duration: float):
        """Log training completion"""
        self.logger.info(
            "training_completed",
            duration=duration,
            **final_metrics
        )

    def log_prediction(self, input_data: Dict[str, Any], prediction: Any, confidence: float):
        """Log prediction"""
        self.logger.debug(
            "prediction_made",
            input_hash=hash(str(input_data)),
            prediction=prediction,
            confidence=confidence,
        )

    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        self.logger.error(
            "error_occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {},
            exc_info=True,
        )

class DataLogger:
    """Logger for data processing operations"""

    def __init__(self):
        self.logger = get_logger("data")

    def log_dataset_loaded(self, dataset_id: str, stats: Dict[str, Any]):
        """Log dataset loading"""
        self.logger.info(
            "dataset_loaded",
            dataset_id=dataset_id,
            **stats
        )

    def log_processing_start(self, dataset_id: str, operation: str):
        """Log processing start"""
        self.logger.info(
            "processing_started",
            dataset_id=dataset_id,
            operation=operation,
        )

    def log_processing_complete(self, dataset_id: str, operation: str, results: Dict[str, Any]):
        """Log processing completion"""
        self.logger.info(
            "processing_completed",
            dataset_id=dataset_id,
            operation=operation,
            **results
        )

    def log_quality_issues(self, dataset_id: str, issues: list):
        """Log data quality issues"""
        self.logger.warning(
            "quality_issues_detected",
            dataset_id=dataset_id,
            issue_count=len(issues),
            issues=issues,
        )

request_logger = RequestLogger()
ml_logger = MLLogger("general")
data_logger = DataLogger()