import logging
import databases
import sqlalchemy

from .config import settings
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base

logger = logging.getLogger(__name__)

database = databases.Database(settings.DATABASE_URL)

engine = create_engine(settings.DATABASE_URL)
metadata = MetaData()
Base = declarative_base(metadata=metadata)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

datasets_table = sqlalchemy.Table(
    "datasets",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.String, primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String, nullable=False),
    sqlalchemy.Column("type", sqlalchemy.String, nullable=False),
    sqlalchemy.Column("status", sqlalchemy.String, default="pending"),
    sqlalchemy.Column("records_count", sqlalchemy.Integer, default=0),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, server_default=sqlalchemy.func.now()),
    sqlalchemy.Column("updated_at", sqlalchemy.DateTime, onupdate=sqlalchemy.func.now()),
    sqlalchemy.Column("results", sqlalchemy.JSON, nullable=True),
)

dataset_processing_table = sqlalchemy.Table(
    "dataset_processing",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.String, primary_key=True),
    sqlalchemy.Column("dataset_id", sqlalchemy.String, sqlalchemy.ForeignKey("datasets.id")),
    sqlalchemy.Column("status", sqlalchemy.String, default="pending"),
    sqlalchemy.Column("progress", sqlalchemy.Integer, default=0),
    sqlalchemy.Column("started_at", sqlalchemy.DateTime, server_default=sqlalchemy.func.now()),
    sqlalchemy.Column("completed_at", sqlalchemy.DateTime, nullable=True),
    sqlalchemy.Column("error_message", sqlalchemy.Text, nullable=True),
    sqlalchemy.Column("results", sqlalchemy.JSON, nullable=True),
    sqlalchemy.Column("updated_at", sqlalchemy.DateTime, onupdate=sqlalchemy.func.now()),
)

matching_rules_table = sqlalchemy.Table(
    "matching_rules",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("name", sqlalchemy.String, nullable=False),
    sqlalchemy.Column("version", sqlalchemy.Integer, default=1),
    sqlalchemy.Column("rules", sqlalchemy.JSON, nullable=False),
    sqlalchemy.Column("performance_metrics", sqlalchemy.JSON, nullable=True),
    sqlalchemy.Column("created_by", sqlalchemy.String, nullable=True),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, server_default=sqlalchemy.func.now()),
    sqlalchemy.Column("is_active", sqlalchemy.Boolean, default=True),
)

feedback_table = sqlalchemy.Table(
    "feedback",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True, autoincrement=True),
    sqlalchemy.Column("dataset_id", sqlalchemy.String, nullable=True),
    sqlalchemy.Column("record_pair", sqlalchemy.JSON, nullable=False),
    sqlalchemy.Column("feedback_type", sqlalchemy.String, nullable=False),  # 'match', 'no_match', 'uncertain'
    sqlalchemy.Column("confidence", sqlalchemy.Float, nullable=True),
    sqlalchemy.Column("user_id", sqlalchemy.String, nullable=True),
    sqlalchemy.Column("created_at", sqlalchemy.DateTime, server_default=sqlalchemy.func.now()),
    sqlalchemy.Column("metadata", sqlalchemy.JSON, nullable=True),
)

async def init_db():
    """Initialize database connection and create tables"""
    try:
        await database.connect()
        logger.info("Database connection established")

        metadata.create_all(bind=engine)
        logger.info("Database tables created/verified")

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

async def close_db():
    """Close database connection"""
    await database.disconnect()
    logger.info("Database connection closed")

@asynccontextmanager
async def get_db():
    """Database session context manager"""
    async with database.transaction():
        yield database

async def get_dataset_by_id(dataset_id: str):
    """Get dataset by ID"""
    query = datasets_table.select().where(datasets_table.c.id == dataset_id)
    return await database.fetch_one(query)

async def create_dataset(dataset_data: dict):
    """Create new dataset record"""
    query = datasets_table.insert().values(**dataset_data)
    return await database.execute(query)

async def update_dataset_status(dataset_id: str, status: str, results: dict = None):
    """Update dataset status"""
    values = {"status": status, "updated_at": sqlalchemy.func.now()}
    if results:
        values["results"] = results

    query = datasets_table.update().where(
        datasets_table.c.id == dataset_id
    ).values(**values)

    return await database.execute(query)

async def get_active_rules():
    """Get current active matching rules"""
    query = matching_rules_table.select().where(
        matching_rules_table.c.is_active == True
    ).order_by(matching_rules_table.c.version.desc()).limit(1)

    result = await database.fetch_one(query)
    return dict(result["rules"]) if result else None

async def save_rules(rules: dict, created_by: str = None):
    """Save new rule configuration"""
    await database.execute(
        matching_rules_table.update().values(is_active=False)
    )

    query = sqlalchemy.select([sqlalchemy.func.max(matching_rules_table.c.version)])
    latest_version = await database.fetch_val(query) or 0

    query = matching_rules_table.insert().values(
        name=f"rules_v{latest_version + 1}",
        version=latest_version + 1,
        rules=rules,
        created_by=created_by,
        is_active=True
    )

    return await database.execute(query)

async def save_feedback(feedback_data: dict):
    """Save user feedback"""
    query = feedback_table.insert().values(**feedback_data)
    return await database.execute(query)

async def get_recent_feedback(limit: int = 100):
    """Get recent feedback entries"""
    query = feedback_table.select().order_by(
        feedback_table.c.created_at.desc()
    ).limit(limit)

    return await database.fetch_all(query)