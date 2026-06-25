"""
Structured logging using structlog.
Every log event is JSON — parseable by CloudWatch Logs Insights.

Usage:
    from app.logger import get_logger
    log = get_logger(__name__)
    log.info("ingestion_completed", doc_id="abc", vectors=42, cost_usd=0.003)
"""
import logging
import structlog
from app.config import settings


def setup_logging() -> None:
    """Call once at app startup."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),   # output as JSON
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
