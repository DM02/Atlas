import logging

import structlog


def configure_logging() -> None:
    """JSON structured logging via structlog. Called once at app startup
    (main.py) and by the Arq worker entrypoint — both processes need their
    own logs, and structlog's global configuration is per-process.

    contextvars-bound fields (request_id in the HTTP path, document_version_id
    in the worker — see core/middleware.py and workers/ingestion_worker.py)
    are merged into every log line automatically via merge_contextvars,
    without threading a logger/context object through every function call.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
