import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class RequestMetric(Base):
    """One row per instrumented request/job, backing the admin observability
    dashboard's p50/p95 latency view (Phase 6, docs/ARCHITECTURE.md §9).

    `stage_latencies_ms` is JSONB rather than per-stage columns — different
    endpoints have different stages (chat_query: retrieve_ms/generate_ms;
    document_ingestion: extract_ms/chunk_ms/embed_ms) and this is the same
    "write-once report data, don't normalize per metric" call already made
    for EvaluationResult.metrics (app/models/evaluation.py). `total_ms` is
    its own column (not just a JSONB key) specifically so Postgres can
    compute percentile_cont(...) over it directly in SQL without reaching
    into JSONB for the one aggregate that matters most.
    """

    __tablename__ = "request_metrics"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    endpoint: Mapped[str] = mapped_column(String(64), index=True)
    stage_latencies_ms: Mapped[dict[str, Any]] = mapped_column(JSONB)
    total_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
