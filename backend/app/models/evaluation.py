import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.base import Base

STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class EvaluationRun(Base):
    """One execution of eval/runners/*.py against the golden dataset.

    `name` identifies which experiment/variant this run is (e.g.
    "hybrid_vs_vector:hybrid_rrf") — kept as its own indexed column rather
    than buried inside `config`, same rationale as `embedding_model` living
    on DocumentChunk directly (see docs/ARCHITECTURE.md's ERD notes):
    something reports/dashboards filter by shouldn't require parsing JSON.
    """

    __tablename__ = "evaluation_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(256))
    config: Mapped[dict[str, Any]] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(16), default=STATUS_RUNNING)
    started_at: Mapped[datetime] = mapped_column(server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)

    results: Mapped[list["EvaluationResult"]] = relationship(
        back_populates="evaluation_run", cascade="all, delete-orphan"
    )


class EvaluationResult(Base):
    """Per-question outcome within one EvaluationRun.

    `retrieved_chunk_ids` and `metrics` are JSONB rather than normalized
    columns/tables — this is write-once report data read back wholesale for
    docs/EVALUATION.md and the future admin dashboard (Phase 6), never
    queried or joined on individual metric values, so normalizing it would
    add schema churn (every new metric = a migration) for no query benefit.
    """

    __tablename__ = "evaluation_results"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    evaluation_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("evaluation_runs.id", ondelete="CASCADE")
    )
    question_id: Mapped[str] = mapped_column(String(64))
    retrieved_chunk_ids: Mapped[list[Any]] = mapped_column(JSONB)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    evaluation_run: Mapped["EvaluationRun"] = relationship(back_populates="results")
