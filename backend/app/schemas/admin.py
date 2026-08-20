import uuid
from datetime import datetime

from pydantic import BaseModel


class EndpointLatencyOut(BaseModel):
    endpoint: str
    count: int
    p50_total_ms: float
    p95_total_ms: float


class RecentRequestOut(BaseModel):
    endpoint: str
    total_ms: int
    stage_latencies_ms: dict[str, float]
    created_at: datetime


class MetricsOut(BaseModel):
    endpoints: list[EndpointLatencyOut]
    recent: list[RecentRequestOut]


class EvaluationRunOut(BaseModel):
    id: uuid.UUID
    name: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    result_count: int
    mean_metrics: dict[str, float]
