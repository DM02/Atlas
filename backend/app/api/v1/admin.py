from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_admin
from app.db.session import get_db
from app.models.user import User
from app.schemas.admin import EvaluationRunOut, MetricsOut, RecentRequestOut
from app.services import admin_service

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/metrics", response_model=MetricsOut)
async def get_metrics(
    current_user: User = Depends(require_admin), session: AsyncSession = Depends(get_db)
) -> MetricsOut:
    endpoints = await admin_service.get_latency_stats(session)
    recent = await admin_service.get_recent_requests(session)
    return MetricsOut(
        endpoints=endpoints,
        recent=[
            RecentRequestOut(
                endpoint=r.endpoint,
                total_ms=r.total_ms,
                stage_latencies_ms=r.stage_latencies_ms,
                created_at=r.created_at,
            )
            for r in recent
        ],
    )


@router.get("/evaluations", response_model=list[EvaluationRunOut])
async def get_evaluations(
    current_user: User = Depends(require_admin), session: AsyncSession = Depends(get_db)
) -> list[EvaluationRunOut]:
    return await admin_service.list_evaluation_runs(session)
