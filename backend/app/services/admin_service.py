from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.evaluation import EvaluationResult, EvaluationRun
from app.models.metrics import RequestMetric


async def get_latency_stats(session: AsyncSession) -> list[dict]:
    """Per-endpoint request count and p50/p95 total latency, computed in SQL
    via percentile_cont over real persisted RequestMetric rows (Phase 6's
    "sourced from real logged data" exit criterion) — not estimated or
    hardcoded, and not recomputed in Python from a fetched-then-sorted list.
    """
    p50 = func.percentile_cont(0.5).within_group(RequestMetric.total_ms)
    p95 = func.percentile_cont(0.95).within_group(RequestMetric.total_ms)
    stmt = (
        select(
            RequestMetric.endpoint,
            func.count().label("count"),
            p50.label("p50_total_ms"),
            p95.label("p95_total_ms"),
        )
        .group_by(RequestMetric.endpoint)
        .order_by(RequestMetric.endpoint)
    )
    rows = (await session.execute(stmt)).all()
    return [
        {
            "endpoint": row.endpoint,
            "count": row.count,
            "p50_total_ms": float(row.p50_total_ms),
            "p95_total_ms": float(row.p95_total_ms),
        }
        for row in rows
    ]


async def get_recent_requests(session: AsyncSession, limit: int = 20) -> list[RequestMetric]:
    stmt = select(RequestMetric).order_by(RequestMetric.created_at.desc()).limit(limit)
    return list((await session.execute(stmt)).scalars().all())


async def list_evaluation_runs(session: AsyncSession, limit: int = 20) -> list[dict]:
    """Past EvaluationRun rows (Phase 5, eval/runners/*.py) with a generic mean
    of every numeric key found across their EvaluationResult.metrics JSONB —
    generic on purpose, so it works for any experiment's metric set (retrieval
    metrics today, generation metrics once OpenAI credits unblock those —
    see docs/EVALUATION.md) without this endpoint hardcoding metric names.
    """
    runs = (
        (
            await session.execute(
                select(EvaluationRun).order_by(EvaluationRun.started_at.desc()).limit(limit)
            )
        )
        .scalars()
        .all()
    )

    output = []
    for run in runs:
        results = (
            (
                await session.execute(
                    select(EvaluationResult).where(EvaluationResult.evaluation_run_id == run.id)
                )
            )
            .scalars()
            .all()
        )
        output.append(
            {
                "id": run.id,
                "name": run.name,
                "status": run.status,
                "started_at": run.started_at,
                "finished_at": run.finished_at,
                "result_count": len(results),
                "mean_metrics": _mean_metrics(results),
            }
        )
    return output


def _mean_metrics(results: list[EvaluationResult]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for result in results:
        for key, value in result.metrics.items():
            if isinstance(value, int | float):
                sums[key] = sums.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums}
