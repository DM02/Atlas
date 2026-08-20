import pytest

from eval.metrics.latency import p50, p95, percentile


def test_percentile_p50_odd_count() -> None:
    assert percentile([1.0, 2.0, 3.0], 50) == 2.0


def test_percentile_p100_is_max() -> None:
    assert percentile([1.0, 5.0, 3.0], 100) == 5.0


def test_percentile_p0_is_min() -> None:
    assert percentile([1.0, 5.0, 3.0], 0) == 1.0


def test_percentile_empty_samples_is_zero() -> None:
    assert percentile([], 50) == 0.0


def test_percentile_rejects_out_of_range_p() -> None:
    with pytest.raises(ValueError, match="p must be between"):
        percentile([1.0], 150)


def test_p50_and_p95_helpers() -> None:
    samples = [float(i) for i in range(1, 101)]  # 1..100

    assert p50(samples) == 50.0
    assert p95(samples) == 95.0
