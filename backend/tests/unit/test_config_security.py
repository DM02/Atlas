import structlog

from app.core.config import Settings, warn_if_insecure_defaults


def test_warns_when_insecure_default_used_outside_development() -> None:
    settings = Settings(environment="production")

    with structlog.testing.capture_logs() as logs:
        warn_if_insecure_defaults(settings)

    assert any(
        log["event"] == "insecure_jwt_secret_key_in_non_development_environment" for log in logs
    )


def test_no_warning_when_secret_overridden() -> None:
    settings = Settings(environment="production", jwt_secret_key="a-real-random-secret")

    with structlog.testing.capture_logs() as logs:
        warn_if_insecure_defaults(settings)

    assert logs == []


def test_no_warning_in_development_even_with_default_secret() -> None:
    settings = Settings(environment="development")

    with structlog.testing.capture_logs() as logs:
        warn_if_insecure_defaults(settings)

    assert logs == []
