from collections.abc import AsyncGenerator

from httpx import ASGITransport, AsyncClient

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.main import app


def _wire_overrides(db_session, tmp_path) -> Settings:
    settings = Settings(storage_path=str(tmp_path))

    async def override_get_db() -> AsyncGenerator:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_settings] = lambda: settings
    return settings


async def test_first_registered_user_becomes_admin(db_session, tmp_path) -> None:
    _wire_overrides(db_session, tmp_path)
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            register = await client.post(
                "/api/v1/auth/register",
                json={"email": "admin@example.com", "password": "correct horse battery staple"},
            )
            assert register.status_code == 201
            token = register.json()["access_token"]

            me = await client.get(
                "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
            )
            assert me.status_code == 200
            assert me.json()["role"] == "admin"
    finally:
        app.dependency_overrides.clear()


async def test_second_registered_user_becomes_plain_user(db_session, tmp_path) -> None:
    _wire_overrides(db_session, tmp_path)
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/v1/auth/register",
                json={"email": "first@example.com", "password": "correct horse battery staple"},
            )
            second = await client.post(
                "/api/v1/auth/register",
                json={"email": "second@example.com", "password": "correct horse battery staple"},
            )
            token = second.json()["access_token"]

            me = await client.get(
                "/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
            )
            assert me.json()["role"] == "user"
    finally:
        app.dependency_overrides.clear()


async def test_register_rejects_duplicate_email(db_session, tmp_path) -> None:
    _wire_overrides(db_session, tmp_path)
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/v1/auth/register",
                json={"email": "dup@example.com", "password": "correct horse battery staple"},
            )
            second = await client.post(
                "/api/v1/auth/register",
                json={"email": "dup@example.com", "password": "another password entirely"},
            )
            assert second.status_code == 409
    finally:
        app.dependency_overrides.clear()


async def test_login_with_correct_and_wrong_password(db_session, tmp_path) -> None:
    _wire_overrides(db_session, tmp_path)
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post(
                "/api/v1/auth/register",
                json={"email": "login@example.com", "password": "correct horse battery staple"},
            )

            good = await client.post(
                "/api/v1/auth/login",
                data={"username": "login@example.com", "password": "correct horse battery staple"},
            )
            assert good.status_code == 200
            assert "access_token" in good.json()

            bad = await client.post(
                "/api/v1/auth/login",
                data={"username": "login@example.com", "password": "wrong password"},
            )
            assert bad.status_code == 401
    finally:
        app.dependency_overrides.clear()


async def test_me_without_token_is_unauthorized(db_session, tmp_path) -> None:
    _wire_overrides(db_session, tmp_path)
    transport = ASGITransport(app=app)

    try:
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/api/v1/auth/me")
            assert response.status_code == 401
    finally:
        app.dependency_overrides.clear()
