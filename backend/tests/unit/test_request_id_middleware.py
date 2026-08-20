from httpx import ASGITransport, AsyncClient

from app.main import app


async def test_response_includes_generated_request_id() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0


async def test_inbound_request_id_is_echoed_back() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health", headers={"X-Request-ID": "caller-supplied-id"})

    assert response.headers["X-Request-ID"] == "caller-supplied-id"


async def test_different_requests_get_different_generated_ids() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.get("/health")
        second = await client.get("/health")

    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]
