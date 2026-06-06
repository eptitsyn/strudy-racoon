from httpx import AsyncClient


class TestHealth:
    async def test_live(self, client: AsyncClient) -> None:
        r = await client.get("/health/live")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    async def test_ready(self, client: AsyncClient) -> None:
        r = await client.get("/health/ready")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ready"
        assert body["active_model"] == "fast"

    async def test_request_id_echoed(self, client: AsyncClient) -> None:
        r = await client.get("/health/live", headers={"X-Request-ID": "not-a-uuid"})
        # Invalid → server-generated uuid; just check it's present.
        assert "X-Request-ID" in r.headers
