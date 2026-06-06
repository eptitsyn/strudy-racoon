from httpx import AsyncClient


class TestModelsList:
    async def test_lists_all(self, client: AsyncClient) -> None:
        r = await client.get("/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert body["active"] == "fast"
        names = {m["name"] for m in body["available"]}
        assert names == {"fast", "slow", "human-leaning"}
        loaded = {m["name"] for m in body["available"] if m["loaded"]}
        assert loaded == {"fast"}


class TestSwitch:
    async def test_switch_to_existing(self, client: AsyncClient) -> None:
        r = await client.post("/v1/models/switch", json={"name": "human-leaning"})
        assert r.status_code == 200
        assert r.json()["active"]["name"] == "human-leaning"

        r2 = await client.get("/v1/models")
        assert r2.json()["active"] == "human-leaning"

    async def test_switch_unknown(self, client: AsyncClient) -> None:
        r = await client.post("/v1/models/switch", json={"name": "ghost"})
        assert r.status_code == 404
        body = r.json()
        assert body["code"] == "model_not_found"

    async def test_switch_idempotent(self, client: AsyncClient) -> None:
        r = await client.post("/v1/models/switch", json={"name": "fast"})
        assert r.status_code == 200
        assert r.json()["active"]["name"] == "fast"

    async def test_switch_validation(self, client: AsyncClient) -> None:
        r = await client.post("/v1/models/switch", json={"name": ""})
        assert r.status_code == 422
        assert r.json()["code"] == "validation_error"
