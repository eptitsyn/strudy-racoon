import asyncio

from httpx import AsyncClient


class TestDetect:
    async def test_basic(self, client: AsyncClient) -> None:
        r = await client.post("/v1/detect", json={"text": "hello world"})
        assert r.status_code == 200
        body = r.json()
        assert body["model"]["name"] == "fast"
        assert 0.0 <= body["ai_probability"] <= 1.0
        assert 0.0 <= body["human_probability"] <= 1.0
        assert body["verdict"] in {"ai", "human", "unknown"}
        assert "request_id" in body

    async def test_request_id_propagated(self, client: AsyncClient) -> None:
        rid = "11111111-1111-1111-1111-111111111111"
        r = await client.post(
            "/v1/detect", json={"text": "x"}, headers={"X-Request-ID": rid}
        )
        assert r.status_code == 200
        assert r.json()["request_id"] == rid
        assert r.headers["X-Request-ID"] == rid

    async def test_empty_text_rejected(self, client: AsyncClient) -> None:
        r = await client.post("/v1/detect", json={"text": ""})
        assert r.status_code == 422

    async def test_oversize_rejected(self, client: AsyncClient) -> None:
        r = await client.post("/v1/detect", json={"text": "x" * 50_000})
        assert r.status_code == 422

    async def test_model_override_must_match_active(self, client: AsyncClient) -> None:
        r = await client.post(
            "/v1/detect", json={"text": "x", "model": "human-leaning"}
        )
        assert r.status_code == 409
        assert r.json()["code"] == "conflict"


class TestConcurrency:
    async def test_parallel_requests_succeed(self, client: AsyncClient) -> None:
        async def one(i: int) -> int:
            r = await client.post("/v1/detect", json={"text": f"sample-{i}"})
            return r.status_code

        results = await asyncio.gather(*[one(i) for i in range(20)])
        assert all(s == 200 for s in results)

    async def test_switch_during_traffic_no_500(self, client: AsyncClient) -> None:
        async def detect_burst() -> list[int]:
            results: list[int] = []
            for i in range(15):
                r = await client.post("/v1/detect", json={"text": f"t-{i}"})
                results.append(r.status_code)
                await asyncio.sleep(0.005)
            return results

        async def switcher() -> None:
            await asyncio.sleep(0.01)
            await client.post("/v1/models/switch", json={"name": "human-leaning"})
            await asyncio.sleep(0.02)
            await client.post("/v1/models/switch", json={"name": "fast"})

        codes, _ = await asyncio.gather(detect_burst(), switcher())
        assert all(c == 200 for c in codes)
