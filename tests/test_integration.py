import asyncio


async def test_embedding_response_includes_provenance(service) -> None:
    client, _ = service
    response = await client.post(
        "/v1/embeddings", json={"request_id": "r-1", "sequences": ["MKT", "AAAA"]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "r-1"
    assert body["dimensions"] == 3
    assert body["batch"]["pooling"] == "masked_mean_excluding_special_tokens"
    assert body["provenance"]["checkpoint_sha256"] == "a" * 64
    assert body["provenance"]["checkpoint_files"][0]["sha256"] == "b" * 64
    assert "sequence" not in body["embeddings"][0]


async def test_concurrent_requests_share_dynamic_batch(service) -> None:
    client, backend = service
    responses = await asyncio.gather(
        client.post("/v1/embeddings", json={"sequences": ["AAAA"]}),
        client.post("/v1/embeddings", json={"sequences": ["MKT"]}),
    )
    assert all(response.status_code == 200 for response in responses)
    assert len(backend.calls) == 1
    assert responses[0].json()["batch"]["batch_id"] == responses[1].json()["batch"]["batch_id"]
    assert responses[0].json()["batch"]["batch_size"] == 2


async def test_rejects_invalid_or_oversized_sequences(service) -> None:
    client, _ = service
    invalid = await client.post("/v1/embeddings", json={"sequences": ["MKT?"]})
    oversized = await client.post("/v1/embeddings", json={"sequences": ["A" * 65]})
    assert invalid.status_code == 422
    assert oversized.status_code == 422


async def test_health_and_readiness(service) -> None:
    client, _ = service
    assert (await client.get("/healthz")).json() == {"status": "ok"}
    ready = await client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
