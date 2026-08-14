import asyncio
import hashlib

import pytest
from httpx import ASGITransport, AsyncClient

from terpedia_esm2.app import create_app
from terpedia_esm2.backend import InferenceResult
from terpedia_esm2.config import Settings
from terpedia_esm2.schemas import Provenance


class FakeBackend:
    model_id = "terpedia/fake-esm2"
    dimensions = 3
    device = "cpu"
    provenance = Provenance(
        model_id=model_id,
        requested_revision="test",
        resolved_revision="0123456789abcdef0123456789abcdef01234567",
        checkpoint_sha256="a" * 64,
        checkpoint_files=[{"path": "model.safetensors", "sha256": "b" * 64, "size_bytes": 42}],
        transformers_version="test",
        torch_version="test",
    )

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, sequences: list[str]) -> InferenceResult:
        self.calls.append(sequences)
        await asyncio.sleep(0.002)
        embeddings = []
        for sequence in sequences:
            sequence_code = int(hashlib.sha1(sequence.encode()).hexdigest()[:4], 16)
            embeddings.append(
                [float(len(sequence)), float(sequence.count("A")), float(sequence_code)]
            )
        return InferenceResult(embeddings, sum(map(len, sequences)), "cpu", "float32")


@pytest.fixture
async def service():
    backend = FakeBackend()
    settings = Settings(
        model_id=backend.model_id,
        max_sequence_length=64,
        max_batch_sequences=8,
        max_batch_tokens=256,
        batch_wait_ms=20,
    )
    app = create_app(settings, lambda _: backend)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client, backend
