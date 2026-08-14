import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from .backend import EmbeddingBackend, ESM2Backend, sequence_sha256
from .batcher import DynamicBatcher
from .config import Settings, get_settings
from .schemas import (
    BatchMetadata,
    EmbeddingItem,
    EmbeddingRequest,
    EmbeddingResponse,
    ModelMetadata,
)

BackendFactory = Callable[[Settings], EmbeddingBackend]


def create_app(
    settings: Settings | None = None,
    backend_factory: BackendFactory = ESM2Backend,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        backend = backend_factory(settings)
        batcher = DynamicBatcher(
            backend,
            max_sequences=settings.max_batch_sequences,
            max_tokens=settings.max_batch_tokens,
            wait_ms=settings.batch_wait_ms,
            queue_capacity=settings.queue_capacity,
        )
        await batcher.start()
        app.state.backend = backend
        app.state.batcher = batcher
        yield
        await batcher.close()

    app = FastAPI(
        title="Terpedia ESM-2 Inference",
        version="0.1.0",
        description="Batched protein embeddings with checkpoint provenance.",
        lifespan=lifespan,
    )

    @app.exception_handler(asyncio.QueueFull)
    async def queue_full_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=429, content={"detail": "inference queue is full"})

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", response_model=ModelMetadata)
    async def readyz(request: Request) -> ModelMetadata:
        backend: EmbeddingBackend = request.app.state.backend
        return ModelMetadata(
            status="ready",
            model=backend.model_id,
            dimensions=backend.dimensions,
            device=getattr(backend, "device", "unknown").__str__(),
            dtype=str(next(backend.model.parameters()).dtype).removeprefix("torch.")
            if hasattr(backend, "model")
            else "float32",
            provenance=backend.provenance,
        )

    @app.post("/v1/embeddings", response_model=EmbeddingResponse)
    async def embeddings(payload: EmbeddingRequest, request: Request) -> EmbeddingResponse:
        if any(len(sequence) > settings.max_sequence_length for sequence in payload.sequences):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"sequence exceeds maximum length {settings.max_sequence_length}",
            )
        try:
            result, batch_id, shared_batch_size = await request.app.state.batcher.submit(
                payload.sequences
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        backend: EmbeddingBackend = request.app.state.backend
        return EmbeddingResponse(
            request_id=payload.request_id,
            model=backend.model_id,
            dimensions=backend.dimensions,
            embeddings=[
                EmbeddingItem(
                    sequence_sha256=sequence_sha256(sequence),
                    sequence_length=len(sequence),
                    embedding=embedding,
                )
                for sequence, embedding in zip(payload.sequences, result.embeddings, strict=True)
            ],
            provenance=backend.provenance,
            batch=BatchMetadata(
                batch_id=batch_id,
                batch_size=shared_batch_size,
                token_count=result.token_count,
                device=result.device,
                dtype=result.dtype,
            ),
        )

    return app


app = create_app()
