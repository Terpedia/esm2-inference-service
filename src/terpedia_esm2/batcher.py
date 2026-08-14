import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass

from .backend import EmbeddingBackend, InferenceResult


@dataclass
class WorkItem:
    sequences: list[str]
    future: asyncio.Future[tuple[InferenceResult, str, int]]


class DynamicBatcher:
    """Coalesce concurrent requests by count, residue budget, and a short wait window."""

    def __init__(
        self,
        backend: EmbeddingBackend,
        *,
        max_sequences: int,
        max_tokens: int,
        wait_ms: float,
        queue_capacity: int,
    ) -> None:
        self.backend = backend
        self.max_sequences = max_sequences
        self.max_tokens = max_tokens
        self.wait_seconds = wait_ms / 1000
        self.queue: asyncio.Queue[WorkItem] = asyncio.Queue(maxsize=queue_capacity)
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="esm2-dynamic-batcher")

    async def close(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def submit(self, sequences: list[str]) -> tuple[InferenceResult, str, int]:
        sequence_count = len(sequences)
        residue_count = sum(map(len, sequences))
        if sequence_count > self.max_sequences or residue_count > self.max_tokens:
            raise ValueError("request exceeds configured batch limits")
        future: asyncio.Future[tuple[InferenceResult, str, int]] = (
            asyncio.get_running_loop().create_future()
        )
        self.queue.put_nowait(WorkItem(sequences=sequences, future=future))
        return await future

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            items = [first]
            sequences = list(first.sequences)
            residues = sum(map(len, sequences))
            deadline = time.monotonic() + self.wait_seconds
            while len(sequences) < self.max_sequences and residues < self.max_tokens:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    candidate = await asyncio.wait_for(self.queue.get(), timeout)
                except TimeoutError:
                    break
                candidate_residues = sum(map(len, candidate.sequences))
                if (
                    len(sequences) + len(candidate.sequences) > self.max_sequences
                    or residues + candidate_residues > self.max_tokens
                ):
                    await self.queue.put(candidate)
                    break
                items.append(candidate)
                sequences.extend(candidate.sequences)
                residues += candidate_residues

            batch_id = str(uuid.uuid4())
            try:
                result = await self.backend.embed(sequences)
                offset = 0
                for item in items:
                    size = len(item.sequences)
                    sliced = InferenceResult(
                        embeddings=result.embeddings[offset : offset + size],
                        token_count=sum(map(len, item.sequences)),
                        device=result.device,
                        dtype=result.dtype,
                    )
                    item.future.set_result((sliced, batch_id, len(sequences)))
                    offset += size
            except Exception as exc:
                for item in items:
                    if not item.future.done():
                        item.future.set_exception(exc)
            finally:
                for _ in items:
                    self.queue.task_done()
