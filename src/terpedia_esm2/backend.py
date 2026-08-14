import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
import transformers
from huggingface_hub import snapshot_download
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from .config import Settings
from .pooling import masked_mean_pool
from .provenance import checkpoint_provenance
from .schemas import Provenance


@dataclass(frozen=True)
class InferenceResult:
    embeddings: list[list[float]]
    token_count: int
    device: str
    dtype: str


class EmbeddingBackend(Protocol):
    model_id: str
    dimensions: int
    provenance: Provenance

    async def embed(self, sequences: list[str]) -> InferenceResult: ...


class ESM2Backend:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_id = settings.model_id
        self.device = self._resolve_device(settings.device)
        snapshot = Path(
            snapshot_download(
                repo_id=settings.model_id,
                revision=settings.model_revision,
                allow_patterns=[
                    "config.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "vocab.txt",
                    "model.safetensors",
                    "pytorch_model.bin",
                ],
            )
        )
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(
            snapshot,
            trust_remote_code=settings.trust_remote_code,
            dtype="auto",
            add_pooling_layer=False,
        ).to(self.device)
        self.model.eval()
        self.dimensions = int(self.model.config.hidden_size)
        self.provenance = checkpoint_provenance(
            model_id=settings.model_id,
            requested_revision=settings.model_revision,
            snapshot_path=snapshot,
            transformers_version=transformers.__version__,
            torch_version=torch.__version__,
        )

    @staticmethod
    def _resolve_device(configured: str) -> torch.device:
        if configured != "auto":
            return torch.device(configured)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    async def embed(self, sequences: list[str]) -> InferenceResult:
        return await asyncio.to_thread(self._embed_sync, sequences)

    def _embed_sync(self, sequences: list[str]) -> InferenceResult:
        encoded = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.settings.max_sequence_length + 2,
            return_special_tokens_mask=True,
        )
        special_tokens_mask = encoded.pop("special_tokens_mask")
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden_states = self.model(**inputs).last_hidden_state
            pooled = masked_mean_pool(
                hidden_states,
                inputs["attention_mask"],
                special_tokens_mask.to(self.device),
            )
            if self.settings.normalize_embeddings:
                pooled = F.normalize(pooled, p=2, dim=-1)
        return InferenceResult(
            embeddings=pooled.float().cpu().tolist(),
            token_count=int(inputs["attention_mask"].sum().item()),
            device=str(self.device),
            dtype=str(hidden_states.dtype).removeprefix("torch."),
        )


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()
