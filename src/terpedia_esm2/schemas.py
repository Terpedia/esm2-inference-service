from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

ProteinSequence = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequences: list[ProteinSequence] = Field(min_length=1, max_length=256)
    request_id: str | None = Field(default=None, max_length=128)

    @field_validator("sequences")
    @classmethod
    def normalize_sequences(cls, sequences: list[str]) -> list[str]:
        allowed = set("ABCDEFGHIKLMNOPQRSTUVWYZX*-")
        normalized: list[str] = []
        for sequence in sequences:
            cleaned = "".join(sequence.split()).upper()
            invalid = sorted(set(cleaned) - allowed)
            if invalid:
                raise ValueError(f"unsupported amino-acid symbols: {''.join(invalid)}")
            normalized.append(cleaned)
        return normalized


class CheckpointFile(BaseModel):
    path: str
    sha256: str
    size_bytes: int


class Provenance(BaseModel):
    model_id: str
    requested_revision: str | None
    resolved_revision: str | None
    checkpoint_sha256: str
    checkpoint_files: list[CheckpointFile]
    transformers_version: str
    torch_version: str


class EmbeddingItem(BaseModel):
    sequence_sha256: str
    sequence_length: int
    embedding: list[float]


class BatchMetadata(BaseModel):
    batch_id: str
    batch_size: int
    token_count: int
    device: str
    dtype: str
    pooling: str = "masked_mean_excluding_special_tokens"


class EmbeddingResponse(BaseModel):
    request_id: str | None
    model: str
    dimensions: int
    embeddings: list[EmbeddingItem]
    provenance: Provenance
    batch: BatchMetadata


class ModelMetadata(BaseModel):
    status: str
    model: str
    dimensions: int
    device: str
    dtype: str
    provenance: Provenance
