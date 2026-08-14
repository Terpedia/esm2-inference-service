import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi

from .schemas import CheckpointFile, Provenance


def _is_checkpoint(path: Path) -> bool:
    return path.name.endswith(".safetensors") or (
        path.name.startswith("pytorch_model") and path.name.endswith(".bin")
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_provenance(
    *,
    model_id: str,
    requested_revision: str | None,
    snapshot_path: Path,
    transformers_version: str,
    torch_version: str,
) -> Provenance:
    candidates = [
        path for path in snapshot_path.rglob("*") if path.is_file() and _is_checkpoint(path)
    ]
    safetensors = [path for path in candidates if path.name.endswith(".safetensors")]
    files = sorted(
        safetensors or candidates, key=lambda path: path.relative_to(snapshot_path).as_posix()
    )
    if not files:
        raise RuntimeError(f"no supported checkpoint file found under {snapshot_path}")

    records = [
        CheckpointFile(
            path=path.relative_to(snapshot_path).as_posix(),
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )
        for path in files
    ]
    manifest = json.dumps(
        [record.model_dump() for record in records], sort_keys=True, separators=(",", ":")
    ).encode()
    combined = hashlib.sha256(manifest).hexdigest()
    if requested_revision and len(requested_revision) == 40:
        resolved_revision = requested_revision
    else:
        try:
            resolved_revision = (
                HfApi()
                .model_info(model_id, revision=requested_revision or "main", files_metadata=False)
                .sha
            )
        except Exception:
            resolved_revision = snapshot_path.name if len(snapshot_path.name) == 40 else None
    return Provenance(
        model_id=model_id,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        checkpoint_sha256=combined,
        checkpoint_files=records,
        transformers_version=transformers_version,
        torch_version=torch_version,
    )
