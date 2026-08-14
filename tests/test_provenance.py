from terpedia_esm2.provenance import checkpoint_provenance, sha256_file


def test_provenance_prefers_safetensors_and_hashes_exact_bytes(tmp_path) -> None:
    safetensor = tmp_path / "model.safetensors"
    legacy = tmp_path / "pytorch_model.bin"
    safetensor.write_bytes(b"canonical checkpoint")
    legacy.write_bytes(b"unused legacy checkpoint")

    provenance = checkpoint_provenance(
        model_id="terpedia/test",
        requested_revision="0" * 40,
        snapshot_path=tmp_path,
        transformers_version="test",
        torch_version="test",
    )

    assert provenance.resolved_revision == "0" * 40
    assert len(provenance.checkpoint_files) == 1
    assert provenance.checkpoint_files[0].path == "model.safetensors"
    assert provenance.checkpoint_files[0].sha256 == sha256_file(safetensor)
    assert len(provenance.checkpoint_sha256) == 64
