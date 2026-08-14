# Terpedia ESM-2 inference service

A production-oriented HTTP service for ESM-2 protein embeddings with:

- masked mean pooling that excludes padding and tokenizer special tokens;
- request coalescing into dynamic, residue-bounded GPU batches;
- checkpoint-file SHA-256 provenance in every inference response;
- deterministic input-sequence hashes instead of returning raw sequences;
- integration tests for the API, batching, validation, pooling, and provenance;
- a reproducible benchmark runner that never labels projections as measurements; and
- a private-by-default, scale-to-zero NVIDIA L4 deployment for Google Cloud Run.

Documentation and benchmark report: <https://terpedia.github.io/esm2-inference-service/>

## Architecture

The service loads one Hugging Face ESM-2 checkpoint per process. Incoming requests wait for a
small configurable window, then the dynamic batcher merges requests until either its sequence or
residue budget is full. ESM-2 returns per-token hidden states. `masked_mean_pool` removes padding,
BOS, and EOS before averaging residue embeddings. Results are sliced back to their original
requests.

Every response includes a deterministic manifest hash over checkpoint files, each file's SHA-256,
the requested and resolved Hugging Face revision, and PyTorch/Transformers versions. Pin
`ESM2_MODEL_REVISION` to a commit for production repeatability.

## Local development

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest
ruff check .
ESM2_DEVICE=cpu terpedia-esm2
```

The default 8M-parameter model is intentionally small for validation and its Hugging Face commit
is pinned. Set both `ESM2_MODEL_ID` and `ESM2_MODEL_REVISION` for another checkpoint.

```bash
curl -s http://localhost:8080/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"request_id":"example","sequences":["MKTIIALSYIFCLVFAD"]}'
```

## Configuration

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `ESM2_MODEL_ID` | `facebook/esm2_t6_8M_UR50D` | Hugging Face repository |
| `ESM2_MODEL_REVISION` | `c731040f…b0f5df` | Immutable Hugging Face commit |
| `ESM2_DEVICE` | `auto` | `cuda`, `mps`, `cpu`, or automatic |
| `ESM2_MAX_SEQUENCE_LENGTH` | 1022 | Maximum biological residues |
| `ESM2_MAX_BATCH_SEQUENCES` | 32 | Sequence-count batch budget |
| `ESM2_MAX_BATCH_TOKENS` | 8192 | Residue-count batch budget |
| `ESM2_BATCH_WAIT_MS` | 8 | Request coalescing window |
| `ESM2_QUEUE_CAPACITY` | 512 | Backpressure bound |
| `ESM2_NORMALIZE_EMBEDDINGS` | false | Optional L2 normalization |

## Reproducible benchmark

```bash
python benchmark/run.py --device cpu --iterations 10 \
  --output benchmark/results/latest.json
```

The JSON captures the checkpoint digest, resolved revision, hardware/software environment, batch
shape, latency distribution, and residue throughput. Local CPU and Cloud Run L4 runs must remain
separate artifacts; this repository does not invent or extrapolate cloud performance.

## Google Cloud Run GPU scale-to-zero

Cloud Run supports one NVIDIA L4 GPU per instance and can scale GPU-backed services to zero. GPU
services require instance-based billing; when `min=0`, no instance remains running while idle.
The service is private by default, uses 8 CPU/32 GiB, maximum three instances, and request
concurrency 32. Tune service concurrency together with the batch limits from measured load tests.

```bash
PROJECT_ID=your-project REGION=us-central1 ./cloudrun/deploy.sh
```

Prerequisites: authenticated `gcloud`, billing, the Cloud Run/Cloud Build/Artifact Registry APIs,
GPU quota in the selected region, and IAM roles for deployment. The script enables APIs, creates
the Artifact Registry repository if absent, builds the immutable Git-SHA-tagged image, and deploys
without public access. Do not put credentials in environment files committed to Git.

Primary platform references:

- [Cloud Run GPU support](https://cloud.google.com/run/docs/configuring/services/gpu)
- [Cloud Run GPU inference best practices](https://cloud.google.com/run/docs/configuring/services/gpu-best-practices)
- [Cloud Run minimum instances](https://cloud.google.com/run/docs/configuring/min-instances)

## API and privacy

OpenAPI is available at `/docs`. The service returns SHA-256 sequence identifiers, not input
sequences, but embeddings may still reveal biological information. Keep the Cloud Run service
authenticated, apply least-privilege IAM, configure request logging intentionally, and establish a
retention policy appropriate to the data owner.

## License

Apache-2.0.
