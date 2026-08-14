#!/usr/bin/env python3
"""Reproducible end-to-end local benchmark; emits machine-readable JSON."""

import argparse
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from terpedia_esm2.backend import ESM2Backend
from terpedia_esm2.config import Settings

SEQUENCES = [
    "MKTIIALSYIFCLVFADYKDDDDK",
    "MALWMRLLPLLALLALWGPGPGAG",
    "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHG",
    "MSDSEVNQEAKPEVKPEVKPETHINLVEKLKAEGVEVVEFDVVEGDTDVVVDFSATWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANLA",
]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/latest.json"))
    args = parser.parse_args()

    backend = ESM2Backend(Settings(model_id=args.model, device=args.device))
    for _ in range(args.warmup):
        await backend.embed(SEQUENCES)
    timings = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        await backend.embed(SEQUENCES)
        timings.append(time.perf_counter() - started)
    residues = sum(map(len, SEQUENCES))
    result = {
        "schema_version": 1,
        "measured_at": datetime.now(UTC).isoformat(),
        "scope": "local_measured",
        "model": args.model,
        "checkpoint_sha256": backend.provenance.checkpoint_sha256,
        "resolved_revision": backend.provenance.resolved_revision,
        "device": str(backend.device),
        "hardware": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "batch_sequences": len(SEQUENCES),
        "batch_residues": residues,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "latency_ms": {
            "median": statistics.median(timings) * 1000,
            "mean": statistics.mean(timings) * 1000,
            "min": min(timings) * 1000,
            "max": max(timings) * 1000,
        },
        "throughput_residues_per_second": residues / statistics.mean(timings),
        "note": "Measured locally. This is not a Cloud Run L4 result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
