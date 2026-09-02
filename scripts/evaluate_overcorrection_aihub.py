#!/usr/bin/env python3
"""Evaluate one over-correction checkpoint on the AI Hub handwriting set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_overcorrection import (
    ROOT,
    atomic_write_text,
    evaluate_partition,
    load_model,
    load_run,
    read_jsonl,
    resolve_path,
    summarize_normal,
)

DEFAULT_MANIFEST = ROOT / "data" / "benchmark" / "eval_handwriting_2000.jsonl"
EXPECTED_SAMPLES = 2_000


def validate_samples(samples: list[dict[str, Any]], manifest: Path) -> None:
    if len(samples) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_SAMPLES:,} samples in {manifest}, found {len(samples):,}"
        )
    identifiers = [str(sample.get("id") or "") for sample in samples]
    if any(not identifier for identifier in identifiers):
        raise RuntimeError(f"Empty sample identifier in {manifest}")
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError(f"Duplicate sample identifiers in {manifest}")
    for sample in samples:
        if not str(sample.get("ground_truth") or "").strip():
            raise RuntimeError(f"Empty ground truth for {sample['id']}")
        if not Path(sample["image_path"]).is_file():
            raise FileNotFoundError(sample["image_path"])


def build_metrics(
    records: list[dict[str, Any]],
    manifest: Path,
    checkpoint: Path,
    instruction: str,
) -> dict[str, Any]:
    summary = summarize_normal(records)
    total_inference_time = sum(float(record["inference_time"]) for record in records)
    return {
        "dataset": "aihub_real",
        "manifest": str(manifest),
        "checkpoint": str(checkpoint),
        "instruction": instruction,
        "scoring": (
            "Unicode NFC, whitespace-collapsed, punctuation-insensitive "
            "character error rate"
        ),
        "sample_count": summary["sample_count"],
        "mean_cer": summary["mean_cer"],
        "corpus_cer": summary["corpus_cer"],
        "character_accuracy": max(0.0, 1.0 - summary["corpus_cer"]),
        "exact_match_count": summary["normal_preservation_count"],
        "exact_match_rate": summary["normal_preservation_rate"],
        "total_distance": summary["total_distance"],
        "total_characters": summary["total_characters"],
        "mean_inference_time": total_inference_time / max(1, len(records)),
        "total_inference_time": total_inference_time,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=EXPECTED_SAMPLES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    manifest = resolve_path(args.manifest)
    config, checkpoint = load_run(config_path, args.checkpoint)
    run_name = config_path.stem.removeprefix("overcorrection_")
    results_dir = (
        args.results_dir.expanduser().resolve()
        if args.results_dir
        else ROOT / "results" / "overcorrection" / f"{run_name}_aihub"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    samples = read_jsonl(manifest)
    validate_samples(samples, manifest)
    if not 1 <= args.limit <= EXPECTED_SAMPLES:
        raise SystemExit(f"--limit must be in [1, {EXPECTED_SAMPLES}]")
    samples = samples[: args.limit]

    model, processor, device = load_model(config, checkpoint)
    instruction = str(config["instruction"])
    records = evaluate_partition(
        "normal",
        samples,
        results_dir / "predictions.jsonl",
        model,
        processor,
        device,
        instruction,
        args.max_new_tokens,
    )
    metrics = build_metrics(records, manifest, checkpoint, instruction)
    atomic_write_text(
        results_dir / "metrics.json",
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
