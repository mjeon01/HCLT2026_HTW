#!/usr/bin/env python3
"""Evaluate one model on the full synthetic error-handwriting holdout set."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from run_error_smoke_test import (
    MODEL_SPECS,
    PROMPT,
    canonical_text,
    infer_model_batch,
    levenshtein,
    load_inference_backend,
    score_prediction,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "data" / "benchmark" / "eval_synthetic_error_2000.jsonl"
RESULTS_DIR = ROOT / "results" / "error_handwriting_eval"
TARGET_SAMPLE_COUNT = 1_825


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def load_samples() -> list[dict[str, Any]]:
    samples = read_jsonl(SOURCE_MANIFEST)
    if len(samples) != TARGET_SAMPLE_COUNT:
        raise RuntimeError(
            f"Expected {TARGET_SAMPLE_COUNT} samples in {SOURCE_MANIFEST}, "
            f"found {len(samples)}"
        )
    ids = [str(sample["id"]) for sample in samples]
    images = [str(sample["image_path"]) for sample in samples]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate IDs in {SOURCE_MANIFEST}")
    if len(images) != len(set(images)):
        raise RuntimeError(f"Duplicate image paths in {SOURCE_MANIFEST}")
    missing = [image_path for image_path in images if not Path(image_path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} evaluation images; first: {missing[0]}"
        )
    unchanged = [
        sample["id"]
        for sample in samples
        if canonical_text(sample["ground_truth"])
        == canonical_text(sample["corrected_text"])
    ]
    if unchanged:
        raise RuntimeError(
            "Every error sample must differ from corrected_text; "
            f"first invalid ID: {unchanged[0]}"
        )
    return samples


def synchronize_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def result_record(
    sample: dict[str, Any],
    model_name: str,
    raw_prediction: str,
    inference_time: float,
) -> dict[str, Any]:
    scored = score_prediction(sample, raw_prediction)
    return {
        "id": sample["id"],
        "image_path": sample["image_path"],
        "ground_truth": sample["ground_truth"],
        "corrected_text": sample["corrected_text"],
        "error_type": sample["error_type"],
        "model_name": model_name,
        "prompt": PROMPT,
        "raw_prediction": raw_prediction,
        **scored,
        "inference_time": inference_time,
    }


def validate_existing_records(
    existing: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    model_name: str,
) -> None:
    required = {
        "id",
        "image_path",
        "ground_truth",
        "corrected_text",
        "error_type",
        "model_name",
        "prompt",
        "raw_prediction",
        "prediction",
        "exact_match",
        "error_preserved",
        "over_corrected",
        "recognition_error",
        "classification",
        "distance_to_ground_truth",
        "distance_to_corrected",
        "cer",
        "closer_to",
        "inference_time",
    }
    for index, record in enumerate(existing):
        missing = required - record.keys()
        if missing:
            raise RuntimeError(
                f"Existing record {index} is missing fields: {sorted(missing)}"
            )
        if index >= len(samples) or record["id"] != samples[index]["id"]:
            raise RuntimeError(
                "Existing result IDs are not an ordered prefix of the evaluation set"
            )
        if record["model_name"] != model_name or record["prompt"] != PROMPT:
            raise RuntimeError(
                f"Existing record {index} uses a different model or prompt"
            )


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    classifications = Counter(record["classification"] for record in records)
    closer = Counter(record["closer_to"] for record in records)
    total_distance = 0
    total_characters = 0
    by_error_type: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        ground_truth = canonical_text(record["ground_truth"])
        prediction = canonical_text(record["prediction"])
        total_distance += levenshtein(ground_truth, prediction)
        total_characters += len(ground_truth)
        for error_type in record["error_type"]:
            by_error_type[error_type]["total"] += 1
            by_error_type[error_type][record["classification"]] += 1
            by_error_type[error_type][f"closer_to_{record['closer_to']}"] += 1
    corpus_cer = total_distance / max(1, total_characters)

    def rate(count: int) -> float:
        return count / total if total else 0.0

    return {
        "sample_count": total,
        "mean_cer": (
            sum(float(record["cer"]) for record in records) / total
            if total
            else 0.0
        ),
        "corpus_cer": corpus_cer,
        "character_accuracy": max(0.0, 1.0 - corpus_cer),
        "error_preserved": classifications["error_preserved"],
        "error_preserved_rate": rate(classifications["error_preserved"]),
        "over_corrected": classifications["over_corrected"],
        "over_corrected_rate": rate(classifications["over_corrected"]),
        "recognition_error": classifications["recognition_error"],
        "recognition_error_rate": rate(classifications["recognition_error"]),
        "closer_to_ground_truth": closer["ground_truth"],
        "closer_to_ground_truth_rate": rate(closer["ground_truth"]),
        "closer_to_corrected": closer["corrected"],
        "closer_to_corrected_rate": rate(closer["corrected"]),
        "closer_tie": closer["tie"],
        "closer_tie_rate": rate(closer["tie"]),
        "mean_inference_time": (
            sum(float(record["inference_time"]) for record in records) / total
            if total
            else 0.0
        ),
        "total_inference_time": sum(
            float(record["inference_time"]) for record in records
        ),
        "by_error_type": {
            error_type: dict(counts)
            for error_type, counts in sorted(by_error_type.items())
        },
    }


def write_combined_summary(results_dir: Path = RESULTS_DIR) -> None:
    model_summaries = []
    for model_alias, spec in MODEL_SPECS.items():
        result_path = results_dir / spec.output_name
        if not result_path.is_file():
            continue
        records = read_jsonl(result_path)
        summary = summarize_records(records)
        summary.update(
            {
                "model_alias": model_alias,
                "model_name": spec.model_id,
                "result_file": str(result_path),
            }
        )
        model_summaries.append(summary)

    payload = {
        "dataset": str(SOURCE_MANIFEST),
        "target_sample_count": TARGET_SAMPLE_COUNT,
        "prompt": PROMPT,
        "classification_rule": (
            "strict full-string: prediction==ground_truth => error_preserved; "
            "prediction==corrected_text => over_corrected; otherwise recognition_error"
        ),
        "timing": (
            "Seconds per image; includes image preprocessing, generation, and "
            "decoding; excludes model loading, scoring, and file writing."
        ),
        "models": model_summaries,
    }
    atomic_write_text(
        results_dir / "summary.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )

    lines = [
        "# 합성 오류 손글씨 평가 결과",
        "",
        f"- 평가셋: `{SOURCE_MANIFEST.relative_to(ROOT)}`",
        f"- 실제 평가 표본 수: `{TARGET_SAMPLE_COUNT}`",
        "- 모든 모델에 동일한 이미지 순서와 동일한 프롬프트를 사용함.",
        "- 분류는 문장 전체가 정확히 일치하는 엄격한 기준임.",
        "- 추론 시간은 모델 로딩·채점·파일 쓰기를 제외하고 이미지 전처리, 생성, 디코딩을 포함함.",
        "",
        "## 프롬프트",
        "",
        "```text",
        PROMPT,
        "```",
        "",
        "## 결과",
        "",
        "| Model | N | Mean CER | Error preserved | Over-corrected | Recognition/other | Closer to error GT | Mean inference time |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in model_summaries:
        lines.append(
            f"| {summary['model_name']} | {summary['sample_count']} | "
            f"{summary['mean_cer']:.4f} | "
            f"{summary['error_preserved']} ({summary['error_preserved_rate']:.2%}) | "
            f"{summary['over_corrected']} ({summary['over_corrected_rate']:.2%}) | "
            f"{summary['recognition_error']} ({summary['recognition_error_rate']:.2%}) | "
            f"{summary['closer_to_ground_truth']} ({summary['closer_to_ground_truth_rate']:.2%}) | "
            f"{summary['mean_inference_time']:.3f}s |"
        )
    lines.extend(
        [
            "",
            "## 판정 기준",
            "",
            "- Error preserved: 정규화한 예측 전체가 오류 포함 `ground_truth`와 일치.",
            "- Over-corrected: 정규화한 예측 전체가 `corrected_text`와 일치.",
            "- Recognition/other: 위 두 문장 어느 쪽과도 완전히 일치하지 않음.",
            "- Closer to error GT: 편집 거리가 교정문보다 오류 포함 정답에 더 가까움. 부분적인 오류 보존을 보는 보조 지표임.",
            "- Mean CER: 오류 포함 `ground_truth`를 기준으로 계산한 표본별 CER의 산술평균.",
        ]
    )
    atomic_write_text(results_dir / "결과.md", "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS))
    parser.add_argument("--limit", type=int, default=TARGET_SAMPLE_COUNT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.summarize_only:
        write_combined_summary()
        print(f"Summary: {RESULTS_DIR / '결과.md'}", flush=True)
        return 0
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only is used")
    if not 1 <= args.limit <= TARGET_SAMPLE_COUNT:
        raise SystemExit(
            f"--limit must be between 1 and {TARGET_SAMPLE_COUNT}"
        )
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    samples = load_samples()
    spec = MODEL_SPECS[args.model]
    output_path = RESULTS_DIR / spec.output_name
    existing = read_jsonl(output_path) if output_path.is_file() else []
    validate_existing_records(existing, samples, spec.model_id)
    if len(existing) > args.limit:
        raise RuntimeError(
            f"{output_path} already has {len(existing)} records, exceeding "
            f"requested --limit {args.limit}"
        )
    pending = samples[len(existing) : args.limit]
    print(
        f"Model: {spec.model_id}; completed={len(existing)}; pending={len(pending)}",
        flush=True,
    )

    if pending:
        from tqdm.auto import tqdm

        backend = load_inference_backend(
            args.model,
            spec,
            local_files_only=args.local_files_only,
        )
        progress = tqdm(
            total=args.limit,
            initial=len(existing),
            desc=args.model,
            unit="img",
            dynamic_ncols=True,
            disable=args.no_progress,
        )
        classifications = Counter(
            record["classification"] for record in existing
        )
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            synchronize_cuda()
            started = time.perf_counter()
            predictions = infer_model_batch(
                args.model,
                backend,
                [str(sample["image_path"]) for sample in batch],
                args.max_new_tokens,
                len(batch),
            )
            synchronize_cuda()
            elapsed_per_image = (time.perf_counter() - started) / len(batch)
            if len(predictions) != len(batch):
                raise RuntimeError(
                    f"Expected {len(batch)} predictions, got {len(predictions)}"
                )
            for sample, prediction in zip(batch, predictions):
                record = result_record(
                    sample,
                    spec.model_id,
                    prediction,
                    elapsed_per_image,
                )
                append_jsonl(output_path, record)
                classifications[record["classification"]] += 1
            progress.update(len(batch))
            progress.set_postfix(
                preserved=classifications["error_preserved"],
                corrected=classifications["over_corrected"],
                other=classifications["recognition_error"],
                refresh=True,
            )
        progress.close()

    if not args.skip_summary:
        write_combined_summary()
    print(f"Results: {output_path}", flush=True)
    if not args.skip_summary:
        print(f"Summary: {RESULTS_DIR / '결과.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
