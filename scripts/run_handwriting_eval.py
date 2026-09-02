#!/usr/bin/env python3
"""Evaluate one vision-language model on the 2,000-image handwriting set."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from run_error_smoke_test import (
    MODEL_SPECS,
    PIPELINE_MODEL_ALIASES,
    MiniCPMAdapter,
    MinistralAdapter,
    InternVLAdapter,
    canonical_text,
    clean_prediction,
    extract_generated_text,
    levenshtein,
    load_inference_backend,
    load_internvl_image,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "data" / "benchmark" / "eval_handwriting_2000.jsonl"
RESULTS_DIR = ROOT / "results" / "handwriting_eval"
TARGET_SAMPLE_COUNT = 2_000
PROMPT = """손글씨 이미지에 작성된 내용을 그대로 텍스트로 인식하세요.
이미지에 보이는 내용만 출력하세요."""


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


def load_samples(
    source_manifest: Path,
    expected_count: int,
) -> list[dict[str, Any]]:
    samples = read_jsonl(source_manifest)
    if len(samples) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} samples in {source_manifest}, "
            f"found {len(samples)}"
        )
    ids = [str(sample["id"]) for sample in samples]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate IDs in {source_manifest}")
    missing = [
        str(sample["image_path"])
        for sample in samples
        if not Path(sample["image_path"]).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} evaluation images; first: {missing[0]}"
        )
    return samples


def messages_for_image(image_path: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]


def infer_pipeline_batch(
    pipe: Any,
    image_paths: list[str],
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    outputs = pipe(
        text=[messages_for_image(image_path) for image_path in image_paths],
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
        enable_thinking=False,
    )
    if not isinstance(outputs, list) or len(outputs) != len(image_paths):
        length = len(outputs) if isinstance(outputs, list) else "unknown"
        raise RuntimeError(
            f"Expected {len(image_paths)} pipeline outputs, received "
            f"{type(outputs).__name__} with length {length}"
        )
    return [extract_generated_text(item) for item in outputs]


def infer_internvl_batch(
    adapter: InternVLAdapter,
    image_paths: list[str],
    max_new_tokens: int,
) -> list[str]:
    import torch

    generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}
    predictions = []
    with torch.inference_mode():
        for image_path in image_paths:
            pixel_values = load_internvl_image(image_path).to(
                device=adapter.device,
                dtype=torch.bfloat16,
            )
            response = adapter.model.chat(
                adapter.tokenizer,
                pixel_values,
                f"<image>\n{PROMPT}",
                generation_config,
            )
            predictions.append(str(response))
    return predictions


def infer_minicpm_batch(
    adapter: MiniCPMAdapter,
    image_paths: list[str],
    max_new_tokens: int,
) -> list[str]:
    import torch

    downsample_mode = "4x"
    predictions = []
    with torch.inference_mode():
        for image_path in image_paths:
            inputs = adapter.processor.apply_chat_template(
                messages_for_image(image_path),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                downsample_mode=downsample_mode,
                max_slice_nums=36,
            ).to(adapter.model.device)
            generated = adapter.model.generate(
                **inputs,
                downsample_mode=downsample_mode,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            prompt_length = inputs.input_ids.shape[1]
            predictions.append(
                adapter.processor.decode(
                    generated[0, prompt_length:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
    return predictions


def infer_ministral_batch(
    adapter: MinistralAdapter,
    image_paths: list[str],
    max_new_tokens: int,
) -> list[str]:
    import torch

    predictions = []
    with torch.inference_mode():
        for image_path in image_paths:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "path": image_path},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ]
            inputs = adapter.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
            )
            for key, value in inputs.items():
                if not isinstance(value, torch.Tensor):
                    continue
                inputs[key] = value.to(
                    device=adapter.device,
                    dtype=torch.bfloat16 if key == "pixel_values" else value.dtype,
                )
            prompt_length = inputs["input_ids"].shape[1]
            generated = adapter.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )[0]
            predictions.append(
                adapter.tokenizer.decode(
                    generated[prompt_length:],
                    skip_special_tokens=True,
                )
            )
    return predictions


def infer_model_batch(
    model_alias: str,
    backend: Any,
    image_paths: list[str],
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    if model_alias in PIPELINE_MODEL_ALIASES:
        return infer_pipeline_batch(
            backend,
            image_paths,
            max_new_tokens,
            batch_size,
        )
    if model_alias == "internvl3-8b":
        return infer_internvl_batch(backend, image_paths, max_new_tokens)
    if model_alias == "minicpm-v46":
        return infer_minicpm_batch(backend, image_paths, max_new_tokens)
    if model_alias == "ministral3-8b":
        return infer_ministral_batch(backend, image_paths, max_new_tokens)
    raise ValueError(f"Unsupported model alias: {model_alias}")


def synchronize_cuda() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def score_prediction(
    sample: dict[str, Any],
    raw_prediction: str,
    inference_time: float,
) -> dict[str, Any]:
    ground_truth = canonical_text(str(sample["ground_truth"]))
    prediction = clean_prediction(raw_prediction)
    distance = levenshtein(ground_truth, prediction)
    return {
        "id": sample["id"],
        "image_path": sample["image_path"],
        "ground_truth": sample["ground_truth"],
        "prediction": prediction,
        "cer": distance / max(1, len(ground_truth)),
        "exact_match": prediction == ground_truth,
        "inference_time": inference_time,
    }


def validate_existing_records(
    existing: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> None:
    required = {
        "id",
        "image_path",
        "ground_truth",
        "prediction",
        "cer",
        "exact_match",
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


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(records)
    exact_match_count = sum(bool(record["exact_match"]) for record in records)
    total_distance = 0
    total_characters = 0
    for record in records:
        ground_truth = canonical_text(str(record["ground_truth"]))
        prediction = canonical_text(str(record["prediction"]))
        total_distance += levenshtein(ground_truth, prediction)
        total_characters += len(ground_truth)
    corpus_cer = total_distance / max(1, total_characters)
    return {
        "sample_count": sample_count,
        "mean_cer": (
            sum(float(record["cer"]) for record in records) / sample_count
            if sample_count
            else 0.0
        ),
        "corpus_cer": corpus_cer,
        "character_accuracy": max(0.0, 1.0 - corpus_cer),
        "exact_match_count": exact_match_count,
        "exact_match_rate": (
            exact_match_count / sample_count if sample_count else 0.0
        ),
        "mean_inference_time": (
            sum(float(record["inference_time"]) for record in records) / sample_count
            if sample_count
            else 0.0
        ),
        "total_inference_time": sum(
            float(record["inference_time"]) for record in records
        ),
    }


def write_combined_summary(
    results_dir: Path,
    source_manifest: Path,
    target_sample_count: int,
    report_title: str,
) -> None:
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
        "dataset": str(source_manifest),
        "target_sample_count": target_sample_count,
        "prompt": PROMPT,
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
        f"# {report_title}",
        "",
        f"- 평가셋: `{source_manifest.relative_to(ROOT)}`",
        f"- 목표 표본 수: `{target_sample_count}`",
        "- 모든 모델에 동일한 이미지 순서와 동일한 프롬프트를 사용함.",
        "- 추론 시간은 모델 로딩·채점·파일 쓰기를 제외하고 이미지 전처리, 생성, 디코딩을 포함한 초 단위 시간임.",
        "",
        "## 프롬프트",
        "",
        "```text",
        PROMPT,
        "```",
        "",
        "## 결과",
        "",
        "| Model | N | Mean CER | Corpus CER | Character Accuracy | Exact Match | Mean inference time |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in model_summaries:
        lines.append(
            f"| {summary['model_name']} | {summary['sample_count']} | "
            f"{summary['mean_cer']:.4f} | {summary['corpus_cer']:.4f} | "
            f"{summary['character_accuracy']:.2%} | "
            f"{summary['exact_match_count']} ({summary['exact_match_rate']:.2%}) | "
            f"{summary['mean_inference_time']:.3f}s |"
        )
    lines.extend(
        [
            "",
            "## 지표 정의",
            "",
            "- Mean CER: 각 표본 CER의 산술평균.",
            "- Corpus CER: 전체 편집 거리 합을 전체 정답 문자 수로 나눈 값.",
            "- Character Accuracy: `max(0, 1 - Corpus CER)`.",
            "- Exact Match: NFC 정규화와 연속 공백 정리 후 문장 전체가 같은 경우.",
        ]
    )
    atomic_write_text(results_dir / "결과.md", "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS))
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=SOURCE_MANIFEST,
        help=f"Evaluation JSONL (default: {SOURCE_MANIFEST})",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Output directory (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--report-title",
        default="AI Hub 실제 손글씨 평가 결과",
    )
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
    source_manifest = args.source_manifest.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    if args.summarize_only:
        write_combined_summary(
            results_dir,
            source_manifest,
            args.limit,
            args.report_title,
        )
        print(f"Summary: {results_dir / '결과.md'}", flush=True)
        return 0
    if not args.model:
        raise SystemExit("--model is required unless --summarize-only is used")
    if not 1 <= args.limit <= TARGET_SAMPLE_COUNT:
        raise SystemExit(
            f"--limit must be between 1 and {TARGET_SAMPLE_COUNT}"
        )
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    samples = load_samples(source_manifest, TARGET_SAMPLE_COUNT)
    spec = MODEL_SPECS[args.model]
    output_path = results_dir / spec.output_name
    existing = read_jsonl(output_path) if output_path.is_file() else []
    validate_existing_records(existing, samples)
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
        exact_count = sum(bool(record["exact_match"]) for record in existing)
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
                record = score_prediction(sample, prediction, elapsed_per_image)
                append_jsonl(output_path, record)
                exact_count += int(record["exact_match"])
            progress.update(len(batch))
            progress.set_postfix(exact=exact_count, refresh=True)
        progress.close()

    if not args.skip_summary:
        write_combined_summary(
            results_dir,
            source_manifest,
            args.limit,
            args.report_title,
        )
    print(f"Results: {output_path}", flush=True)
    if not args.skip_summary:
        print(f"Summary: {results_dir / '결과.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
