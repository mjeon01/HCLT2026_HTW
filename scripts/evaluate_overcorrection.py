#!/usr/bin/env python3
"""Evaluate one fine-tuned run on the fixed 3,825-image benchmark."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
NORMAL_MANIFEST = ROOT / "data" / "benchmark" / "eval_synthetic_normal_2000.jsonl"
ERROR_MANIFEST = ROOT / "data" / "benchmark" / "eval_synthetic_error_2000.jsonl"
EXPECTED_NORMAL = 2_000
EXPECTED_ERROR = 1_825


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


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def canonical_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFC", str(value or ""))
    normalized = normalized.replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", normalized).strip()


def punctuation_insensitive_text(value: Any) -> str:
    without_punctuation = "".join(
        character
        for character in canonical_text(value)
        if not unicodedata.category(character).startswith("P")
    )
    return re.sub(r"\s+", " ", without_punctuation).strip()


def clean_prediction(value: str) -> str:
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.DOTALL).strip()
    if value.startswith("```") and value.endswith("```"):
        value = re.sub(r"^```(?:text)?\s*|\s*```$", "", value, flags=re.IGNORECASE)
    value = re.sub(
        r"^(?:인식(?:된)?\s*(?:결과|텍스트)|출력|텍스트)\s*[:：]\s*",
        "",
        value.strip(),
    )
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'", "“", "”"}:
        value = value[1:-1]
    return canonical_text(value)


def levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_character in enumerate(left, start=1):
        current = [row]
        for column, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def aligned_correction_spans(ground_truth: str, corrected: str) -> list[dict[str, Any]]:
    matcher = difflib.SequenceMatcher(a=ground_truth, b=corrected, autojunk=False)
    spans: list[dict[str, Any]] = []
    for tag, gt_start, gt_end, corrected_start, corrected_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        variant = (
            ground_truth[:gt_start]
            + corrected[corrected_start:corrected_end]
            + ground_truth[gt_end:]
        )
        spans.append(
            {
                "operation": tag,
                "gt_start": gt_start,
                "gt_end": gt_end,
                "error": ground_truth[gt_start:gt_end],
                "correct": corrected[corrected_start:corrected_end],
                "single_correction_variant": variant,
            }
        )
    return spans


def score_error_spans(
    ground_truth: str,
    corrected: str,
    prediction: str,
) -> tuple[list[dict[str, Any]], str]:
    spans = aligned_correction_spans(ground_truth, corrected)
    if not spans:
        raise RuntimeError("Error and corrected strings have no aligned differences")
    distance_to_gt = levenshtein(ground_truth, prediction)
    outcomes = []
    for span in spans:
        variant_distance = levenshtein(span["single_correction_variant"], prediction)
        if variant_distance < distance_to_gt:
            outcome = "over_corrected"
        elif variant_distance > distance_to_gt:
            outcome = "error_preserved"
        else:
            outcome = "other_recognition_error"
        outcomes.append(
            {
                **span,
                "outcome": outcome,
                "distance_to_error_ground_truth": distance_to_gt,
                "distance_to_single_correction": variant_distance,
            }
        )
    if prediction == ground_truth:
        sample_classification = "error_preserved"
    elif any(span["outcome"] == "over_corrected" for span in outcomes):
        sample_classification = "over_corrected"
    else:
        sample_classification = "other_recognition_error"
    return outcomes, sample_classification


def load_run(config_path: Path, checkpoint: Path | None) -> tuple[dict[str, Any], Path]:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Invalid config: {config_path}")
    if checkpoint is None:
        checkpoint = resolve_path(config["output_dir"]) / "final"
    checkpoint = checkpoint.expanduser().resolve()
    metadata_path = checkpoint / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise RuntimeError(f"Checkpoint is not complete: {checkpoint}")
    return config, checkpoint


def find_projector(model: Any) -> Any:
    matches = [
        module
        for name, module in model.named_modules()
        if name.endswith("model.visual.merger")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one projector, found {len(matches)}")
    return matches[0]


def configure_generation_tokens(model: Any, processor: Any) -> None:
    """Stop generation on both the chat-template and model EOS tokens."""
    configured_eos = model.generation_config.eos_token_id
    if configured_eos is None:
        eos_token_ids: list[int] = []
    elif isinstance(configured_eos, int):
        eos_token_ids = [configured_eos]
    else:
        eos_token_ids = list(configured_eos)
    tokenizer_eos = processor.tokenizer.eos_token_id
    if tokenizer_eos is not None:
        eos_token_ids.insert(0, int(tokenizer_eos))
    eos_token_ids = list(dict.fromkeys(eos_token_ids))
    if not eos_token_ids:
        raise RuntimeError("No EOS token is configured for generation")
    model.generation_config.eos_token_id = eos_token_ids
    tokenizer_pad = processor.tokenizer.pad_token_id
    if tokenizer_pad is not None:
        model.generation_config.pad_token_id = int(tokenizer_pad)


def load_model(config: dict[str, Any], checkpoint: Path) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    device = torch.device("cuda:0")
    model_name = str(config["model_name"])
    local_files_only = bool(config.get("local_files_only", True))
    processor_source = checkpoint / "processor"
    processor = AutoProcessor.from_pretrained(
        processor_source if processor_source.is_dir() else model_name,
        local_files_only=local_files_only,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation=str(config.get("attn_implementation", "sdpa")),
        local_files_only=local_files_only,
    )
    adapter_dir = checkpoint / "adapter"
    if adapter_dir.is_dir():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
    projector = find_projector(model)
    projector_state = torch.load(
        checkpoint / "projector.pt",
        map_location="cpu",
        weights_only=True,
    )
    projector.load_state_dict(projector_state, strict=True)
    configure_generation_tokens(model, processor)
    model.config.use_cache = True
    model.to(device)
    model.eval()
    return model, processor, device


def prompt_messages(image_path: str, instruction: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": instruction},
            ],
        }
    ]


def predict(
    model: Any,
    processor: Any,
    device: Any,
    image_path: str,
    instruction: str,
    max_new_tokens: int,
) -> tuple[str, float]:
    import torch

    inputs = processor.apply_chat_template(
        prompt_messages(image_path, instruction),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("assistant_masks", None)
    inputs = {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    prompt_length = inputs["input_ids"].shape[1]
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    raw = processor.decode(
        generated[0, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return clean_prediction(raw), elapsed


def validate_resume(
    existing: list[dict[str, Any]], samples: list[dict[str, Any]], kind: str
) -> None:
    for index, record in enumerate(existing):
        if index >= len(samples) or record["id"] != samples[index]["id"]:
            raise RuntimeError(f"Existing {kind} results are not an ordered prefix")


def evaluate_partition(
    kind: str,
    samples: list[dict[str, Any]],
    output_path: Path,
    model: Any,
    processor: Any,
    device: Any,
    instruction: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    from tqdm.auto import tqdm

    existing = read_jsonl(output_path) if output_path.is_file() else []
    validate_resume(existing, samples, kind)
    progress = tqdm(
        samples[len(existing) :],
        total=len(samples),
        initial=len(existing),
        desc=f"Evaluate {kind}",
        unit="img",
        dynamic_ncols=True,
    )
    for sample in progress:
        prediction, inference_time = predict(
            model,
            processor,
            device,
            sample["image_path"],
            instruction,
            max_new_tokens,
        )
        gt = punctuation_insensitive_text(sample["ground_truth"])
        scored_prediction = punctuation_insensitive_text(prediction)
        distance = levenshtein(gt, scored_prediction)
        record: dict[str, Any] = {
            "id": sample["id"],
            "image_path": sample["image_path"],
            "ground_truth": sample["ground_truth"],
            "prediction": prediction,
            "scored_ground_truth": gt,
            "scored_prediction": scored_prediction,
            "distance": distance,
            "ground_truth_characters": len(gt),
            "cer": distance / max(1, len(gt)),
            "exact_match": gt == scored_prediction,
            "inference_time": inference_time,
        }
        if kind == "error":
            corrected = punctuation_insensitive_text(sample["corrected_text"])
            spans, sample_classification = score_error_spans(
                gt,
                corrected,
                scored_prediction,
            )
            record.update(
                {
                    "corrected_text": sample["corrected_text"],
                    "scored_corrected_text": corrected,
                    "error_type": sample.get("error_type") or [],
                    "error_spans": spans,
                    "sample_classification": sample_classification,
                }
            )
        append_jsonl(output_path, record)
    progress.close()
    return read_jsonl(output_path)


def summarize_normal(records: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(records)
    distance = sum(int(row["distance"]) for row in records)
    characters = sum(int(row["ground_truth_characters"]) for row in records)
    exact = sum(bool(row["exact_match"]) for row in records)
    return {
        "sample_count": sample_count,
        "mean_cer": sum(float(row["cer"]) for row in records) / max(1, sample_count),
        "corpus_cer": distance / max(1, characters),
        "normal_preservation_count": exact,
        "normal_preservation_rate": exact / max(1, sample_count),
        "total_distance": distance,
        "total_characters": characters,
    }


def summarize_error(records: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(records)
    distance = sum(int(row["distance"]) for row in records)
    characters = sum(int(row["ground_truth_characters"]) for row in records)
    span_counts = Counter(
        span["outcome"] for row in records for span in row["error_spans"]
    )
    sample_counts = Counter(row["sample_classification"] for row in records)
    span_total = sum(span_counts.values())
    return {
        "sample_count": sample_count,
        "mean_cer": sum(float(row["cer"]) for row in records) / max(1, sample_count),
        "corpus_cer": distance / max(1, characters),
        "error_span_count": span_total,
        "over_corrected_span_count": span_counts["over_corrected"],
        "over_correction_rate": span_counts["over_corrected"] / max(1, span_total),
        "error_preserved_span_count": span_counts["error_preserved"],
        "error_preservation_rate": span_counts["error_preserved"] / max(1, span_total),
        "other_span_count": span_counts["other_recognition_error"],
        "other_span_rate": span_counts["other_recognition_error"] / max(1, span_total),
        "sample_classification": dict(sample_counts),
        "total_distance": distance,
        "total_characters": characters,
    }


def write_qualitative_examples(path: Path, records: list[dict[str, Any]]) -> None:
    quotas = {
        "error_preserved": 20,
        "over_corrected": 20,
        "other_recognition_error": 20,
    }
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in records:
        category = row["sample_classification"]
        if counts[category] >= quotas[category]:
            continue
        selected.append(
            {
                "category": category,
                "id": row["id"],
                "image_path": row["image_path"],
                "ground_truth": row["ground_truth"],
                "corrected_text": row["corrected_text"],
                "prediction": row["prediction"],
                "error_type": row["error_type"],
                "error_spans": row["error_spans"],
            }
        )
        counts[category] += 1
    atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--normal-limit", type=int, default=EXPECTED_NORMAL)
    parser.add_argument("--error-limit", type=int, default=EXPECTED_ERROR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config, checkpoint = load_run(config_path, args.checkpoint)
    setting = str(config["setting"])
    results_dir = (
        args.results_dir.expanduser().resolve()
        if args.results_dir
        else ROOT / "results" / "overcorrection" / setting
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    normal_samples = read_jsonl(NORMAL_MANIFEST)
    error_samples = read_jsonl(ERROR_MANIFEST)
    if len(normal_samples) != EXPECTED_NORMAL or len(error_samples) != EXPECTED_ERROR:
        raise RuntimeError("Fixed benchmark counts changed")
    if not 1 <= args.normal_limit <= EXPECTED_NORMAL:
        raise SystemExit(f"--normal-limit must be in [1, {EXPECTED_NORMAL}]")
    if not 1 <= args.error_limit <= EXPECTED_ERROR:
        raise SystemExit(f"--error-limit must be in [1, {EXPECTED_ERROR}]")
    normal_samples = normal_samples[: args.normal_limit]
    error_samples = error_samples[: args.error_limit]
    model, processor, device = load_model(config, checkpoint)
    instruction = str(config["instruction"])
    normal_records = evaluate_partition(
        "normal",
        normal_samples,
        results_dir / "normal_predictions.jsonl",
        model,
        processor,
        device,
        instruction,
        args.max_new_tokens,
    )
    error_records = evaluate_partition(
        "error",
        error_samples,
        results_dir / "error_predictions.jsonl",
        model,
        processor,
        device,
        instruction,
        args.max_new_tokens,
    )
    normal = summarize_normal(normal_records)
    error = summarize_error(error_records)
    overall_distance = normal["total_distance"] + error["total_distance"]
    overall_characters = normal["total_characters"] + error["total_characters"]
    metrics = {
        "setting": setting,
        "train_mode": config["train_mode"],
        "error_sampling_ratio": float(config["error_sampling_ratio"]),
        "checkpoint": str(checkpoint),
        "scoring": {
            "cer": "Unicode NFC, whitespace-collapsed, punctuation-insensitive character error rate",
            "normal_preservation": "punctuation-insensitive full-string exact match",
            "error_spans": "SequenceMatcher alignment between error GT and corrected text",
            "span_outcome": (
                "compare prediction distance to error GT versus a one-correction "
                "counterfactual; closer counterfactual=over-corrected, farther=preserved, tie=other"
            ),
        },
        "normal": normal,
        "error": error,
        "overall": {
            "sample_count": len(normal_records) + len(error_records),
            "corpus_cer": overall_distance / max(1, overall_characters),
            "total_distance": overall_distance,
            "total_characters": overall_characters,
        },
    }
    atomic_write_text(
        results_dir / "metrics.json",
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
    )
    write_qualitative_examples(
        results_dir / "qualitative_examples.jsonl",
        error_records,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
