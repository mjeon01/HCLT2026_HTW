#!/usr/bin/env python3
"""Run the fixed 100-sample handwritten-error preservation smoke test."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "data/benchmark/eval_synthetic_error_2000.jsonl"
RESULTS_DIR = ROOT / "results/smoke_test"
SAMPLE_MANIFEST = RESULTS_DIR / "smoke_samples_100.jsonl"
SEED = 42
SAMPLE_COUNT = 100
PROMPT = """손글씨 이미지에 작성된 내용을 그대로 텍스트로 인식하세요.
맞춤법, 띄어쓰기, 조사, 어미, 문법 오류를 수정하거나
자연스러운 표현으로 바꾸지 마세요.
이미지에 보이는 내용을 그대로 출력하세요."""


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    output_name: str
    trust_remote_code: bool = False


@dataclass(frozen=True)
class InternVLAdapter:
    model: Any
    tokenizer: Any
    device: Any


@dataclass(frozen=True)
class MiniCPMAdapter:
    model: Any
    processor: Any


@dataclass(frozen=True)
class MinistralAdapter:
    model: Any
    tokenizer: Any
    device: Any


MODEL_SPECS = {
    "qwen35-9b": ModelSpec("Qwen/Qwen3.5-9B", "qwen35_9b_results.jsonl"),
    "qwen35-4b": ModelSpec("Qwen/Qwen3.5-4B", "qwen35_4b_results.jsonl"),
    "qwen3-vl-4b": ModelSpec(
        "Qwen/Qwen3-VL-4B-Instruct", "qwen3_vl_4b_results.jsonl"
    ),
    "internvl3-8b": ModelSpec(
        "OpenGVLab/InternVL3-8B",
        "internvl3_8b_results.jsonl",
        trust_remote_code=True,
    ),
    "ministral3-8b": ModelSpec(
        "mistralai/Ministral-3-8B-Instruct-2512",
        "ministral3_8b_results.jsonl",
    ),
    "minicpm-v46": ModelSpec(
        "openbmb/MiniCPM-V-4.6",
        "minicpm_v46_results.jsonl",
        trust_remote_code=True,
    ),
}

PIPELINE_MODEL_ALIASES = frozenset({"qwen35-9b", "qwen35-4b", "qwen3-vl-4b"})
MINICPM_MIN_TRANSFORMERS = "5.7.0"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    )


def prepare_samples() -> list[dict[str, Any]]:
    source = read_jsonl(SOURCE_MANIFEST)
    if len(source) < SAMPLE_COUNT:
        raise RuntimeError(f"Need {SAMPLE_COUNT} records, found {len(source)}")
    expected = random.Random(SEED).sample(source, SAMPLE_COUNT)
    missing = [item["image_path"] for item in expected if not Path(item["image_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} sampled images; first: {missing[0]}")

    if SAMPLE_MANIFEST.exists():
        existing = read_jsonl(SAMPLE_MANIFEST)
        if [item["id"] for item in existing] != [item["id"] for item in expected]:
            raise RuntimeError(
                f"Existing {SAMPLE_MANIFEST} does not match seed={SEED}; refusing to replace it"
            )
        return existing

    write_jsonl(SAMPLE_MANIFEST, expected)
    return expected


def canonical_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", value).strip()


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
    for row, left_char in enumerate(left, start=1):
        current = [row]
        for column, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def score_prediction(sample: dict[str, Any], prediction: str) -> dict[str, Any]:
    ground_truth = canonical_text(sample["ground_truth"])
    corrected = canonical_text(sample["corrected_text"])
    prediction = clean_prediction(prediction)
    exact_match = prediction == ground_truth
    corrected_match = prediction == corrected
    if exact_match:
        classification = "error_preserved"
    elif corrected_match:
        classification = "over_corrected"
    else:
        classification = "recognition_error"
    distance_to_gt = levenshtein(ground_truth, prediction)
    distance_to_corrected = levenshtein(corrected, prediction)
    return {
        "prediction": prediction,
        "exact_match": exact_match,
        "error_preserved": exact_match,
        "over_corrected": corrected_match,
        "recognition_error": classification == "recognition_error",
        "classification": classification,
        "distance_to_ground_truth": distance_to_gt,
        "distance_to_corrected": distance_to_corrected,
        "cer": distance_to_gt / max(1, len(ground_truth)),
        "closer_to": (
            "ground_truth"
            if distance_to_gt < distance_to_corrected
            else "corrected"
            if distance_to_corrected < distance_to_gt
            else "tie"
        ),
    }


def extract_generated_text(output: Any) -> str:
    if isinstance(output, list) and output:
        output = output[0]
    if isinstance(output, dict):
        output = output.get("generated_text", output.get("text", output))
    if isinstance(output, list):
        for message in reversed(output):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content", "")
                if isinstance(content, list):
                    return "".join(
                        str(part.get("text", "")) if isinstance(part, dict) else str(part)
                        for part in content
                    )
                return str(content)
    if not isinstance(output, str):
        raise TypeError(f"Unsupported pipeline output: {type(output).__name__}: {output!r}")
    return output


def load_pipeline(spec: ModelSpec, *, local_files_only: bool) -> Any:
    import torch
    from transformers import pipeline

    torch.manual_seed(SEED)
    return pipeline(
        "image-text-to-text",
        model=spec.model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=spec.trust_remote_code,
        local_files_only=local_files_only,
    )


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


def infer_batch(
    pipe: Any,
    image_paths: list[str],
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    messages = [messages_for_image(image_path) for image_path in image_paths]
    output = pipe(
        text=messages,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
        enable_thinking=False,
    )
    if not isinstance(output, list) or len(output) != len(image_paths):
        raise RuntimeError(
            f"Expected {len(image_paths)} pipeline outputs, received {type(output).__name__} "
            f"with length {len(output) if isinstance(output, list) else 'unknown'}"
        )
    return [extract_generated_text(item) for item in output]


def resolve_model_snapshot(model_id: str, *, local_files_only: bool) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, local_files_only=local_files_only)


def build_internvl_transform(input_size: int) -> Any:
    import torchvision.transforms as transforms
    from torchvision.transforms.functional import InterpolationMode

    return transforms.Compose(
        [
            transforms.Lambda(
                lambda image: image.convert("RGB") if image.mode != "RGB" else image
            ),
            transforms.Resize(
                (input_size, input_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * target_area:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess_internvl(
    image: Any,
    *,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Any]:
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    target_ratios = sorted(
        {
            (width, height)
            for blocks in range(min_num, max_num + 1)
            for width in range(1, blocks + 1)
            for height in range(1, blocks + 1)
            if min_num <= width * height <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    target_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        original_width,
        original_height,
        image_size,
    )
    target_width = image_size * target_ratio[0]
    target_height = image_size * target_ratio[1]
    blocks = target_ratio[0] * target_ratio[1]
    resized = image.resize((target_width, target_height))
    processed = []
    columns = target_width // image_size
    for index in range(blocks):
        left = (index % columns) * image_size
        top = (index // columns) * image_size
        processed.append(
            resized.crop((left, top, left + image_size, top + image_size))
        )
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed


def load_internvl_image(image_path: str, *, max_num: int = 12) -> Any:
    import torch
    from PIL import Image

    transform = build_internvl_transform(448)
    with Image.open(image_path) as source:
        images = dynamic_preprocess_internvl(
            source.convert("RGB"), image_size=448, max_num=max_num
        )
    return torch.stack([transform(image) for image in images])


def load_internvl_adapter(
    spec: ModelSpec, *, local_files_only: bool
) -> InternVLAdapter:
    import torch
    from transformers import AutoModel, AutoTokenizer, PreTrainedModel

    torch.manual_seed(SEED)
    source = resolve_model_snapshot(
        spec.model_id, local_files_only=local_files_only
    )
    # InternVL3's remote wrapper predates the Transformers 5 tied-weight cache.
    # Its outer config does not tie embeddings, so an empty inherited mapping is
    # the compatible value expected by the current from_pretrained finalizer.
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}
    model = AutoModel.from_pretrained(
        source,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        local_files_only=True,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        trust_remote_code=True,
        use_fast=False,
        fix_mistral_regex=True,
        local_files_only=True,
    )
    return InternVLAdapter(
        model=model,
        tokenizer=tokenizer,
        device=next(model.parameters()).device,
    )


def infer_internvl_batch(
    adapter: InternVLAdapter,
    image_paths: list[str],
    max_new_tokens: int,
) -> list[str]:
    import torch

    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    predictions = []
    with torch.inference_mode():
        for image_path in image_paths:
            pixel_values = load_internvl_image(image_path).to(
                device=adapter.device, dtype=torch.bfloat16
            )
            response = adapter.model.chat(
                adapter.tokenizer,
                pixel_values,
                f"<image>\n{PROMPT}",
                generation_config,
            )
            predictions.append(str(response))
    return predictions


def require_minicpm_transformers() -> None:
    from packaging.version import Version

    current = importlib.metadata.version("transformers")
    if Version(current) < Version(MINICPM_MIN_TRANSFORMERS):
        raise RuntimeError(
            "MiniCPM-V-4.6 requires transformers>=5.7.0 according to its official "
            f"model card; current environment has transformers=={current}. The "
            "MiniCPMV4_6ForConditionalGeneration architecture is unavailable, so "
            "this model cannot be tested without upgrading the environment."
        )


def load_minicpm_adapter(
    spec: ModelSpec, *, local_files_only: bool
) -> MiniCPMAdapter:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    require_minicpm_transformers()
    torch.manual_seed(SEED)
    source = resolve_model_snapshot(
        spec.model_id, local_files_only=local_files_only
    )
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        source,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    ).eval()
    return MiniCPMAdapter(model=model, processor=processor)


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
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": image_path},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ]
            inputs = adapter.processor.apply_chat_template(
                messages,
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
            trimmed = generated[0, inputs.input_ids.shape[1] :]
            predictions.append(
                adapter.processor.decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
    return predictions


def load_ministral_adapter(
    spec: ModelSpec, *, local_files_only: bool
) -> MinistralAdapter:
    import torch
    from transformers import (
        FineGrainedFP8Config,
        Mistral3ForConditionalGeneration,
        MistralCommonBackend,
    )

    torch.manual_seed(SEED)
    source = resolve_model_snapshot(
        spec.model_id, local_files_only=local_files_only
    )
    tokenizer = MistralCommonBackend.from_pretrained(
        source, local_files_only=True
    )
    model = Mistral3ForConditionalGeneration.from_pretrained(
        source,
        device_map="auto",
        quantization_config=FineGrainedFP8Config(dequantize=True),
        local_files_only=True,
    ).eval()
    return MinistralAdapter(
        model=model,
        tokenizer=tokenizer,
        device=next(model.parameters()).device,
    )


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
                    generated[prompt_length:], skip_special_tokens=True
                )
            )
    return predictions


def load_inference_backend(
    model_alias: str,
    spec: ModelSpec,
    *,
    local_files_only: bool,
) -> Any:
    if model_alias in PIPELINE_MODEL_ALIASES:
        return load_pipeline(spec, local_files_only=local_files_only)
    if model_alias == "internvl3-8b":
        return load_internvl_adapter(spec, local_files_only=local_files_only)
    if model_alias == "minicpm-v46":
        return load_minicpm_adapter(spec, local_files_only=local_files_only)
    if model_alias == "ministral3-8b":
        return load_ministral_adapter(spec, local_files_only=local_files_only)
    raise ValueError(f"Unsupported model alias: {model_alias}")


def infer_model_batch(
    model_alias: str,
    backend: Any,
    image_paths: list[str],
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    if model_alias in PIPELINE_MODEL_ALIASES:
        return infer_batch(backend, image_paths, max_new_tokens, batch_size)
    if model_alias == "internvl3-8b":
        return infer_internvl_batch(backend, image_paths, max_new_tokens)
    if model_alias == "minicpm-v46":
        return infer_minicpm_batch(backend, image_paths, max_new_tokens)
    if model_alias == "ministral3-8b":
        return infer_ministral_batch(backend, image_paths, max_new_tokens)
    raise ValueError(f"Unsupported model alias: {model_alias}")


def result_record(
    sample: dict[str, Any], spec: ModelSpec, raw_prediction: str
) -> dict[str, Any]:
    scored = score_prediction(sample, raw_prediction)
    return {
        "id": sample["id"],
        "image_path": sample["image_path"],
        "ground_truth": sample["ground_truth"],
        "error_type": sample["error_type"],
        "corrected_text": sample["corrected_text"],
        "model_name": spec.model_id,
        "prompt": PROMPT,
        "raw_prediction": raw_prediction,
        **scored,
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def summarize_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    classifications = Counter(item["classification"] for item in records)
    by_error: dict[str, Counter[str]] = defaultdict(Counter)
    for item in records:
        for error_type in item["error_type"]:
            by_error[error_type]["total"] += 1
            by_error[error_type][item["classification"]] += 1
    return {
        "model_name": records[0]["model_name"] if records else None,
        "sample_count": total,
        "exact_match": classifications["error_preserved"],
        "exact_match_rate": classifications["error_preserved"] / total if total else 0,
        "error_preserved": classifications["error_preserved"],
        "error_preserved_rate": classifications["error_preserved"] / total if total else 0,
        "over_corrected": classifications["over_corrected"],
        "over_corrected_rate": classifications["over_corrected"] / total if total else 0,
        "recognition_error": classifications["recognition_error"],
        "recognition_error_rate": classifications["recognition_error"] / total if total else 0,
        "mean_cer": sum(item["cer"] for item in records) / total if total else 0,
        "by_error_type": {
            error_type: {
                **dict(counts),
                "over_correction_rate": counts["over_corrected"] / counts["total"],
            }
            for error_type, counts in sorted(by_error.items())
        },
        "over_correction_examples": [
            {
                key: item[key]
                for key in ("id", "ground_truth", "prediction", "corrected_text", "error_type")
            }
            for item in records
            if item["over_corrected"]
        ][:10],
    }


def write_combined_summary() -> None:
    summaries: list[dict[str, Any]] = []
    for spec in MODEL_SPECS.values():
        result_path = RESULTS_DIR / spec.output_name
        if result_path.exists():
            records = read_jsonl(result_path)
            if records:
                summaries.append(summarize_results(records))
    payload = {
        "seed": SEED,
        "target_sample_count": SAMPLE_COUNT,
        "classification_rule": (
            "strict: prediction==ground_truth => error_preserved; "
            "prediction==corrected_text => over_corrected; otherwise recognition_error"
        ),
        "models": summaries,
    }
    atomic_write_text(
        RESULTS_DIR / "smoke_summary.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    lines = [
        "# Error Handwriting Smoke Test",
        "",
        f"- Seed: `{SEED}`",
        f"- Fixed sample target: `{SAMPLE_COUNT}`",
        "- Classification: strict full-string matching against `ground_truth` and `corrected_text`.",
        "",
        "| Model | N | Exact / preserved | Over-corrected | Recognition error | Mean CER |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['model_name']} | {summary['sample_count']} | "
            f"{summary['error_preserved']} ({summary['error_preserved_rate']:.1%}) | "
            f"{summary['over_corrected']} ({summary['over_corrected_rate']:.1%}) | "
            f"{summary['recognition_error']} ({summary['recognition_error_rate']:.1%}) | "
            f"{summary['mean_cer']:.4f} |"
        )
    atomic_write_text(RESULTS_DIR / "smoke_summary.md", "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--limit", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    if not args.prepare_only and not args.model:
        parser.error("--model is required unless --prepare-only is used")
    if not 1 <= args.limit <= SAMPLE_COUNT:
        parser.error(f"--limit must be between 1 and {SAMPLE_COUNT}")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    samples = prepare_samples()
    print(f"Fixed samples: {SAMPLE_MANIFEST} ({len(samples)})", flush=True)
    if args.prepare_only:
        return 0

    spec = MODEL_SPECS[args.model]
    output_path = RESULTS_DIR / spec.output_name
    existing = read_jsonl(output_path) if output_path.exists() else []
    if any(item.get("model_name") != spec.model_id for item in existing):
        raise RuntimeError(f"Unexpected model record in {output_path}")
    completed_ids = {item["id"] for item in existing}
    pending = [item for item in samples[: args.limit] if item["id"] not in completed_ids]
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
        counts = Counter(item["classification"] for item in existing)
        progress = tqdm(
            total=args.limit,
            initial=min(len(existing), args.limit),
            desc=args.model,
            unit="img",
            dynamic_ncols=True,
            disable=args.no_progress,
        )
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            predictions = infer_model_batch(
                args.model,
                backend,
                [sample["image_path"] for sample in batch],
                args.max_new_tokens,
                len(batch),
            )
            for sample, raw_prediction in zip(batch, predictions):
                record = result_record(sample, spec, raw_prediction)
                append_jsonl(output_path, record)
                counts[record["classification"]] += 1
            progress.update(len(batch))
            progress.set_postfix(
                preserved=counts["error_preserved"],
                corrected=counts["over_corrected"],
                error=counts["recognition_error"],
                refresh=True,
            )
        progress.close()
    write_combined_summary()
    print(f"Results: {output_path}", flush=True)
    print(f"Summary: {RESULTS_DIR / 'smoke_summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
