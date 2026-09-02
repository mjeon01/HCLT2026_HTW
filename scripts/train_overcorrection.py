#!/usr/bin/env python3
"""Train Qwen3.5-4B for verbatim handwriting transcription.

Only the vision merger (the Qwen projector) and, in ``projector_lora`` mode,
LoRA adapters are trainable.  The vision encoder and all remaining base-model
weights stay frozen.  Error exposure is controlled by sampling frequency, not
by loss weights.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTRUCTION = (
    "이미지에 작성된 한국어 문장을 맞춤법이나 문법을 수정하지 말고 그대로 전사하세요."
)


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


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    return config


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class ExposureCounts:
    normal: int = 0
    error: int = 0

    @property
    def total(self) -> int:
        return self.normal + self.error

    @property
    def error_ratio(self) -> float:
        return self.error / self.total if self.total else 0.0


class ShuffledPool:
    def __init__(self, records: list[dict[str, Any]], seed: int) -> None:
        if not records:
            raise ValueError("Sampling pool must not be empty")
        self.records = records
        self.random = random.Random(seed)
        self.order: list[int] = []
        self.position = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        self.order = list(range(len(self.records)))
        self.random.shuffle(self.order)
        self.position = 0

    def next(self) -> dict[str, Any]:
        if self.position >= len(self.order):
            self._reshuffle()
        record = self.records[self.order[self.position]]
        self.position += 1
        return record


class RatioSampler:
    """Deterministic exact-quota sampler with equal per-sample loss weights."""

    def __init__(
        self,
        normal: list[dict[str, Any]],
        error: list[dict[str, Any]],
        error_ratio: float,
        seed: int,
        total_exposures: int,
    ) -> None:
        if not 0.0 <= error_ratio <= 1.0:
            raise ValueError("error_sampling_ratio must be in [0, 1]")
        if total_exposures < 1:
            raise ValueError("total_exposures must be positive")
        if error_ratio < 1.0 and not normal:
            raise ValueError("Normal pool is empty")
        if error_ratio > 0.0 and not error:
            raise ValueError("Error pool is empty")
        self.error_ratio = error_ratio
        self.normal_pool = ShuffledPool(normal, seed + 1) if normal else None
        self.error_pool = ShuffledPool(error, seed + 2) if error else None
        self.planned_error = round(total_exposures * error_ratio)
        self.planned_normal = total_exposures - self.planned_error
        self.decisions = [True] * self.planned_error + [False] * self.planned_normal
        random.Random(seed).shuffle(self.decisions)
        self.position = 0
        self.counts = ExposureCounts()

    def next(self) -> dict[str, Any]:
        if self.position >= len(self.decisions):
            raise RuntimeError("The planned exposure quota has been exhausted")
        choose_error = self.decisions[self.position]
        self.position += 1
        if choose_error:
            if self.error_pool is None:
                raise RuntimeError("Error pool was not initialized")
            self.counts.error += 1
            return self.error_pool.next()
        if self.normal_pool is None:
            raise RuntimeError("Normal pool was not initialized")
        self.counts.normal += 1
        return self.normal_pool.next()


def validate_manifests(
    normal: list[dict[str, Any]],
    error: list[dict[str, Any]],
    validation: list[dict[str, Any]],
) -> None:
    for name, records, expected_error in (
        ("normal", normal, False),
        ("error", error, True),
    ):
        if not records:
            raise RuntimeError(f"Empty {name} training manifest")
        if any(bool(row.get("has_error")) is not expected_error for row in records):
            raise RuntimeError(f"Wrong partition record in {name} manifest")
    for name, records in (
        ("normal", normal),
        ("error", error),
        ("validation", validation),
    ):
        ids = [str(row["id"]) for row in records]
        hashes = [str(row["image_sha256"]) for row in records]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Duplicate IDs in {name} manifest")
        if len(hashes) != len(set(hashes)):
            raise RuntimeError(f"Duplicate image hashes in {name} manifest")
        for row in records:
            if row.get("target_field") != "rendered_text":
                raise RuntimeError(
                    f"Non-verbatim target field for {row.get('id')}: "
                    f"{row.get('target_field')}"
                )
            if not str(row.get("target") or "").strip():
                raise RuntimeError(f"Empty target for {row.get('id')}")
            if not Path(row["image_path"]).is_file():
                raise FileNotFoundError(row["image_path"])
            if row.get("has_error") and row["target"] == row.get("corrected_text"):
                raise RuntimeError(
                    f"Error target equals corrected text for {row.get('id')}"
                )
    train_ids = {row["id"] for row in normal + error}
    train_hashes = {row["image_sha256"] for row in normal + error}
    if train_ids & {row["id"] for row in validation}:
        raise RuntimeError("Train/validation identifier leakage")
    if train_hashes & {row["image_sha256"] for row in validation}:
        raise RuntimeError("Train/validation image leakage")


def messages(record: dict[str, Any], instruction: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": record["image_path"]},
                {"type": "text", "text": instruction},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": record["target"]}],
        },
    ]


def find_last_subsequence(values: list[int], needle: list[int]) -> int:
    if not needle:
        raise ValueError("Target token sequence is empty")
    for start in range(len(values) - len(needle), -1, -1):
        if values[start : start + len(needle)] == needle:
            return start
    raise RuntimeError("Could not locate the verbatim target token sequence")


def build_batch(
    processor: Any,
    record: dict[str, Any],
    instruction: str,
    device: Any,
) -> dict[str, Any]:
    import torch

    encoded = processor.apply_chat_template(
        messages(record, instruction),
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    encoded.pop("assistant_masks", None)
    input_ids = encoded["input_ids"][0].tolist()
    target_ids = processor.tokenizer(
        record["target"], add_special_tokens=False
    )["input_ids"]
    target_start = find_last_subsequence(input_ids, target_ids)
    labels = torch.full_like(encoded["input_ids"], -100)
    # Include the assistant end marker so generation learns when to stop.
    labels[:, target_start:] = encoded["input_ids"][:, target_start:]
    if not torch.any(labels != -100):
        raise RuntimeError(f"No supervised tokens for {record['id']}")
    encoded["labels"] = labels
    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }


def find_projector(model: Any) -> tuple[str, Any]:
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith("model.visual.merger")
    ]
    if len(matches) != 1:
        names = [name for name, _ in matches]
        raise RuntimeError(f"Expected one Qwen vision merger, found {names}")
    return matches[0]


def configure_trainable_model(
    model: Any,
    config: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:

    train_mode = str(config["train_mode"])
    if train_mode not in {"projector_lora", "projector_only"}:
        raise ValueError(f"Unsupported train_mode: {train_mode}")
    for parameter in model.parameters():
        parameter.requires_grad = False

    has_lora = train_mode == "projector_lora"
    if has_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora = config["lora"]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=list(lora["target_modules"]),
            bias="none",
        )
        model = get_peft_model(model, peft_config)

    projector_name, projector = find_projector(model)
    for parameter in projector.parameters():
        parameter.requires_grad = True

    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    projector_parameters = [
        parameter
        for name, parameter in trainable
        if f"{projector_name}." in name
    ]
    lora_parameters = [
        parameter for name, parameter in trainable if "lora_" in name
    ]
    unexpected = [
        name
        for name, _ in trainable
        if f"{projector_name}." not in name and "lora_" not in name
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters: {unexpected[:10]}")
    if not projector_parameters:
        raise RuntimeError("Projector has no trainable parameters")
    if has_lora and not lora_parameters:
        raise RuntimeError("LoRA has no trainable parameters")
    if not has_lora and lora_parameters:
        raise RuntimeError("LoRA parameters are trainable in projector-only mode")

    vision_trainable_outside_projector = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and ".visual." in name
        and f"{projector_name}." not in name
    ]
    if vision_trainable_outside_projector:
        raise RuntimeError(
            "Vision encoder is not frozen: "
            f"{vision_trainable_outside_projector[:10]}"
        )
    parameter_counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for _, parameter in trainable),
        "projector_trainable": sum(parameter.numel() for parameter in projector_parameters),
        "lora_trainable": sum(parameter.numel() for parameter in lora_parameters),
        "projector_module": projector_name,
        "vision_encoder_frozen": True,
        "has_lora": has_lora,
    }
    return model, {
        "projector": projector,
        "projector_parameters": projector_parameters,
        "lora_parameters": lora_parameters,
        "counts": parameter_counts,
    }


def build_optimizer(model_parts: dict[str, Any], config: dict[str, Any]) -> Any:
    import torch

    groups = [
        {
            "params": model_parts["projector_parameters"],
            "lr": float(config["projector_learning_rate"]),
            "name": "projector",
        }
    ]
    if model_parts["lora_parameters"]:
        groups.append(
            {
                "params": model_parts["lora_parameters"],
                "lr": float(config["lora_learning_rate"]),
                "name": "lora",
            }
        )
    kwargs = {
        "lr": float(config["projector_learning_rate"]),
        "weight_decay": float(config.get("weight_decay", 0.01)),
        "betas": tuple(config.get("adam_betas", [0.9, 0.999])),
        "eps": float(config.get("adam_epsilon", 1e-8)),
    }
    try:
        return torch.optim.AdamW(groups, fused=True, **kwargs)
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(groups, **kwargs)


def gradient_status(parameters: list[Any]) -> dict[str, Any]:
    import torch

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    return {
        "parameter_tensors_with_grad": len(gradients),
        "finite": bool(gradients) and all(torch.isfinite(gradient).all().item() for gradient in gradients),
        "nonzero": bool(gradients) and any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients),
    }


def save_trainable_checkpoint(
    model: Any,
    processor: Any,
    model_parts: dict[str, Any],
    destination: Path,
    metadata: dict[str, Any],
    *,
    final: bool,
) -> None:
    import torch

    destination.mkdir(parents=True, exist_ok=True)
    projector_state = {
        key: value.detach().cpu()
        for key, value in model_parts["projector"].state_dict().items()
    }
    torch.save(projector_state, destination / "projector.pt")
    if model_parts["counts"]["has_lora"]:
        model.save_pretrained(destination / "adapter")
    if final:
        processor.save_pretrained(destination / "processor")
    atomic_write_text(
        destination / "run_metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )


def evaluate_validation_loss(
    model: Any,
    processor: Any,
    validation: list[dict[str, Any]],
    instruction: str,
    device: Any,
    limit: int,
) -> float:
    import torch

    if limit <= 0 or not validation:
        return float("nan")
    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for record in validation[:limit]:
            batch = build_batch(processor, record, instruction, device)
            loss = model(**batch).loss
            losses.append(float(loss.detach().cpu()))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sanity-check", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.sanity_check:
        config["max_steps"] = int(config.get("sanity_steps", 20))
        config["save_steps"] = 0
        config["validation_max_samples"] = 0
        config["output_dir"] = str(
            ROOT / "outputs" / "overcorrection" / f"sanity_{config['setting']}"
        )
    if int(config.get("per_device_train_batch_size", 1)) != 1:
        raise SystemExit("This deterministic sampler currently requires per_device_train_batch_size=1")
    max_steps = int(config["max_steps"])
    gradient_accumulation_steps = int(config["gradient_accumulation_steps"])
    if max_steps < 1 or gradient_accumulation_steps < 1:
        raise SystemExit("max_steps and gradient_accumulation_steps must be positive")

    output_dir = resolve_path(config["output_dir"])
    train_log = output_dir / "train_log.jsonl"
    final_dir = output_dir / "final"
    if train_log.exists() or final_dir.exists():
        raise SystemExit(
            f"Refusing to mix with an existing run: {output_dir}. "
            "Choose a new --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "resolved_source_config.yaml")

    seed = int(config.get("seed", 42))
    seed_everything(seed)
    normal_manifest = resolve_path(config["train_normal_manifest"])
    error_manifest = resolve_path(config["train_error_manifest"])
    validation_manifest = resolve_path(config["validation_manifest"])
    normal = read_jsonl(normal_manifest)
    error = read_jsonl(error_manifest)
    validation = read_jsonl(validation_manifest)
    validate_manifests(normal, error, validation)
    leakage_report = resolve_path(config["leakage_report"])
    leakage = json.loads(leakage_report.read_text(encoding="utf-8"))
    if leakage.get("leakage_checks", {}).get("passed") is not True:
        raise RuntimeError(f"Leakage report did not pass: {leakage_report}")

    import torch
    from tqdm.auto import tqdm
    from transformers import (
        AutoProcessor,
        Qwen3_5ForConditionalGeneration,
        get_cosine_schedule_with_warmup,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support bf16")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model_name = str(config["model_name"])
    local_files_only = bool(config.get("local_files_only", True))
    processor = AutoProcessor.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation=str(config.get("attn_implementation", "sdpa")),
        local_files_only=local_files_only,
    )
    model.config.use_cache = False
    model, model_parts = configure_trainable_model(model, config)
    model.config.use_cache = False
    if bool(config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.to(device)
    model.train()

    optimizer = build_optimizer(model_parts, config)
    warmup_steps = round(max_steps * float(config.get("warmup_ratio", 0.03)))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    sampler = RatioSampler(
        normal,
        error,
        float(config["error_sampling_ratio"]),
        seed,
        max_steps * gradient_accumulation_steps,
    )
    instruction = str(config.get("instruction", DEFAULT_INSTRUCTION))
    effective_batch_size = gradient_accumulation_steps
    parameter_counts = model_parts["counts"]
    start_metadata = {
        "setting": config["setting"],
        "train_mode": config["train_mode"],
        "model_name": model_name,
        "physical_gpu_requested": 1,
        "visible_cuda_device": 0,
        "gpu_name": torch.cuda.get_device_name(device),
        "normal_pool_count": len(normal),
        "error_pool_count": len(error),
        "validation_count": len(validation),
        "error_sampling_ratio_requested": float(config["error_sampling_ratio"]),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "max_steps": max_steps,
        "total_exposures_planned": sampler.planned_normal + sampler.planned_error,
        "normal_exposures_planned": sampler.planned_normal,
        "error_exposures_planned": sampler.planned_error,
        "normal_resampled_exposures_planned": max(0, sampler.planned_normal - len(normal)),
        "error_resampled_exposures_planned": max(0, sampler.planned_error - len(error)),
        "lora_learning_rate": float(config["lora_learning_rate"]),
        "projector_learning_rate": float(config["projector_learning_rate"]),
        "parameter_counts": parameter_counts,
        "instruction": instruction,
        "target_field": "rendered_text",
        "leakage_report": str(leakage_report),
        "config": config,
    }
    atomic_write_text(
        output_dir / "run_start.json",
        json.dumps(start_metadata, ensure_ascii=False, indent=2) + "\n",
    )
    print(
        f"학습 1개 | 총 {sampler.planned_normal + sampler.planned_error:,}회 "
        f"(정상 {sampler.planned_normal:,} / 오류 {sampler.planned_error:,})",
        flush=True,
    )
    print(
        f"{max_steps:,} steps x batch {effective_batch_size} | "
        f"{torch.cuda.get_device_name(device)} | 결과: {output_dir}",
        flush=True,
    )

    progress = tqdm(
        range(1, max_steps + 1),
        desc=f"Setting {config['setting']}",
        unit="step",
        dynamic_ncols=True,
    )
    started = time.perf_counter()
    gradient_check: dict[str, Any] | None = None
    save_steps = int(config.get("save_steps", 0))
    logging_steps = int(config.get("logging_steps", 1))
    all_trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer.zero_grad(set_to_none=True)
    for optimizer_step in progress:
        accumulated_loss = 0.0
        for _ in range(gradient_accumulation_steps):
            record = sampler.next()
            batch = build_batch(processor, record, instruction, device)
            outputs = model(**batch)
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at optimizer step {optimizer_step}: {loss.item()}"
                )
            accumulated_loss += float(loss.detach().cpu())
            (loss / gradient_accumulation_steps).backward()
        if optimizer_step == 1:
            gradient_check = {
                "projector": gradient_status(model_parts["projector_parameters"]),
                "lora": gradient_status(model_parts["lora_parameters"])
                if model_parts["lora_parameters"]
                else None,
            }
            if not gradient_check["projector"]["finite"] or not gradient_check["projector"]["nonzero"]:
                raise RuntimeError(f"Projector gradient check failed: {gradient_check}")
            if model_parts["lora_parameters"] and (
                not gradient_check["lora"]["finite"]
                or not gradient_check["lora"]["nonzero"]
            ):
                raise RuntimeError(f"LoRA gradient check failed: {gradient_check}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            all_trainable_parameters,
            float(config.get("max_grad_norm", 1.0)),
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        mean_loss = accumulated_loss / gradient_accumulation_steps
        elapsed = time.perf_counter() - started
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        lrs = {group.get("name", str(index)): group["lr"] for index, group in enumerate(optimizer.param_groups)}
        progress.set_postfix(
            loss=f"{mean_loss:.4f}",
            seen=f"{sampler.counts.total:,}/{sampler.planned_normal + sampler.planned_error:,}",
            normal=f"{sampler.counts.normal:,}",
            error=f"{sampler.counts.error:,}",
            gpu=f"{peak_gb:.1f}GB",
        )
        record = {
            "step": optimizer_step,
            "total_steps": max_steps,
            "loss": mean_loss,
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "learning_rates": lrs,
            "elapsed_seconds": elapsed,
            "peak_memory_gb": peak_gb,
            "normal_exposures": sampler.counts.normal,
            "error_exposures": sampler.counts.error,
            "realized_error_ratio": sampler.counts.error_ratio,
        }
        if optimizer_step % logging_steps == 0:
            append_jsonl(train_log, record)
        if save_steps > 0 and optimizer_step % save_steps == 0 and optimizer_step < max_steps:
            save_trainable_checkpoint(
                model,
                processor,
                model_parts,
                output_dir / f"checkpoint-{optimizer_step:06d}",
                {**start_metadata, **record, "gradient_check": gradient_check},
                final=False,
            )

    progress.close()
    if (
        sampler.counts.normal != sampler.planned_normal
        or sampler.counts.error != sampler.planned_error
    ):
        raise RuntimeError(
            "Exposure quota mismatch: "
            f"normal={sampler.counts.normal}/{sampler.planned_normal}, "
            f"error={sampler.counts.error}/{sampler.planned_error}"
        )
    validation_loss = evaluate_validation_loss(
        model,
        processor,
        validation,
        instruction,
        device,
        min(int(config.get("validation_max_samples", 128)), len(validation)),
    )
    final_metadata = {
        **start_metadata,
        "completed_steps": max_steps,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        "normal_exposures": sampler.counts.normal,
        "error_exposures": sampler.counts.error,
        "realized_error_ratio": sampler.counts.error_ratio,
        "validation_loss": validation_loss,
        "gradient_check": gradient_check,
        "status": "complete",
    }
    save_trainable_checkpoint(
        model,
        processor,
        model_parts,
        final_dir,
        final_metadata,
        final=True,
    )
    atomic_write_text(
        output_dir / "run_summary.json",
        json.dumps(final_metadata, ensure_ascii=False, indent=2) + "\n",
    )
    print(
        f"완료 | 정상 {sampler.counts.normal:,} / 오류 {sampler.counts.error:,} "
        f"| 결과: {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
