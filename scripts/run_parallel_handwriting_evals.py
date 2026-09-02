#!/usr/bin/env python3
"""Run the 2,000-image handwriting evaluations concurrently on selected GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from tqdm.auto import tqdm

from run_error_smoke_test import MODEL_SPECS
from run_handwriting_eval import (
    RESULTS_DIR,
    SOURCE_MANIFEST,
    TARGET_SAMPLE_COUNT,
    write_combined_summary,
)


DEFAULT_MODELS = list(MODEL_SPECS)


@dataclass
class RunningJob:
    model: str
    gpu: str
    process: subprocess.Popen[str]
    log_handle: TextIO
    log_path: Path


def count_records(path: Path, limit: int) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as handle:
        return min(sum(1 for line in handle if line.strip()), limit)


def tail(path: Path, lines: int = 30) -> str:
    if not path.is_file():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def parse_python_overrides(
    values: list[str],
    parser: argparse.ArgumentParser,
) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        model, separator, executable = value.partition("=")
        if not separator or not model or not executable:
            parser.error("--python-override must use MODEL=/path/to/python syntax")
        if model not in MODEL_SPECS:
            parser.error(f"Unknown model in --python-override: {model}")
        executable_path = Path(executable).expanduser()
        if not executable_path.is_absolute():
            executable_path = Path.cwd() / executable_path
        executable_path = executable_path.absolute()
        if not executable_path.is_file():
            parser.error(
                f"Python executable for {model} does not exist: {executable_path}"
            )
        overrides[model] = str(executable_path)
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=DEFAULT_MODELS,
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated physical GPU IDs (default: 0).",
    )
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=TARGET_SAMPLE_COUNT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
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
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument(
        "--python-override",
        action="append",
        default=[],
        metavar="MODEL=/PATH/TO/PYTHON",
    )
    args = parser.parse_args()
    source_manifest = args.source_manifest.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    python_overrides = parse_python_overrides(args.python_override, parser)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        parser.error("--gpus must contain at least one GPU ID")
    if args.workers_per_gpu < 1:
        parser.error("--workers-per-gpu must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 1 <= args.limit <= TARGET_SAMPLE_COUNT:
        parser.error(
            f"--limit must be between 1 and {TARGET_SAMPLE_COUNT}"
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    bars: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for position, model in enumerate(args.models):
        result_path = results_dir / MODEL_SPECS[model].output_name
        counts[model] = count_records(result_path, args.limit)
        bars[model] = tqdm(
            total=args.limit,
            initial=counts[model],
            desc=model,
            unit="img",
            position=position,
            dynamic_ncols=True,
        )

    queue = deque(model for model in args.models if counts[model] < args.limit)
    available_gpus = deque(
        gpu for gpu in gpus for _ in range(args.workers_per_gpu)
    )
    running: dict[str, RunningJob] = {}
    failures: list[RunningJob] = []
    runner = Path(__file__).with_name("run_handwriting_eval.py")
    parallelism = min(len(args.models), len(available_gpus))
    print(
        f"Starting up to {parallelism} model(s) concurrently on physical "
        f"GPU(s) {','.join(gpus)}.",
        flush=True,
    )
    if args.workers_per_gpu > 1:
        print(
            "WARNING: concurrent model processes share GPU memory.",
            file=sys.stderr,
            flush=True,
        )

    try:
        while queue or running:
            while queue and available_gpus:
                model = queue.popleft()
                gpu = available_gpus.popleft()
                log_path = log_dir / f"{model}.log"
                log_handle = log_path.open("a", encoding="utf-8")
                command = [
                    python_overrides.get(model, sys.executable),
                    str(runner),
                    "--model",
                    model,
                    "--limit",
                    str(args.limit),
                    "--batch-size",
                    str(args.batch_size),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--source-manifest",
                    str(source_manifest),
                    "--results-dir",
                    str(results_dir),
                    "--report-title",
                    args.report_title,
                    "--no-progress",
                    "--skip-summary",
                ]
                if not args.allow_downloads:
                    command.append("--local-files-only")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                environment["PYTHONUNBUFFERED"] = "1"
                process = subprocess.Popen(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                running[model] = RunningJob(
                    model=model,
                    gpu=gpu,
                    process=process,
                    log_handle=log_handle,
                    log_path=log_path,
                )
                bars[model].set_postfix_str(f"gpu={gpu} loading")

            time.sleep(0.5)
            for model, job in list(running.items()):
                result_path = results_dir / MODEL_SPECS[model].output_name
                current = count_records(result_path, args.limit)
                if current > counts[model]:
                    bars[model].update(current - counts[model])
                    counts[model] = current
                    bars[model].set_postfix_str(f"gpu={job.gpu} infer")
                return_code = job.process.poll()
                if return_code is None:
                    continue
                job.log_handle.close()
                available_gpus.append(job.gpu)
                del running[model]
                if return_code == 0 and counts[model] >= args.limit:
                    bars[model].set_postfix_str(f"gpu={job.gpu} done")
                else:
                    bars[model].set_postfix_str(f"gpu={job.gpu} FAILED")
                    failures.append(job)
    except KeyboardInterrupt:
        for job in running.values():
            job.process.terminate()
            job.log_handle.close()
        raise
    finally:
        for bar in bars.values():
            bar.close()

    write_combined_summary(
        results_dir,
        source_manifest,
        args.limit,
        args.report_title,
    )
    if failures:
        print("\nFailed model jobs:", file=sys.stderr)
        for job in failures:
            print(f"\n[{job.model}] {job.log_path}", file=sys.stderr)
            print(tail(job.log_path), file=sys.stderr)
        return 1
    print(
        f"Completed {len(args.models)} model(s). Summary: "
        f"{results_dir / '결과.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
