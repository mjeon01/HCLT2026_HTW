#!/usr/bin/env python3
"""Download the exact VLM checkpoints listed for the smoke test."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from run_error_smoke_test import MODEL_SPECS


DEFAULT_MODELS = [
    "qwen35-9b",
    "qwen35-4b",
    "qwen3-vl-4b",
    "internvl3-8b",
    "ministral3-8b",
    "minicpm-v46",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=DEFAULT_MODELS,
    )
    parser.add_argument(
        "--use-xet",
        action="store_true",
        help="Use Hugging Face Xet instead of resumable HTTP downloads.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum concurrent Hugging Face file downloads (default: 2).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Download attempts per model; cached partial files are resumed (default: 3).",
    )
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    if args.retries < 1:
        parser.error("--retries must be positive")
    environment = os.environ.copy()
    if not args.use_xet:
        environment["HF_HUB_DISABLE_XET"] = "1"

    failures: list[str] = []
    for index, alias in enumerate(args.models, start=1):
        model_id = MODEL_SPECS[alias].model_id
        succeeded = False
        for attempt in range(1, args.retries + 1):
            print(
                f"\n[{index}/{len(args.models)}] Downloading {model_id} "
                f"(attempt {attempt}/{args.retries})",
                flush=True,
            )
            result = subprocess.run(
                [
                    "hf",
                    "download",
                    model_id,
                    "--max-workers",
                    str(args.max_workers),
                ],
                env=environment,
                check=False,
            )
            if result.returncode == 0:
                succeeded = True
                break
            if attempt < args.retries:
                print(
                    f"Download interrupted; resuming {model_id} in 3 seconds.",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(3)
        if not succeeded:
            failures.append(alias)
            print(f"Download failed: {model_id}", file=sys.stderr, flush=True)

    if failures:
        print(
            "Failed model aliases: " + ", ".join(failures),
            file=sys.stderr,
        )
        print(
            "A gated model may require `hf auth login` and acceptance of its model terms.",
            file=sys.stderr,
        )
        return 1
    print("\nAll requested checkpoints are available in the Hugging Face cache.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
