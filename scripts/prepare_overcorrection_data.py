#!/usr/bin/env python3
"""Build leak-free train/validation manifests for over-correction SFT.

The two existing synthetic benchmark manifests are immutable test sets.  This
script excludes them, their source identifiers, their text variants, and image
content hashes before creating deterministic training and validation pools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_ROOT = ROOT / "data" / "generated-handwriting-project-b"
SOURCE_METADATA = SYNTHETIC_ROOT / "metadata.jsonl"
NORMAL_BENCHMARK = ROOT / "data" / "benchmark" / "eval_synthetic_normal_2000.jsonl"
ERROR_BENCHMARK = ROOT / "data" / "benchmark" / "eval_synthetic_error_2000.jsonl"
OUTPUT_DIR = ROOT / "data" / "overcorrection"
SEED = 42
ID_FIELDS = ("sid", "sample_id", "source_sid", "record_id")
TEXT_FIELDS = ("text", "rendered_text", "reference_text", "text_src")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    )


def canonical_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFC", str(value or ""))
    normalized = normalized.replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"\s+", " ", normalized).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(namespace: str, row: dict[str, Any]) -> str:
    value = f"{namespace}:{row.get('source_sid') or row.get('sid')}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def row_identifiers(row: dict[str, Any]) -> set[str]:
    return {
        canonical_text(row.get(field))
        for field in ID_FIELDS
        if canonical_text(row.get(field))
    }


def row_texts(row: dict[str, Any]) -> set[str]:
    return {
        canonical_text(row.get(field))
        for field in TEXT_FIELDS
        if canonical_text(row.get(field))
    }


def benchmark_exclusions(
    benchmarks: list[dict[str, Any]],
) -> tuple[set[str], set[str], set[str], set[str]]:
    paths: set[str] = set()
    identifiers: set[str] = set()
    texts: set[str] = set()
    hashes: set[str] = set()
    for row in benchmarks:
        paths.add(str(Path(row["image_path"]).resolve()))
        identifiers.update(row_identifiers(row))
        texts.add(canonical_text(row.get("ground_truth")))
        texts.add(canonical_text(row.get("corrected_text")))
        digest = canonical_text(row.get("image_sha256"))
        if digest:
            hashes.add(digest)
    texts.discard("")
    return paths, identifiers, texts, hashes


def classify_rejection(
    row: dict[str, Any],
    image_path: Path,
    image_hash: str,
    excluded_paths: set[str],
    excluded_ids: set[str],
    excluded_texts: set[str],
    excluded_hashes: set[str],
) -> str | None:
    if not image_path.is_file():
        return "missing_image"
    if not canonical_text(row.get("rendered_text")):
        return "empty_rendered_text"
    if row.get("has_unrendered"):
        return "unrendered_character"
    if str(image_path.resolve()) in excluded_paths:
        return "benchmark_image_path"
    if row_identifiers(row) & excluded_ids:
        return "benchmark_identifier"
    if row_texts(row) & excluded_texts:
        return "benchmark_text"
    if image_hash in excluded_hashes:
        return "benchmark_image_hash"
    if bool(row.get("has_error")):
        reference = canonical_text(row.get("reference_text"))
        target = canonical_text(row.get("rendered_text"))
        if not reference or target == reference:
            return "invalid_error_target"
    return None


def manifest_record(row: dict[str, Any], image_path: Path, image_hash: str) -> dict[str, Any]:
    target = canonical_text(row["rendered_text"])
    return {
        "id": row["sid"],
        "image_path": str(image_path.resolve()),
        "target": target,
        "target_field": "rendered_text",
        "partition": "error" if row.get("has_error") else "normal",
        "has_error": bool(row.get("has_error")),
        "corrected_text": canonical_text(row.get("reference_text")),
        "source_split": row["split"],
        "source_sid": row.get("source_sid"),
        "sample_id": row.get("sample_id"),
        "record_id": row.get("record_id"),
        "text_src": canonical_text(row.get("text_src")),
        "writer": row.get("writer"),
        "style_ref": row.get("style_ref"),
        "image_sha256": image_hash,
        "errors": row.get("errors") or [],
    }


def overlap_counts(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, int]:
    right_paths = {row["image_path"] for row in right}
    right_hashes = {row["image_sha256"] for row in right}
    right_ids = {
        canonical_text(row.get(field))
        for row in right
        for field in ID_FIELDS
        if canonical_text(row.get(field))
    }
    right_texts = {
        canonical_text(row.get(field))
        for row in right
        for field in ("target", "corrected_text", "text_src")
        if canonical_text(row.get(field))
    }
    return {
        "image_path": sum(row["image_path"] in right_paths for row in left),
        "image_sha256": sum(row["image_sha256"] in right_hashes for row in left),
        "identifier": sum(bool(row_identifiers(row) & right_ids) for row in left),
        "text": sum(
            bool(
                {
                    canonical_text(row.get(field))
                    for field in ("target", "corrected_text", "text_src")
                    if canonical_text(row.get(field))
                }
                & right_texts
            )
            for row in left
        ),
    }


def choose_validation(
    dev_test: list[dict[str, Any]],
    train: list[dict[str, Any]],
    validation_count: int,
) -> list[dict[str, Any]]:
    quotas = {
        False: validation_count // 2,
        True: validation_count - validation_count // 2,
    }
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for has_error in (False, True):
        external = sorted(
            (row for row in dev_test if row["has_error"] is has_error),
            key=lambda row: stable_key("validation-existing", row),
        )
        take = min(quotas[has_error], len(external))
        selected.extend(external[:take])
        selected_ids.update(row["id"] for row in external[:take])
        shortage = quotas[has_error] - take
        if shortage:
            supplemental = sorted(
                (row for row in train if row["has_error"] is has_error),
                key=lambda row: stable_key("validation-supplement", row),
            )
            if len(supplemental) < shortage:
                raise RuntimeError(
                    f"Need {shortage} supplemental validation records for "
                    f"has_error={has_error}, found {len(supplemental)}"
                )
            selected.extend(supplemental[:shortage])
            selected_ids.update(row["id"] for row in supplemental[:shortage])
    if len(selected) != validation_count or len(selected_ids) != validation_count:
        raise RuntimeError("Validation selection is not unique or has the wrong size")
    return sorted(selected, key=lambda row: stable_key("validation-order", row))


def remove_validation_derivatives(
    train: list[dict[str, Any]], validation: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    validation_paths = {row["image_path"] for row in validation}
    validation_hashes = {row["image_sha256"] for row in validation}
    validation_ids = {
        canonical_text(row.get(field))
        for row in validation
        for field in ID_FIELDS
        if canonical_text(row.get(field))
    }
    validation_texts = {
        canonical_text(row.get(field))
        for row in validation
        for field in ("target", "corrected_text", "text_src")
        if canonical_text(row.get(field))
    }
    kept: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for row in train:
        reason = None
        if row["image_path"] in validation_paths:
            reason = "validation_image_path"
        elif row["image_sha256"] in validation_hashes:
            reason = "validation_image_hash"
        elif row_identifiers(row) & validation_ids:
            reason = "validation_identifier"
        elif {
            canonical_text(row.get(field))
            for field in ("target", "corrected_text", "text_src")
            if canonical_text(row.get(field))
        } & validation_texts:
            reason = "validation_text"
        if reason:
            rejected[reason] += 1
        else:
            kept.append(row)
    return kept, rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=SOURCE_METADATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--validation-count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validation_count < 2:
        raise SystemExit("--validation-count must be at least 2")
    random.seed(args.seed)
    metadata_path = args.metadata.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    source_rows = read_jsonl(metadata_path)
    benchmarks = read_jsonl(NORMAL_BENCHMARK) + read_jsonl(ERROR_BENCHMARK)
    excluded_paths, excluded_ids, excluded_texts, excluded_hashes = benchmark_exclusions(
        benchmarks
    )

    accepted_train: list[dict[str, Any]] = []
    accepted_dev_test: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    for index, source in enumerate(source_rows, start=1):
        image_path = (SYNTHETIC_ROOT / source["image"]).resolve()
        if not image_path.is_file():
            rejections["missing_image"] += 1
            continue
        image_hash = sha256_file(image_path)
        reason = classify_rejection(
            source,
            image_path,
            image_hash,
            excluded_paths,
            excluded_ids,
            excluded_texts,
            excluded_hashes,
        )
        if reason:
            rejections[reason] += 1
            continue
        record = manifest_record(source, image_path, image_hash)
        if source["split"] == "train":
            accepted_train.append(record)
        elif source["split"] in {"dev", "test"}:
            accepted_dev_test.append(record)
        else:
            rejections["unknown_split"] += 1
        if index % 10_000 == 0:
            print(f"Hashed and checked {index:,}/{len(source_rows):,} images", flush=True)

    validation = choose_validation(
        accepted_dev_test,
        accepted_train,
        args.validation_count,
    )
    train, validation_rejections = remove_validation_derivatives(
        accepted_train,
        validation,
    )
    train_normal = sorted(
        (row for row in train if not row["has_error"]),
        key=lambda row: stable_key("train-normal", row),
    )
    train_error = sorted(
        (row for row in train if row["has_error"]),
        key=lambda row: stable_key("train-error", row),
    )

    benchmark_projection = [
        {
            "image_path": str(Path(row["image_path"]).resolve()),
            "image_sha256": canonical_text(row.get("image_sha256")),
            "source_sid": row.get("source_sid"),
            "record_id": row.get("record_id"),
            "target": canonical_text(row.get("ground_truth")),
            "corrected_text": canonical_text(row.get("corrected_text")),
            "text_src": "",
        }
        for row in benchmarks
    ]
    train_all = train_normal + train_error
    train_benchmark_overlap = overlap_counts(train_all, benchmark_projection)
    validation_benchmark_overlap = overlap_counts(validation, benchmark_projection)
    train_validation_overlap = overlap_counts(train_all, validation)
    all_overlap_counts = (
        list(train_benchmark_overlap.values())
        + list(validation_benchmark_overlap.values())
        + list(train_validation_overlap.values())
    )
    if any(all_overlap_counts):
        raise RuntimeError(
            "Leakage audit failed: "
            f"train/benchmark={train_benchmark_overlap}, "
            f"validation/benchmark={validation_benchmark_overlap}, "
            f"train/validation={train_validation_overlap}"
        )
    for partition_name, records in (
        ("train_normal", train_normal),
        ("train_error", train_error),
        ("validation", validation),
    ):
        if len({row["id"] for row in records}) != len(records):
            raise RuntimeError(f"Duplicate IDs in {partition_name}")
        if len({row["image_sha256"] for row in records}) != len(records):
            raise RuntimeError(f"Duplicate image hashes in {partition_name}")
        if any(row["target"] != canonical_text(row["target"]) for row in records):
            raise RuntimeError(f"Non-canonical target in {partition_name}")
        if any(row["target_field"] != "rendered_text" for row in records):
            raise RuntimeError(f"Non-verbatim target in {partition_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train_normal.jsonl", train_normal)
    write_jsonl(output_dir / "train_error.jsonl", train_error)
    write_jsonl(output_dir / "validation.jsonl", validation)
    report = {
        "source_metadata": str(metadata_path),
        "normal_benchmark": str(NORMAL_BENCHMARK.resolve()),
        "error_benchmark": str(ERROR_BENCHMARK.resolve()),
        "seed": args.seed,
        "source_count": len(source_rows),
        "benchmark_count": len(benchmarks),
        "train_normal_count": len(train_normal),
        "train_error_count": len(train_error),
        "validation_count": len(validation),
        "validation_partition": dict(
            Counter(row["partition"] for row in validation)
        ),
        "validation_source_split": dict(
            Counter(row["source_split"] for row in validation)
        ),
        "source_rejection_counts": dict(sorted(rejections.items())),
        "validation_derivative_rejection_counts": dict(
            sorted(validation_rejections.items())
        ),
        "leakage_checks": {
            "train_vs_benchmark": train_benchmark_overlap,
            "validation_vs_benchmark": validation_benchmark_overlap,
            "train_vs_validation": train_validation_overlap,
            "passed": True,
        },
        "verbatim_target": {
            "field": "rendered_text",
            "error_target_is_corrected_text": False,
            "all_targets_nonempty": True,
        },
    }
    atomic_write_text(
        output_dir / "leakage_report.json",
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
