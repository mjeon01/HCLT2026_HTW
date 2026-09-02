#!/usr/bin/env python3
"""Build the 4,000-line AI Hub 605 handwriting evaluation set.

The final split contains every eligible Validation T.Tablet/R.Free line (2,578
with the current labels/images) plus 1,422 Training lines.  Training selection
uses at most one line per source image and avoids GT values already used by the
Validation split whenever possible.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from build_aihub605_eval import (
    atomic_write_json,
    atomic_write_text,
    index_images,
    load_label_pages,
    stable_seed,
    validate_image_matching,
)
from preview_aihub605_lines import LineRecord, add_visual_gt_coverage, extract_all_lines


DATA_ROOT = Path("data/053.대용량_손글씨_OCR_데이터/01.데이터")
DEFAULT_VALIDATION_LABELS = (
    DATA_ROOT
    / "2.Validation/라벨링데이터/VL/라벨/HW-OCR/4.Validation/T.Tablet/R.Free"
)
DEFAULT_VALIDATION_IMAGES = (
    DATA_ROOT
    / "2.Validation/원천데이터/VS/HW-OCR/4.Validation/T.Tablet/R.Free"
)
DEFAULT_TRAINING_LABELS = (
    DATA_ROOT
    / "1.Training/라벨링데이터/TL/라벨/HW-OCR/4.Validation/T.Tablet/R.Free"
)
DEFAULT_TRAINING_IMAGES = (
    DATA_ROOT
    / "1.Training/원천데이터/TS9/HW-OCR/4.Validation/T.Tablet/R.Free"
)
DEFAULT_OUTPUT_DIR = Path("data/benchmark")

EMAIL_RE = re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b")
PHONE_RE = re.compile(
    r"(?<!\d)(?:01[016789][- ]?\d{3,4}[- ]?\d{4}|"
    r"0\d{1,2}[- ]?\d{3,4}[- ]?\d{4})(?!\d)"
)
RRN_RE = re.compile(r"(?<!\d)\d{6}[- ]?[1-4]\d{6}(?!\d)")
LONG_DIGIT_RE = re.compile(r"(?<!\d)\d{7,}(?!\d)")
URL_RE = re.compile(r"(?i)(?:https?://|www\.)\S+")
PII_PATTERNS = (EMAIL_RE, PHONE_RE, RRN_RE, LONG_DIGIT_RE, URL_RE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-labels", type=Path, default=DEFAULT_VALIDATION_LABELS)
    parser.add_argument("--validation-images", type=Path, default=DEFAULT_VALIDATION_IMAGES)
    parser.add_argument("--training-labels", type=Path, default=DEFAULT_TRAINING_LABELS)
    parser.add_argument("--training-images", type=Path, default=DEFAULT_TRAINING_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-count", type=int, default=2578)
    parser.add_argument("--training-count", type=int, default=1422)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--padding", type=int, default=12)
    return parser.parse_args()


def has_pii(text: str) -> bool:
    return any(pattern.search(text) for pattern in PII_PATTERNS)


def numeric_or_symbol_dominant(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    hangul_count = sum("가" <= char <= "힣" for char in compact)
    # Keep this identical to the previously reported Validation 2,578 rule:
    # Korean characters must make up at least half of the non-space GT.
    return hangul_count / len(compact) < 0.50


def eligible_line(record: LineRecord) -> tuple[bool, str]:
    if len(record.ground_truth.split()) < 2:
        return False, "single_eojeol"
    if numeric_or_symbol_dominant(record.ground_truth):
        return False, "numeric_or_symbol_dominant"
    if has_pii(record.ground_truth):
        return False, "personal_information"
    if record.bbox_id_order_matches_x_order is not True:
        return False, "uncertain_reading_order"
    if record.visual_gt_coverage_pass is not True:
        return False, "visual_gt_coverage_failed"
    return True, "accepted"


def prepare_split(
    split: str,
    labels_root: Path,
    images_root: Path,
    padding: int,
) -> tuple[
    list[LineRecord],
    dict[str, Any],
    dict[str, Path],
    dict[str, Any],
]:
    pages, label_errors = load_label_pages(labels_root)
    image_index, suffixes = index_images(images_root)
    match_report, image_matches, matching_passed = validate_image_matching(
        pages, label_errors, image_index, suffixes
    )
    if not matching_passed:
        raise RuntimeError(f"{split} image/label matching failed")

    by_category, line_diagnostics = extract_all_lines(pages, padding)
    coverage = add_visual_gt_coverage(by_category, image_matches)
    records = by_category.get("T.Tablet/R.Free", [])
    rejection_counts: Counter[str] = Counter()
    candidates: list[LineRecord] = []
    for record in records:
        accepted, reason = eligible_line(record)
        if accepted:
            candidates.append(record)
        else:
            rejection_counts[reason] += 1

    page_meta = {
        page.identifier: {
            "writer_age": page.writer_age,
            "writer_sex": page.writer_sex,
            "media_type": page.media_type,
            "pen_type": page.pen_type,
            "label_path": str(page.json_path.resolve()),
        }
        for page in pages
    }
    diagnostics = {
        "split": split,
        "label_page_count": len(pages),
        "line_count": len(records),
        "eligible_candidate_count": len(candidates),
        "candidate_source_image_count": len({record.identifier for record in candidates}),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "visual_gt_coverage": coverage,
        "line_grouping": line_diagnostics,
        "image_label_matching": match_report,
    }
    return candidates, diagnostics, image_matches, page_meta


def select_training(
    candidates: list[LineRecord],
    count: int,
    validation_ground_truths: set[str],
    seed: int,
) -> tuple[list[LineRecord], dict[str, Any]]:
    by_identifier: dict[str, list[LineRecord]] = defaultdict(list)
    for record in candidates:
        by_identifier[record.identifier].append(record)

    identifiers = sorted(by_identifier)
    random.Random(stable_seed(seed, "training-identifiers")).shuffle(identifiers)
    selected: list[LineRecord] = []
    selected_identifiers: set[str] = set()
    used_gt = set(validation_ground_truths)

    # First pass: one source image and one globally new GT per sample.
    for identifier in identifiers:
        choices = sorted(
            by_identifier[identifier], key=lambda item: (item.line_index, item.ground_truth)
        )
        random.Random(stable_seed(seed, f"training:{identifier}")).shuffle(choices)
        choice = next((item for item in choices if item.ground_truth not in used_gt), None)
        if choice is None:
            continue
        selected.append(choice)
        selected_identifiers.add(identifier)
        used_gt.add(choice.ground_truth)
        if len(selected) == count:
            break

    # Fallback: still at most one line per source, choosing the least frequent GT.
    fallback_count = 0
    if len(selected) < count:
        gt_frequency = Counter(record.ground_truth for record in candidates)
        remaining = [item for item in identifiers if item not in selected_identifiers]
        remaining.sort(
            key=lambda identifier: (
                min(gt_frequency[item.ground_truth] for item in by_identifier[identifier]),
                stable_seed(seed, f"fallback:{identifier}"),
            )
        )
        for identifier in remaining:
            choice = min(
                by_identifier[identifier],
                key=lambda item: (
                    gt_frequency[item.ground_truth],
                    item.ground_truth in used_gt,
                    item.line_index,
                ),
            )
            selected.append(choice)
            selected_identifiers.add(identifier)
            used_gt.add(choice.ground_truth)
            fallback_count += 1
            if len(selected) == count:
                break

    if len(selected) != count:
        raise RuntimeError(f"Could select only {len(selected):,}/{count:,} Training lines")
    selected.sort(key=lambda item: (item.identifier, item.line_index))
    return selected, {
        "selected_count": len(selected),
        "unique_source_image_count": len({item.identifier for item in selected}),
        "maximum_lines_per_source_image": max(
            Counter(item.identifier for item in selected).values(), default=0
        ),
        "unique_ground_truth_count": len({item.ground_truth for item in selected}),
        "ground_truth_overlap_with_validation": sum(
            item.ground_truth in validation_ground_truths for item in selected
        ),
        "fallback_selection_count": fallback_count,
    }


def build_manifest_records(
    validation: list[LineRecord],
    training: list[LineRecord],
    validation_images: dict[str, Path],
    training_images: dict[str, Path],
    validation_meta: dict[str, Any],
    training_meta: dict[str, Any],
    crop_dir: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    combined = [
        ("validation", item, validation_images, validation_meta) for item in validation
    ] + [("training", item, training_images, training_meta) for item in training]
    for sequence, (split, line, images, metadata) in enumerate(combined, start=1):
        image_path = images[line.identifier]
        crop_name = f"{sequence:06d}_{line.identifier}_line{line.line_index:02d}.png"
        meta = metadata[line.identifier]
        records.append(
            {
                "id": f"{sequence:06d}",
                "image_path": str((crop_dir / crop_name).resolve()),
                "ground_truth": line.ground_truth,
                "source": "aihub605",
                "source_split": split,
                "source_image": str(image_path.resolve()),
                "source_identifier": line.identifier,
                "line_index": line.line_index,
                "crop_bbox": line.crop_bbox,
                "token_bboxes": line.token_bboxes,
                "bbox_ids": line.bbox_ids,
                "bbox_count": line.bbox_count,
                "category": line.category,
                "label_path": meta["label_path"],
                "writer_age": meta["writer_age"],
                "writer_sex": meta["writer_sex"],
                "media_type": meta["media_type"],
                "pen_type": meta["pen_type"],
                "uncovered_dark_pixel_ratio": line.uncovered_dark_pixel_ratio,
                "visual_gt_coverage_pass": line.visual_gt_coverage_pass,
            }
        )
    return records


def create_crops(records: list[dict[str, Any]], output_dir: Path, crop_dir: Path) -> None:
    staging = Path(tempfile.mkdtemp(prefix=".eval_handwriting_4000_crops_", dir=output_dir))
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[record["source_image"]].append(record)
    try:
        for source_index, (source_path, source_records) in enumerate(
            sorted(by_source.items()), start=1
        ):
            with Image.open(source_path) as source:
                source.load()
                for record in source_records:
                    crop = source.crop(tuple(record["crop_bbox"]))
                    if crop.width <= 0 or crop.height <= 0:
                        raise RuntimeError(f"Empty crop: {record['id']}")
                    extrema = crop.convert("L").getextrema()
                    if extrema[1] - extrema[0] < 3:
                        raise RuntimeError(f"Blank crop: {record['id']}")
                    crop_name = Path(record["image_path"]).name
                    crop.save(staging / crop_name, format="PNG")
                    record["crop_width"] = crop.width
                    record["crop_height"] = crop.height
            if source_index % 500 == 0:
                print(
                    f"Source images cropped: {source_index:,}/{len(by_source):,}", flush=True
                )
        staging.rename(crop_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def numeric_stats(values: list[int | float]) -> dict[str, int | float]:
    return {
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": min(values),
        "max": max(values),
    }


def validate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    if len(records) != 4000:
        errors.append(f"sample_count={len(records)}")
    split_counts = Counter(item["source_split"] for item in records)
    if split_counts != Counter({"validation": 2578, "training": 1422}):
        errors.append(f"split_counts={dict(split_counts)}")

    for field in ("id", "image_path"):
        values = [item[field] for item in records]
        if len(values) != len(set(values)):
            errors.append(f"duplicate_{field}")
    crop_keys = [
        (item["source_image"], tuple(item["crop_bbox"])) for item in records
    ]
    if len(crop_keys) != len(set(crop_keys)):
        errors.append("duplicate_source_crop")
    if any(len(item["ground_truth"].split()) < 2 for item in records):
        errors.append("single_eojeol_present")
    if any(item["visual_gt_coverage_pass"] is not True for item in records):
        errors.append("coverage_failure_present")
    training_sources = [
        item["source_image"] for item in records if item["source_split"] == "training"
    ]
    if len(training_sources) != len(set(training_sources)):
        errors.append("multiple_training_lines_per_source")

    missing = 0
    unreadable = 0
    size_mismatch = 0
    for item in records:
        path = Path(item["image_path"])
        if not path.is_file():
            missing += 1
            continue
        try:
            with Image.open(path) as image:
                image.load()
                if image.size != (item["crop_width"], item["crop_height"]):
                    size_mismatch += 1
        except (OSError, UnidentifiedImageError):
            unreadable += 1
    if missing:
        errors.append(f"missing_crops={missing}")
    if unreadable:
        errors.append(f"unreadable_crops={unreadable}")
    if size_mismatch:
        errors.append(f"size_mismatch={size_mismatch}")
    return {
        "passed": not errors,
        "errors": errors,
        "sample_count": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "missing_crop_count": missing,
        "unreadable_crop_count": unreadable,
        "size_mismatch_count": size_mismatch,
    }


def build_stats(
    records: list[dict[str, Any]],
    validation_diagnostics: dict[str, Any],
    training_diagnostics: dict[str, Any],
    training_selection: dict[str, Any],
    validation: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    gt_counts = Counter(item["ground_truth"] for item in records)
    source_counts = Counter(item["source_image"] for item in records)
    split_counts = Counter(item["source_split"] for item in records)
    lengths = [len(item["ground_truth"]) for item in records]
    compact_lengths = [len(re.sub(r"\s+", "", item["ground_truth"])) for item in records]
    eojeol = [len(item["ground_truth"].split()) for item in records]
    widths = [item["crop_width"] for item in records]
    heights = [item["crop_height"] for item in records]
    return {
        "sample_count": len(records),
        "seed": seed,
        "split_distribution": dict(sorted(split_counts.items())),
        "category_distribution": dict(
            sorted(Counter(item["category"] for item in records).items())
        ),
        "ground_truth_length": numeric_stats(lengths),
        "ground_truth_nonspace_length": numeric_stats(compact_lengths),
        "ground_truth_eojeol_count": numeric_stats(eojeol),
        "unique_ground_truth_count": len(gt_counts),
        "duplicate_ground_truth_occurrence_count": len(records) - len(gt_counts),
        "duplicated_ground_truth_value_count": sum(count > 1 for count in gt_counts.values()),
        "source_image_count": len(source_counts),
        "lines_per_source_image": numeric_stats(list(source_counts.values())),
        "maximum_lines_per_source_image_by_split": {
            split: max(
                Counter(
                    item["source_image"]
                    for item in records
                    if item["source_split"] == split
                ).values(),
                default=0,
            )
            for split in sorted(split_counts)
        },
        "writer_count": None,
        "writer_count_available": False,
        "writer_note": "라벨에 writer ID가 없어 고유 작성자 수는 검증할 수 없음.",
        "writer_age_distribution": dict(
            sorted(Counter(str(item["writer_age"]) for item in records).items())
        ),
        "writer_sex_distribution": dict(
            sorted(Counter(str(item["writer_sex"]) for item in records).items())
        ),
        "crop_width": numeric_stats(widths),
        "crop_height": numeric_stats(heights),
        "coverage_uncovered_dark_pixel_ratio": numeric_stats(
            [item["uncovered_dark_pixel_ratio"] for item in records]
        ),
        "selection_policy": {
            "included_category": "T.Tablet/R.Free",
            "minimum_eojeol_count": 2,
            "single_word_excluded": True,
            "numeric_or_symbol_dominant_excluded": True,
            "minimum_hangul_ratio": 0.50,
            "personal_information_patterns_excluded": True,
            "bbox_id_order_must_match_x_order": True,
            "maximum_uncovered_dark_pixel_ratio": 0.05,
            "validation_candidates_all_included": True,
            "training_lines_per_source_image_maximum": 1,
            "training_ground_truth_unique_against_validation_when_possible": True,
        },
        "validation_candidate_diagnostics": validation_diagnostics,
        "training_candidate_diagnostics": training_diagnostics,
        "training_selection": training_selection,
        "validation": validation,
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def preview_html(records: list[dict[str, Any]], output_dir: Path, seed: int) -> str:
    selected = random.Random(seed).sample(records, 20)
    cards = []
    for item in selected:
        image_path = Path(item["image_path"]).relative_to(output_dir.resolve()).as_posix()
        cards.append(
            '<article class="card">'
            f'<img src="{html.escape(image_path)}" loading="lazy" '
            f'alt="{html.escape(item["ground_truth"])}">'
            f'<h2>{html.escape(item["ground_truth"])}</h2>'
            f'<dl><dt>ID / split</dt><dd>{item["id"]} / {item["source_split"]}</dd>'
            f'<dt>source</dt><dd>{html.escape(item["source_identifier"])}</dd>'
            f'<dt>line / bbox count</dt><dd>{item["line_index"]} / {item["bbox_count"]}</dd>'
            f'<dt>coverage</dt><dd>{item["uncovered_dark_pixel_ratio"]}</dd>'
            f'<dt>crop bbox</dt><dd>{html.escape(json.dumps(item["crop_bbox"]))}</dd>'
            "</dl></article>"
        )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Hub 605 handwriting evaluation preview</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f3f5f7;color:#17202a}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}}
.card{{background:#fff;border:1px solid #d8dde3;border-radius:10px;padding:14px}}
img{{display:block;max-width:100%;max-height:230px;margin:auto;background:#eee}}
h2{{font-size:1.05rem}}dl{{font-size:.82rem;overflow-wrap:anywhere}}
dt{{font-weight:700;margin-top:6px}}dd{{margin-left:0;color:#445}}
</style></head><body>
<h1>AI Hub 605 손글씨 평가셋 Preview — random 20</h1>
<p>전체 4,000개, seed={seed}</p><main class="grid">{''.join(cards)}</main>
</body></html>\n"""


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    crop_dir = output_dir / "eval_handwriting_4000_crops"
    jsonl_path = output_dir / "eval_handwriting_4000.jsonl"
    stats_path = output_dir / "eval_handwriting_4000_stats.json"
    preview_path = output_dir / "sample_preview.html"
    conflicts = [
        path for path in (crop_dir, jsonl_path, stats_path, preview_path) if path.exists()
    ]
    if conflicts:
        raise SystemExit(
            "Refusing to overwrite existing artifacts: "
            + ", ".join(str(path) for path in conflicts)
        )
    for path in (
        args.validation_labels,
        args.validation_images,
        args.training_labels,
        args.training_images,
    ):
        if not path.resolve().is_dir():
            raise SystemExit(f"Required directory does not exist: {path.resolve()}")

    print("Preparing Validation candidates...", flush=True)
    validation_candidates, validation_diag, validation_images, validation_meta = (
        prepare_split(
            "validation",
            args.validation_labels.resolve(),
            args.validation_images.resolve(),
            args.padding,
        )
    )
    if len(validation_candidates) != args.validation_count:
        raise RuntimeError(
            f"Validation candidates={len(validation_candidates):,}, "
            f"expected={args.validation_count:,}"
        )
    validation_candidates.sort(key=lambda item: (item.identifier, item.line_index))

    print("Preparing Training candidates...", flush=True)
    training_candidates, training_diag, training_images, training_meta = prepare_split(
        "training",
        args.training_labels.resolve(),
        args.training_images.resolve(),
        args.padding,
    )
    selected_training, training_selection = select_training(
        training_candidates,
        args.training_count,
        {item.ground_truth for item in validation_candidates},
        args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    records = build_manifest_records(
        validation_candidates,
        selected_training,
        validation_images,
        training_images,
        validation_meta,
        training_meta,
        crop_dir,
    )
    print("Creating line crops...", flush=True)
    create_crops(records, output_dir, crop_dir)
    validation = validate_records(records)
    if not validation["passed"]:
        raise RuntimeError("Final validation failed: " + "; ".join(validation["errors"]))
    stats = build_stats(
        records,
        validation_diag,
        training_diag,
        training_selection,
        validation,
        args.seed,
    )
    write_jsonl(jsonl_path, records)
    atomic_write_json(stats_path, stats)
    atomic_write_text(preview_path, preview_html(records, output_dir, args.seed))
    print(f"Completed: {jsonl_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
