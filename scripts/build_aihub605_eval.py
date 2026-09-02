#!/usr/bin/env python3
"""Validate AI Hub 605 images and build the 2,000-sample real-handwriting split."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import random
import re
import shutil
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


DEFAULT_LABELS_ROOT = Path(
    "data/053.대용량_손글씨_OCR_데이터/01.데이터/2.Validation/"
    "라벨링데이터/VL/라벨/HW-OCR/4.Validation"
)
DEFAULT_IMAGES_ROOT = Path(
    "data/053.대용량_손글씨_OCR_데이터/01.데이터/2.Validation/원천데이터"
)
DEFAULT_OUTPUT_DIR = Path("data/benchmark")
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
HANGUL_RE = re.compile(r"[가-힣]")
DIGIT_RE = re.compile(r"\d")
LONG_DIGIT_RE = re.compile(r"\d{7,}")
EMAIL_RE = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")
WHITESPACE_RE = re.compile(r"\s")


@dataclass(frozen=True)
class CropCandidate:
    identifier: str
    json_path: Path
    category: str
    bbox_id: Any
    text: str
    bbox: tuple[int, int, int, int]
    writer_age: Any
    writer_sex: Any
    media_type: Any
    pen_type: Any


@dataclass
class LabelPage:
    identifier: str
    json_path: Path
    category: str
    image_type: str
    width: int
    height: int
    writer_age: Any
    writer_sex: Any
    media_type: Any
    pen_type: Any
    bboxes: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS_ROOT)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--match-only",
        action="store_true",
        help="Write the image/label matching report and stop before crop selection.",
    )
    return parser.parse_args()


def category_from_path(label_path: Any, json_path: Path) -> str:
    if isinstance(label_path, str):
        parts = Path(label_path).parts
        if "4.Validation" in parts:
            index = parts.index("4.Validation")
            if parts[index + 1 :]:
                return "/".join(parts[index + 1 :])
    parts = json_path.parent.parts
    if "4.Validation" in parts:
        index = parts.index("4.Validation")
        return "/".join(parts[index + 1 :])
    return "<UNKNOWN>"


def load_label_pages(labels_root: Path) -> tuple[list[LabelPage], list[dict[str, str]]]:
    pages: list[LabelPage] = []
    errors: list[dict[str, str]] = []
    json_paths = sorted(labels_root.rglob("*.json"))
    for json_path in json_paths:
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            dataset = payload["Dataset"]
            images = payload["Images"]
            bboxes = payload["bbox"]
            if not isinstance(dataset, dict) or not isinstance(images, dict) or not isinstance(bboxes, list):
                raise TypeError("Dataset/Images/bbox has an unexpected type")
            identifier = images["identifier"]
            width = images["width"]
            height = images["height"]
            image_type = images["type"]
            if not isinstance(identifier, str) or not identifier:
                raise TypeError("Images.identifier is not a non-empty string")
            if not isinstance(width, int) or not isinstance(height, int):
                raise TypeError("Images width/height is not integer")
            pages.append(
                LabelPage(
                    identifier=identifier,
                    json_path=json_path.resolve(),
                    category=category_from_path(dataset.get("label_path"), json_path),
                    image_type=str(image_type).lower(),
                    width=width,
                    height=height,
                    writer_age=images.get("writer_age"),
                    writer_sex=images.get("writer_sex"),
                    media_type=images.get("media_type"),
                    pen_type=images.get("pen_type"),
                    bboxes=bboxes,
                )
            )
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            errors.append({"json_path": str(json_path), "error": str(exc)})
    return pages, errors


def index_images(images_root: Path) -> tuple[dict[str, list[Path]], Counter[str]]:
    by_stem: dict[str, list[Path]] = defaultdict(list)
    suffixes: Counter[str] = Counter()
    for path in sorted(images_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
            resolved = path.resolve()
            by_stem[path.stem].append(resolved)
            suffixes[path.suffix.lower()] += 1
    return dict(by_stem), suffixes


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
    temporary.replace(path)


def validate_image_matching(
    pages: list[LabelPage],
    label_errors: list[dict[str, str]],
    image_index: dict[str, list[Path]],
    image_suffixes: Counter[str],
) -> tuple[dict[str, Any], dict[str, Path], bool]:
    label_identifier_counts = Counter(page.identifier for page in pages)
    duplicate_label_identifiers = sorted(
        identifier for identifier, count in label_identifier_counts.items() if count > 1
    )
    duplicate_image_stems = {
        stem: [str(path) for path in paths]
        for stem, paths in sorted(image_index.items())
        if len(paths) > 1
    }
    label_identifiers = set(label_identifier_counts)
    image_stems = set(image_index)
    missing_identifiers = sorted(label_identifiers - image_stems)
    orphan_image_stems = sorted(image_stems - label_identifiers)
    unique_matches = {
        identifier: paths[0]
        for identifier, paths in image_index.items()
        if identifier in label_identifiers and len(paths) == 1
    }

    unreadable: list[dict[str, str]] = []
    dimension_mismatches: list[dict[str, Any]] = []
    extension_mismatches: list[dict[str, str]] = []
    format_distribution: Counter[str] = Counter()
    page_by_identifier = {page.identifier: page for page in pages}

    for index, identifier in enumerate(sorted(unique_matches), start=1):
        path = unique_matches[identifier]
        page = page_by_identifier[identifier]
        actual_extension = path.suffix.lower().lstrip(".")
        if actual_extension != page.image_type.lower().lstrip("."):
            extension_mismatches.append(
                {
                    "identifier": identifier,
                    "expected": page.image_type,
                    "actual": actual_extension,
                    "image_path": str(path),
                }
            )
        try:
            with Image.open(path) as image:
                actual_size = image.size
                actual_format = (image.format or "<UNKNOWN>").lower()
                format_distribution[actual_format] += 1
            if actual_size != (page.width, page.height):
                dimension_mismatches.append(
                    {
                        "identifier": identifier,
                        "expected": [page.width, page.height],
                        "actual": list(actual_size),
                        "image_path": str(path),
                    }
                )
        except (OSError, UnidentifiedImageError) as exc:
            unreadable.append(
                {"identifier": identifier, "image_path": str(path), "error": str(exc)}
            )
        if index % 1000 == 0:
            print(f"Matched image headers checked: {index:,}/{len(unique_matches):,}", flush=True)

    rfree_identifiers = {page.identifier for page in pages if page.category.endswith("/R.Free")}
    rfree_missing = sorted(rfree_identifiers - set(unique_matches))
    report = {
        "label_json_count": len(pages) + len(label_errors),
        "parsed_label_count": len(pages),
        "label_error_count": len(label_errors),
        "label_errors": label_errors,
        "unique_label_identifier_count": len(label_identifiers),
        "duplicate_label_identifier_count": len(duplicate_label_identifiers),
        "duplicate_label_identifiers": duplicate_label_identifiers,
        "indexed_image_count": sum(len(paths) for paths in image_index.values()),
        "unique_image_stem_count": len(image_stems),
        "image_suffix_distribution": dict(sorted(image_suffixes.items())),
        "unique_match_count": len(unique_matches),
        "missing_identifier_count": len(missing_identifiers),
        "missing_identifiers": missing_identifiers,
        "orphan_image_stem_count": len(orphan_image_stems),
        "orphan_image_stems": orphan_image_stems,
        "duplicate_image_stem_count": len(duplicate_image_stems),
        "duplicate_image_stems": duplicate_image_stems,
        "unreadable_image_count": len(unreadable),
        "unreadable_images": unreadable,
        "dimension_mismatch_count": len(dimension_mismatches),
        "dimension_mismatches": dimension_mismatches,
        "extension_mismatch_count": len(extension_mismatches),
        "extension_mismatches": extension_mismatches,
        "image_format_distribution": dict(sorted(format_distribution.items())),
        "rfree_label_identifier_count": len(rfree_identifiers),
        "rfree_missing_identifier_count": len(rfree_missing),
        "rfree_missing_identifiers": rfree_missing,
    }
    blocking_keys = (
        "label_error_count",
        "duplicate_label_identifier_count",
        "missing_identifier_count",
        "orphan_image_stem_count",
        "duplicate_image_stem_count",
        "unreadable_image_count",
        "dimension_mismatch_count",
        "extension_mismatch_count",
        "rfree_missing_identifier_count",
    )
    passed = all(report[key] == 0 for key in blocking_keys)
    report["passed"] = passed
    return report, unique_matches, passed


def normalize_bbox(raw: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int] | None:
    x_values = raw.get("x")
    y_values = raw.get("y")
    if (
        not isinstance(x_values, list)
        or not isinstance(y_values, list)
        or len(x_values) < 2
        or len(y_values) < 2
    ):
        return None
    try:
        xs = [float(value) for value in x_values]
        ys = [float(value) for value in y_values]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in xs + ys):
        return None
    left = math.floor(min(xs))
    top = math.floor(min(ys))
    right = math.ceil(max(xs))
    bottom = math.ceil(max(ys))
    if left < 0 or top < 0 or right > width or bottom > height:
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def eligible_text(value: Any) -> tuple[bool, str]:
    if not isinstance(value, str):
        return False, "non_string"
    text = value.strip()
    compact = WHITESPACE_RE.sub("", text)
    if not compact:
        return False, "empty"
    if len(compact) < 2:
        return False, "too_short"
    if not HANGUL_RE.search(compact):
        return False, "no_hangul"
    if EMAIL_RE.search(text):
        return False, "email_like"
    if LONG_DIGIT_RE.search(compact):
        return False, "long_digit_sequence"
    digit_ratio = len(DIGIT_RE.findall(compact)) / len(compact)
    if digit_ratio > 0.20:
        return False, "digit_ratio"
    return True, text


def collect_candidates(
    pages: list[LabelPage], image_matches: dict[str, Path]
) -> tuple[dict[str, dict[str, list[CropCandidate]]], Counter[str]]:
    by_category: dict[str, dict[str, list[CropCandidate]]] = defaultdict(lambda: defaultdict(list))
    rejection_counts: Counter[str] = Counter()
    for page in pages:
        if not page.category.endswith("/R.Free"):
            rejection_counts["excluded_o_form"] += len(page.bboxes)
            continue
        if page.identifier not in image_matches:
            rejection_counts["missing_source_image"] += len(page.bboxes)
            continue
        for raw in page.bboxes:
            if not isinstance(raw, dict):
                rejection_counts["invalid_bbox_object"] += 1
                continue
            accepted, text_or_reason = eligible_text(raw.get("data"))
            if not accepted:
                rejection_counts[text_or_reason] += 1
                continue
            bbox = normalize_bbox(raw, page.width, page.height)
            if bbox is None:
                rejection_counts["invalid_bbox_coordinates"] += 1
                continue
            by_category[page.category][page.identifier].append(
                CropCandidate(
                    identifier=page.identifier,
                    json_path=page.json_path,
                    category=page.category,
                    bbox_id=raw.get("id"),
                    text=text_or_reason,
                    bbox=bbox,
                    writer_age=page.writer_age,
                    writer_sex=page.writer_sex,
                    media_type=page.media_type,
                    pen_type=page.pen_type,
                )
            )
    return {category: dict(images) for category, images in by_category.items()}, rejection_counts


def proportional_quotas(image_counts: dict[str, int], sample_count: int) -> dict[str, int]:
    total = sum(image_counts.values())
    if sample_count > total:
        raise ValueError(f"Requested {sample_count} samples but only {total} eligible source images exist")
    exact = {category: sample_count * count / total for category, count in image_counts.items()}
    quotas = {category: math.floor(value) for category, value in exact.items()}
    remainder = sample_count - sum(quotas.values())
    order = sorted(image_counts, key=lambda category: (-(exact[category] - quotas[category]), category))
    for category in order[:remainder]:
        quotas[category] += 1
    return quotas


def stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def crop_is_nonblank(image: Image.Image) -> bool:
    grayscale = image.convert("L")
    minimum, maximum = grayscale.getextrema()
    return maximum - minimum >= 3


def build_crops(
    candidates_by_category: dict[str, dict[str, list[CropCandidate]]],
    image_matches: dict[str, Path],
    output_dir: Path,
    sample_count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], Counter[str], dict[str, int]]:
    final_crop_dir = output_dir / "aihub_real_crops"
    final_jsonl = output_dir / "eval_aihub_real_2000.jsonl"
    final_stats = output_dir / "eval_aihub_real_2000_stats.json"
    final_preview = output_dir / "sample_preview_aihub_real.html"
    conflicts = [path for path in (final_crop_dir, final_jsonl, final_stats, final_preview) if path.exists()]
    if conflicts:
        raise RuntimeError(
            "Refusing to overwrite existing benchmark artifacts: "
            + ", ".join(str(path) for path in conflicts)
        )

    image_counts = {
        category: len(images) for category, images in sorted(candidates_by_category.items())
    }
    quotas = proportional_quotas(image_counts, sample_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=".aihub_real_crops_", dir=output_dir))
    used_ground_truth: set[str] = set()
    used_sources: set[str] = set()
    records: list[dict[str, Any]] = []
    runtime_rejections: Counter[str] = Counter()

    try:
        for category in sorted(quotas):
            quota = quotas[category]
            identifiers = sorted(candidates_by_category[category])
            random.Random(stable_seed(seed, category)).shuffle(identifiers)
            category_count = 0
            for identifier in identifiers:
                if category_count >= quota:
                    break
                if identifier in used_sources:
                    runtime_rejections["duplicate_source"] += 1
                    continue
                choices = list(candidates_by_category[category][identifier])
                random.Random(stable_seed(seed, identifier)).shuffle(choices)
                candidate = next(
                    (choice for choice in choices if choice.text not in used_ground_truth), None
                )
                if candidate is None:
                    runtime_rejections["no_unique_ground_truth_for_source"] += 1
                    continue
                source_path = image_matches[identifier]
                try:
                    with Image.open(source_path) as source_image:
                        source_image.load()
                        crop = source_image.crop(candidate.bbox)
                        if crop.width <= 0 or crop.height <= 0:
                            runtime_rejections["empty_crop"] += 1
                            continue
                        if not crop_is_nonblank(crop):
                            runtime_rejections["blank_crop"] += 1
                            continue
                        sequence = len(records) + 1
                        bbox_id = str(candidate.bbox_id).zfill(3)
                        crop_name = f"{sequence:06d}_{identifier}_b{bbox_id}.png"
                        crop_path = staging_dir / crop_name
                        crop.save(crop_path, format="PNG")
                    with Image.open(crop_path) as verification_image:
                        verification_image.load()
                        verified_size = verification_image.size
                except (OSError, UnidentifiedImageError, ValueError) as exc:
                    runtime_rejections[f"crop_error:{type(exc).__name__}"] += 1
                    continue

                record = {
                    "id": f"{sequence:06d}",
                    "image_path": str((final_crop_dir / crop_name).resolve()),
                    "ground_truth": candidate.text,
                    "source": "aihub_real",
                    "source_image": str(source_path.resolve()),
                    "bbox": list(candidate.bbox),
                    "bbox_id": candidate.bbox_id,
                    "category": category,
                    "label_path": str(candidate.json_path),
                    "writer_age": candidate.writer_age,
                    "writer_sex": candidate.writer_sex,
                    "media_type": candidate.media_type,
                    "pen_type": candidate.pen_type,
                    "crop_width": verified_size[0],
                    "crop_height": verified_size[1],
                }
                records.append(record)
                used_sources.add(identifier)
                used_ground_truth.add(candidate.text)
                category_count += 1
                if len(records) % 250 == 0:
                    print(f"Crops created: {len(records):,}/{sample_count:,}", flush=True)

            if category_count != quota:
                raise RuntimeError(
                    f"Could only create {category_count} of {quota} required crops for {category}"
                )
        if len(records) != sample_count:
            raise RuntimeError(f"Built {len(records)} records, expected {sample_count}")
        staging_dir.rename(final_crop_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return records, runtime_rejections, quotas


def validate_records(records: list[dict[str, Any]], expected_count: int) -> dict[str, Any]:
    errors: list[str] = []
    if len(records) != expected_count:
        errors.append(f"sample_count={len(records)}, expected={expected_count}")
    checks = {
        "id": [record["id"] for record in records],
        "image_path": [record["image_path"] for record in records],
        "source_image": [record["source_image"] for record in records],
        "ground_truth": [record["ground_truth"] for record in records],
    }
    for field_name, values in checks.items():
        if len(values) != len(set(values)):
            errors.append(f"duplicate {field_name}")
    invalid_source_count = sum(record.get("source") != "aihub_real" for record in records)
    invalid_category_count = sum(
        not str(record.get("category", "")).endswith("/R.Free") for record in records
    )
    empty_ground_truth_count = sum(not str(record.get("ground_truth", "")).strip() for record in records)
    if invalid_source_count:
        errors.append(f"invalid source field: {invalid_source_count}")
    if invalid_category_count:
        errors.append(f"non-R.Free records: {invalid_category_count}")
    if empty_ground_truth_count:
        errors.append(f"empty ground truth: {empty_ground_truth_count}")

    missing_path_count = 0
    unreadable_crop_count = 0
    size_mismatch_count = 0
    for index, record in enumerate(records, start=1):
        image_path = Path(record["image_path"])
        source_path = Path(record["source_image"])
        if not image_path.is_file() or not source_path.is_file():
            missing_path_count += 1
            continue
        try:
            with Image.open(image_path) as image:
                image.load()
                actual_size = image.size
            if actual_size != (record["crop_width"], record["crop_height"]):
                size_mismatch_count += 1
        except (OSError, UnidentifiedImageError):
            unreadable_crop_count += 1
        if index % 500 == 0:
            print(f"Final crops validated: {index:,}/{len(records):,}", flush=True)
    if missing_path_count:
        errors.append(f"missing paths: {missing_path_count}")
    if unreadable_crop_count:
        errors.append(f"unreadable crops: {unreadable_crop_count}")
    if size_mismatch_count:
        errors.append(f"crop size mismatches: {size_mismatch_count}")
    return {
        "passed": not errors,
        "errors": errors,
        "sample_count": len(records),
        "missing_path_count": missing_path_count,
        "unreadable_crop_count": unreadable_crop_count,
        "size_mismatch_count": size_mismatch_count,
        "duplicate_id_count": len(checks["id"]) - len(set(checks["id"])),
        "duplicate_image_path_count": len(checks["image_path"]) - len(set(checks["image_path"])),
        "duplicate_source_image_count": len(checks["source_image"]) - len(set(checks["source_image"])),
        "duplicate_ground_truth_count": len(checks["ground_truth"]) - len(set(checks["ground_truth"])),
        "invalid_source_count": invalid_source_count,
        "invalid_category_count": invalid_category_count,
        "empty_ground_truth_count": empty_ground_truth_count,
    }


def numeric_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": min(values),
        "max": max(values),
    }


def create_stats(
    records: list[dict[str, Any]],
    validation: dict[str, Any],
    match_report: dict[str, Any],
    label_rejections: Counter[str],
    runtime_rejections: Counter[str],
    quotas: dict[str, int],
    seed: int,
) -> dict[str, Any]:
    lengths = [len(record["ground_truth"]) for record in records]
    widths = [record["crop_width"] for record in records]
    heights = [record["crop_height"] for record in records]
    areas = [width * height for width, height in zip(widths, heights)]
    category_counts = Counter(record["category"] for record in records)
    source_counts = Counter(record["source"] for record in records)
    writer_ages = Counter(str(record["writer_age"]) for record in records)
    writer_sexes = Counter(str(record["writer_sex"]) for record in records)
    image_extensions = Counter(Path(record["image_path"]).suffix.lower() for record in records)
    source_extensions = Counter(Path(record["source_image"]).suffix.lower() for record in records)
    return {
        "sample_count": len(records),
        "seed": seed,
        "source_distribution": dict(sorted(source_counts.items())),
        "category_distribution": dict(sorted(category_counts.items())),
        "category_quotas": quotas,
        "ground_truth_length": numeric_stats(lengths),
        "source_image_count": len({record["source_image"] for record in records}),
        "writer_count": None,
        "writer_count_available": False,
        "writer_note": (
            "라벨에 writer ID가 없고 identifier는 이미지 ID이므로 작성자 수와 작성자 중복을 검증할 수 없다."
        ),
        "writer_age_distribution": dict(sorted(writer_ages.items())),
        "writer_sex_distribution": dict(sorted(writer_sexes.items())),
        "duplicate_ground_truth_count": validation["duplicate_ground_truth_count"],
        "crop_image_extension_distribution": dict(sorted(image_extensions.items())),
        "source_image_extension_distribution": dict(sorted(source_extensions.items())),
        "bbox_crop_width": numeric_stats(widths),
        "bbox_crop_height": numeric_stats(heights),
        "bbox_crop_area": numeric_stats(areas),
        "label_filter_rejections": dict(sorted(label_rejections.items())),
        "runtime_rejections": dict(sorted(runtime_rejections.items())),
        "selection_policy": {
            "included_categories": ["P.Paper/R.Free", "T.Tablet/R.Free"],
            "excluded_categories": ["P.Paper/O.Form", "T.Tablet/O.Form"],
            "one_bbox_per_source_image": True,
            "unique_ground_truth": True,
            "minimum_nonspace_length": 2,
            "requires_hangul": True,
            "maximum_digit_ratio": 0.20,
            "email_and_long_digit_sequences_excluded": True,
            "category_quota_method": "eligible R.Free source-image proportion, largest remainder",
        },
        "image_label_matching": match_report,
        "validation": validation,
        "future_combination_note": (
            "향후 synthetic_normal 2,000개와 결합할 때 해당 합성 샘플은 학습 미사용 hold-out임을 "
            "별도 manifest로 검증하고 source 필드로 결과를 분리 집계해야 한다."
        ),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def create_preview(records: list[dict[str, Any]], output_dir: Path, seed: int) -> str:
    rng = random.Random(seed)
    selected = rng.sample(records, min(20, len(records)))
    cards = []
    for record in selected:
        relative_image = Path(record["image_path"]).relative_to(output_dir.resolve())
        cards.append(
            "<article class=\"card\">"
            f"<img src=\"{html.escape(relative_image.as_posix())}\" loading=\"lazy\" "
            f"alt=\"{html.escape(record['ground_truth'])}\">"
            f"<h2>{html.escape(record['ground_truth'])}</h2>"
            f"<dl><dt>ID</dt><dd>{html.escape(record['id'])}</dd>"
            f"<dt>source</dt><dd>{html.escape(record['source'])}</dd>"
            f"<dt>category</dt><dd>{html.escape(record['category'])}</dd>"
            f"<dt>source_image</dt><dd>{html.escape(record['source_image'])}</dd>"
            f"<dt>bbox</dt><dd>{html.escape(json.dumps(record['bbox']))}</dd></dl>"
            "</article>"
        )
    return """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Hub real handwriting preview</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#f4f5f7;color:#17202a}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px}
.card{background:white;border:1px solid #d8dde3;border-radius:10px;padding:14px;box-shadow:0 2px 8px #0001}
img{display:block;max-width:100%;max-height:180px;margin:auto;background:#eee;image-rendering:auto}
h2{font-size:1.15rem;margin:12px 0}.card dl{font-size:.82rem;overflow-wrap:anywhere}
dt{font-weight:700;margin-top:6px}dd{margin-left:0;color:#445}
</style></head><body>
<h1>AI Hub 605 real handwriting — random 20</h1>
<p>source=<code>aihub_real</code>, seed=42</p>
<main class="grid">""" + "\n".join(cards) + "</main></body></html>\n"


def main() -> int:
    args = parse_args()
    if args.sample_count <= 0:
        raise SystemExit("--sample-count must be positive")
    labels_root = args.labels_root.resolve()
    images_root = args.images_root.resolve()
    output_dir = args.output_dir.resolve()
    if not labels_root.is_dir():
        raise SystemExit(f"Labels root does not exist: {labels_root}")
    if not images_root.is_dir():
        raise SystemExit(f"Images root does not exist: {images_root}")

    print("Loading labels...", flush=True)
    pages, label_errors = load_label_pages(labels_root)
    print(f"Labels loaded: {len(pages):,}; errors: {len(label_errors):,}", flush=True)
    print("Indexing source images...", flush=True)
    image_index, image_suffixes = index_images(images_root)
    print(f"Images indexed: {sum(len(paths) for paths in image_index.values()):,}", flush=True)
    match_report, image_matches, match_passed = validate_image_matching(
        pages, label_errors, image_index, image_suffixes
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    match_report_path = output_dir / "aihub605_image_match_report.json"
    atomic_write_json(match_report_path, match_report)
    print(f"Image matching passed={match_passed}; report={match_report_path}", flush=True)
    if not match_passed:
        raise SystemExit("Image/label matching failed; crop generation was not started")
    if args.match_only:
        return 0

    candidates_by_category, label_rejections = collect_candidates(pages, image_matches)
    candidate_image_counts = {
        category: len(images) for category, images in sorted(candidates_by_category.items())
    }
    print(f"Eligible R.Free source images: {candidate_image_counts}", flush=True)
    records, runtime_rejections, quotas = build_crops(
        candidates_by_category,
        image_matches,
        output_dir,
        args.sample_count,
        args.seed,
    )
    validation = validate_records(records, args.sample_count)
    if not validation["passed"]:
        raise SystemExit("Final validation failed: " + "; ".join(validation["errors"]))

    jsonl_path = output_dir / "eval_aihub_real_2000.jsonl"
    stats_path = output_dir / "eval_aihub_real_2000_stats.json"
    preview_path = output_dir / "sample_preview_aihub_real.html"
    stats = create_stats(
        records,
        validation,
        match_report,
        label_rejections,
        runtime_rejections,
        quotas,
        args.seed,
    )
    write_jsonl(jsonl_path, records)
    atomic_write_json(stats_path, stats)
    atomic_write_text(preview_path, create_preview(records, output_dir, args.seed))
    print(f"JSONL: {jsonl_path}", flush=True)
    print(f"Stats: {stats_path}", flush=True)
    print(f"Preview: {preview_path}", flush=True)
    print("Build completed and validated", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
