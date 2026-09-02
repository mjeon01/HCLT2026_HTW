#!/usr/bin/env python3
"""Build the final separated handwriting evaluation manifests.

Outputs:
  * 2,000 real AI Hub line crops selected from the preserved 4,000 manifest
  * 2,000 clean synthetic hold-out samples
  * every valid error synthetic hold-out sample, up to 2,000

The script never modifies or removes the existing 4,000-sample AI Hub set.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import mimetypes
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from kiwipiepy import Kiwi
from PIL import Image, UnidentifiedImageError

from build_aihub605_eval import load_label_pages
from preview_aihub605_lines import LineRecord, extract_all_lines


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "data/benchmark"
AIHUB_4000 = BENCHMARK_DIR / "eval_handwriting_4000.jsonl"
SYNTHETIC_ROOT = ROOT / "data/generated-handwriting-project-b"
SYNTHETIC_METADATA = SYNTHETIC_ROOT / "metadata.jsonl"
SYNTHETIC_IMAGES = SYNTHETIC_ROOT / "images"
AIHUB_DATA_ROOT = ROOT / "data/053.대용량_손글씨_OCR_데이터/01.데이터"
AIHUB_LABEL_ROOTS = {
    "validation": AIHUB_DATA_ROOT
    / "2.Validation/라벨링데이터/VL/라벨/HW-OCR/4.Validation/T.Tablet/R.Free",
    "training": AIHUB_DATA_ROOT
    / "1.Training/라벨링데이터/TL/라벨/HW-OCR/4.Validation/T.Tablet/R.Free",
}

SEED = 42
SENTENCE_TERMINAL_RE = re.compile(r"[.!?。！？]\s*$")
EMAIL_RE = re.compile(r"(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b")
PHONE_RE = re.compile(
    r"(?<!\d)(?:01[016789][- ]?\d{3,4}[- ]?\d{4}|"
    r"0\d{1,2}[- ]?\d{3,4}[- ]?\d{4})(?!\d)"
)
RRN_RE = re.compile(r"(?<!\d)\d{6}[- ]?[1-4]\d{6}(?!\d)")
LONG_DIGIT_RE = re.compile(r"(?<!\d)\d{7,}(?!\d)")
URL_RE = re.compile(r"(?i)(?:https?://|www\.)\S+")
PII_PATTERNS = (EMAIL_RE, PHONE_RE, RRN_RE, LONG_DIGIT_RE, URL_RE)

PARTICLE_AREAS = {"FNP", "FOP", "FAP", "FCP", "FGP", "FJC", "FQP", "FXP"}
ERROR_FAMILY_LABELS = {
    "5B": "맞춤법·음운 오류",
    "5C": "활용·형태 오류",
    "5D": "문장 구조·높임/시제 오류",
    "5E": "담화·문체 오류",
    "5F": "어휘 선택 오류",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(value)
    return records


def atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def numeric_stats(values: list[int | float]) -> dict[str, int | float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": min(values),
        "max": max(values),
    }


def percent(count: int, denominator: int) -> float:
    return round(100.0 * count / denominator, 4) if denominator else 0.0


def stable_digest(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def has_pii(text: str) -> bool:
    return any(pattern.search(text) for pattern in PII_PATTERNS)


def numeric_or_symbol_dominant(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    hangul = sum("가" <= char <= "힣" for char in compact)
    return hangul / len(compact) < 0.50


def linked_eojeol_count(text: str, kiwi: Kiwi) -> int:
    """Count eojeols that carry a syntactic link rather than dictionary-list form.

    A bare dictionary form ending in -다 is deliberately not counted.  This is
    the main separator between AI Hub's unrelated word-list pages and pages made
    of connected clauses/sentences.
    """

    count = 0
    for eojeol in text.split():
        tags = [token.tag for token in kiwi.tokenize(eojeol)]
        linked = any(tag.startswith("J") for tag in tags)
        linked = linked or any(tag in {"EP", "ETM", "ETN"} for tag in tags)
        cleaned = eojeol.rstrip(".,!?。！？…")
        linked = linked or (
            any(tag in {"EC", "EF"} for tag in tags) and not cleaned.endswith("다")
        )
        count += int(linked)
    return count


def build_aihub_feature_index() -> tuple[
    dict[tuple[str, str, int], tuple[LineRecord, int]],
    dict[tuple[str, str], int],
]:
    kiwi = Kiwi()
    line_index: dict[tuple[str, str, int], tuple[LineRecord, int]] = {}
    page_lines: dict[tuple[str, str], list[int]] = defaultdict(list)
    for split, labels_root in AIHUB_LABEL_ROOTS.items():
        pages, errors = load_label_pages(labels_root)
        if errors:
            raise RuntimeError(f"AI Hub {split} label parse errors: {errors[:3]}")
        by_category, _ = extract_all_lines(pages, padding=12)
        for line in by_category.get("T.Tablet/R.Free", []):
            linked = linked_eojeol_count(line.ground_truth, kiwi)
            key = (split, line.identifier, line.line_index)
            line_index[key] = (line, linked)
            page_lines[(split, line.identifier)].append(linked)
    page_connected_line_count = {
        key: sum(linked >= 2 for linked in values) for key, values in page_lines.items()
    }
    return line_index, page_connected_line_count


def select_aihub_2000(
    source_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    line_index, page_connected = build_aihub_feature_index()
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejection_counts: Counter[str] = Counter()

    for source in source_records:
        text = source["ground_truth"].strip()
        key = (
            source["source_split"],
            source["source_identifier"],
            source["line_index"],
        )
        feature = line_index.get(key)
        if feature is None:
            rejection_counts["missing_reconstructed_line"] += 1
            continue
        line, linked = feature
        if text != line.ground_truth:
            rejection_counts["ground_truth_reconstruction_mismatch"] += 1
            continue
        if len(text.split()) < 2:
            rejection_counts["single_eojeol"] += 1
            continue
        if numeric_or_symbol_dominant(text):
            rejection_counts["numeric_or_symbol_dominant"] += 1
            continue
        if has_pii(text):
            rejection_counts["personal_information"] += 1
            continue
        if source.get("visual_gt_coverage_pass") is not True:
            rejection_counts["visual_gt_coverage_failed"] += 1
            continue
        if line.bbox_id_order_matches_x_order is not True:
            rejection_counts["uncertain_reading_order"] += 1
            continue
        page_key = (source["source_split"], source["source_identifier"])
        page_connected_lines = page_connected.get(page_key, 0)
        # Requiring two independently connected lines on the page rejects the
        # R.Free pages that are simply rows of unrelated dictionary words.
        if page_connected_lines < 2:
            rejection_counts["unrelated_word_list_page"] += 1
            continue
        path = Path(source["image_path"])
        if not path.is_file():
            rejection_counts["missing_crop"] += 1
            continue

        semantic_quality = (
            10 * int(line.automatic_sentence_candidate)
            + 3 * min(linked, 8)
            + 2 * min(line.predicate_ending_count, 2)
            + min(line.grammatical_ending_count, 4)
            + line.sentence_score
        )
        candidate = dict(source)
        candidate["semantic_linked_eojeol_count"] = linked
        candidate["source_page_connected_line_count"] = page_connected_lines
        candidate["automatic_sentence_candidate"] = line.automatic_sentence_candidate
        candidate["semantic_quality_score"] = semantic_quality
        groups[text].append(candidate)

    for ground_truth, values in groups.items():
        values.sort(
            key=lambda item: (
                -item["semantic_quality_score"],
                stable_digest(f"aihub:{ground_truth}:{item['id']}"),
            )
        )

    source_cap = 6
    ground_truth_cap = 8
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    source_counts: Counter[tuple[str, str]] = Counter()

    # Round-robin over GT values: every distinct GT is used once before any GT
    # receives a second occurrence. This minimizes unavoidable repetition.
    for repetition_round in range(ground_truth_cap):
        ordered_ground_truths = sorted(
            groups,
            key=lambda ground_truth: (
                -groups[ground_truth][
                    min(repetition_round, len(groups[ground_truth]) - 1)
                ]["semantic_quality_score"]
                if repetition_round < len(groups[ground_truth])
                else 10**6,
                stable_digest(f"gt-round:{repetition_round}:{ground_truth}"),
            ),
        )
        for ground_truth in ordered_ground_truths:
            if len(selected) >= 2000:
                break
            if repetition_round >= len(groups[ground_truth]):
                continue
            for candidate in groups[ground_truth]:
                if candidate["id"] in selected_ids:
                    continue
                source_key = (
                    candidate["source_split"],
                    candidate["source_identifier"],
                )
                if source_counts[source_key] >= source_cap:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate["id"])
                source_counts[source_key] += 1
                break
        if len(selected) >= 2000:
            break

    if len(selected) != 2000:
        raise RuntimeError(f"Could select only {len(selected):,}/2,000 AI Hub records")

    selected.sort(
        key=lambda item: (
            item["source_split"],
            item["source_identifier"],
            item["line_index"],
        )
    )
    for sequence, item in enumerate(selected, start=1):
        item["source_eval_id"] = item["id"]
        item["id"] = f"{sequence:06d}"

    diagnostics = {
        "source_manifest_count": len(source_records),
        "semantic_candidate_pool_count": sum(len(values) for values in groups.values()),
        "semantic_candidate_unique_ground_truth_count": len(groups),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "source_image_line_cap": source_cap,
        "ground_truth_occurrence_cap": ground_truth_cap,
        "semantic_page_rule": (
            "원본 페이지에서 조사/어미로 연결된 어절이 2개 이상인 행이 최소 2개 존재"
        ),
    }
    return selected, diagnostics


def classify_error(error: dict[str, Any]) -> str:
    rule_id = str(error.get("rule_id") or "")
    family = rule_id.split(".", 1)[0]
    area = str(error.get("error_area") or "")
    if family == "5B":
        return "맞춤법·음운 오류"
    if family == "5C" and ("PARTICLE" in rule_id or area in PARTICLE_AREAS):
        return "조사 오류"
    return ERROR_FAMILY_LABELS.get(family, "기타 오류")


def validate_image(path: Path, expected_size: tuple[int, int] | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        with Image.open(path) as image:
            image.load()
            if expected_size is not None and image.size != expected_size:
                raise ValueError(
                    f"Image size mismatch for {path}: {image.size} != {expected_size}"
                )
            extrema = image.convert("L").getextrema()
            if extrema[1] - extrema[0] < 3:
                raise ValueError(f"Blank or near-blank image: {path}")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Unreadable image: {path}") from exc


def prepare_synthetic_hashes(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, str], set[str]]:
    """Hash all images once and return source-SID hashes and train hash set."""

    hashes: dict[str, str] = {}
    train_hashes: set[str] = set()
    for index, row in enumerate(rows, start=1):
        path = SYNTHETIC_ROOT / row["image"]
        digest = sha256_file(path)
        hashes[row["sid"]] = digest
        if row["split"] == "train":
            train_hashes.add(digest)
        if index % 10000 == 0:
            print(f"Synthetic images hashed: {index:,}/{len(rows):,}", flush=True)
    return hashes, train_hashes


def synthetic_candidate_filter(
    row: dict[str, Any],
    train_texts: set[str],
    image_hashes: dict[str, str],
    train_hashes: set[str],
    *,
    clean: bool,
) -> tuple[bool, str]:
    if row.get("split") not in {"dev", "test"}:
        return False, "training_split"
    if bool(row.get("has_error")) == clean:
        return False, "wrong_error_partition"
    ground_truth = str(row.get("rendered_text") or "").strip()
    if not ground_truth:
        return False, "empty_rendered_text"
    # Unsupported characters are absent from the rendered image.  Keeping such
    # rows would make the corrected sentence and injected-error spans disagree
    # with the actual pixels, so exclude them from both paper evaluation sets.
    if row.get("has_unrendered"):
        return False, "unrendered_character"
    if ground_truth in train_texts:
        return False, "text_overlap_with_training"
    if image_hashes[row["sid"]] in train_hashes:
        return False, "image_overlap_with_training"
    if clean:
        if row.get("has_unrendered"):
            return False, "unrendered_character"
        if len(ground_truth.split()) < 2:
            return False, "not_sentence_length"
        if not SENTENCE_TERMINAL_RE.search(ground_truth):
            return False, "no_terminal_punctuation"
    return True, "accepted"


def select_clean_synthetic(
    candidates: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_split[row["split"]].append(row)
    quotas = {"dev": count // 2, "test": count - count // 2}
    selected: list[dict[str, Any]] = []
    for split in ("dev", "test"):
        values = sorted(
            by_split[split], key=lambda row: stable_digest(f"clean:{split}:{row['sid']}")
        )
        if len(values) < quotas[split]:
            raise RuntimeError(
                f"Only {len(values):,}/{quotas[split]:,} clean {split} candidates"
            )
        selected.extend(values[: quotas[split]])
    selected.sort(key=lambda row: (row["split"], row["sid"]))
    return selected


def make_synthetic_record(
    row: dict[str, Any], sequence: int, image_hash: str, *, clean: bool
) -> dict[str, Any]:
    image_path = (SYNTHETIC_ROOT / row["image"]).resolve()
    record: dict[str, Any] = {
        "id": f"{sequence:06d}",
        "image_path": str(image_path),
        "ground_truth": row["rendered_text"],
        "source": "generated_handwriting_project_b",
        "source_split": row["split"],
        "source_sid": row["sid"],
        "record_id": row["record_id"],
        "writer": row.get("writer"),
        "style_ref": row.get("style_ref"),
        "width": row["width"],
        "height": row["height"],
        "n_lines": row["n_lines"],
        "image_sha256": image_hash,
        "ground_truth_field": "rendered_text",
        "has_unrendered": bool(row.get("has_unrendered")),
        "dropped_chars": row.get("dropped_chars") or [],
    }
    if not clean:
        errors = row.get("errors") or []
        error_types = sorted({classify_error(error) for error in errors})
        record.update(
            {
                "error_type": error_types,
                "corrected_text": row.get("reference_text"),
                "error_count": row.get("error_count", len(errors)),
                "error_details": [
                    {
                        "wrong_text": error.get("wrong_text"),
                        "correct_text": error.get("correct_text"),
                        "error_category": classify_error(error),
                        "error_level": error.get("error_level"),
                        "error_area": error.get("error_area"),
                        "error_code": error.get("error_code"),
                        "surface_pattern": error.get("surface_pattern"),
                        "rule_id": error.get("rule_id"),
                        "injection_operation": error.get("injection_operation"),
                    }
                    for error in errors
                ],
            }
        )
    return record


def distribution_with_percent(counter: Counter[str], total: int) -> dict[str, Any]:
    return {
        key: {"count": value, "percent": percent(value, total)}
        for key, value in sorted(counter.items())
    }


def base_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    gt_counts = Counter(item["ground_truth"] for item in records)
    return {
        "sample_count": len(records),
        "split_distribution": dict(
            sorted(Counter(item["source_split"] for item in records).items())
        ),
        "unique_image_path_count": len({item["image_path"] for item in records}),
        "unique_image_sha256_count": len(
            {item.get("image_sha256", item["image_path"]) for item in records}
        ),
        "unique_ground_truth_count": len(gt_counts),
        "duplicate_ground_truth_occurrence_count": len(records) - len(gt_counts),
        "maximum_ground_truth_frequency": max(gt_counts.values(), default=0),
        "ground_truth_length": numeric_stats(
            [len(item["ground_truth"]) for item in records]
        ),
        "ground_truth_eojeol_count": numeric_stats(
            [len(item["ground_truth"].split()) for item in records]
        ),
    }


def build_aihub_stats(
    records: list[dict[str, Any]], diagnostics: dict[str, Any]
) -> dict[str, Any]:
    stats = base_stats(records)
    source_counts = Counter(
        (item["source_split"], item["source_identifier"]) for item in records
    )
    stats.update(
        {
            "dataset": "AI Hub 605 real handwriting R.Free line crops",
            "source_manifest": str(AIHUB_4000.resolve()),
            "source_manifest_preserved": AIHUB_4000.is_file(),
            "source_image_count": len(source_counts),
            "maximum_lines_per_source_image": max(source_counts.values()),
            "lines_per_source_image": numeric_stats(list(source_counts.values())),
            "automatic_sentence_candidate_count": sum(
                item["automatic_sentence_candidate"] for item in records
            ),
            "terminal_punctuation_count": sum(
                bool(SENTENCE_TERMINAL_RE.search(item["ground_truth"]))
                for item in records
            ),
            "semantic_linked_eojeol_count": numeric_stats(
                [item["semantic_linked_eojeol_count"] for item in records]
            ),
            "crop_width": numeric_stats([item["crop_width"] for item in records]),
            "crop_height": numeric_stats([item["crop_height"] for item in records]),
            "selection": diagnostics,
            "validation": {
                "all_images_exist_and_readable": True,
                "all_ground_truth_nonempty": True,
                "all_visual_gt_coverage_pass": all(
                    item["visual_gt_coverage_pass"] is True for item in records
                ),
                "unique_image_paths": len({item["image_path"] for item in records})
                == len(records),
            },
        }
    )
    return stats


def build_synthetic_stats(
    records: list[dict[str, Any]],
    *,
    clean: bool,
    candidate_count: int,
    rejection_counts: Counter[str],
    target_count: int,
    train_text_overlap_count: int,
    train_image_overlap_count: int,
) -> dict[str, Any]:
    stats = base_stats(records)
    stats.update(
        {
            "dataset": "project-b synthetic normal" if clean else "project-b synthetic error",
            "target_count": target_count,
            "target_reached": len(records) == target_count,
            "holdout_definition": "metadata split in {dev, test}; split=train excluded",
            "holdout_candidate_count_after_filters": candidate_count,
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "writer_distribution": dict(
                sorted(Counter(str(item["writer"]) for item in records).items())
            ),
            "n_lines": numeric_stats([item["n_lines"] for item in records]),
            "has_unrendered_count": sum(item["has_unrendered"] for item in records),
            "training_overlap_validation": {
                "ground_truth_overlap_with_any_training_text_field": train_text_overlap_count,
                "image_sha256_overlap_with_training": train_image_overlap_count,
                "passed": train_text_overlap_count == 0 and train_image_overlap_count == 0,
            },
            "selection_policy": {
                "seed": SEED,
                "training_split_excluded": True,
                "ground_truth_field": "rendered_text",
                "image_content_hash_deduplicated": True,
                "ground_truth_deduplicated": True,
                "has_unrendered_excluded": True,
            },
            "validation": {
                "all_images_exist_and_readable": True,
                "all_ground_truth_nonempty": True,
                "unique_image_paths": len({item["image_path"] for item in records})
                == len(records),
                "unique_image_hashes": len({item["image_sha256"] for item in records})
                == len(records),
            },
        }
    )
    if not clean:
        annotation_categories = Counter(
            detail["error_category"]
            for item in records
            for detail in item["error_details"]
        )
        sample_categories = Counter(
            category for item in records for category in set(item["error_type"])
        )
        error_levels = Counter(
            str(detail["error_level"] or "UNKNOWN")
            for item in records
            for detail in item["error_details"]
        )
        error_areas = Counter(
            str(detail["error_area"] or "UNKNOWN")
            for item in records
            for detail in item["error_details"]
        )
        surface_patterns = Counter(
            str(detail["surface_pattern"] or "UNKNOWN")
            for item in records
            for detail in item["error_details"]
        )
        annotation_total = sum(annotation_categories.values())
        stats["error_distribution"] = {
            "sample_level_multilabel": distribution_with_percent(
                sample_categories, len(records)
            ),
            "annotation_level": distribution_with_percent(
                annotation_categories, annotation_total
            ),
            "error_level": dict(sorted(error_levels.items())),
            "error_area": dict(sorted(error_areas.items())),
            "surface_pattern": dict(sorted(surface_patterns.items())),
            "error_count_per_sample": dict(
                sorted(
                    Counter(str(item["error_count"]) for item in records).items(),
                    key=lambda pair: int(pair[0]),
                )
            ),
        }
    return stats


def create_preview(
    records: list[dict[str, Any]], title: str, *, error: bool = False
) -> str:
    selected = random.Random(SEED).sample(records, min(20, len(records)))
    cards: list[str] = []
    for item in selected:
        image_path = Path(item["image_path"])
        media_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
        encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
        image_source = f"data:{media_type};base64,{encoded_image}"
        extra = ""
        if error:
            extra = (
                f'<dt>corrected</dt><dd>{html.escape(str(item.get("corrected_text")))}</dd>'
                f'<dt>error type</dt><dd>{html.escape(", ".join(item["error_type"]))}</dd>'
            )
        elif item.get("source") == "aihub605":
            extra = (
                f'<dt>source</dt><dd>{html.escape(item["source_identifier"])}</dd>'
                f'<dt>semantic links / page lines</dt><dd>'
                f'{item["semantic_linked_eojeol_count"]} / '
                f'{item["source_page_connected_line_count"]}</dd>'
            )
        cards.append(
            '<article class="card">'
            f'<img src="{image_source}" loading="lazy" '
            f'alt="{html.escape(item["ground_truth"])}">'
            f'<h2>{html.escape(item["ground_truth"])}</h2>'
            f'<dl><dt>ID / split</dt><dd>{item["id"]} / '
            f'{html.escape(item["source_split"])}</dd>{extra}</dl></article>'
        )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f3f5f7;color:#17202a}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}}
.card{{background:#fff;border:1px solid #d8dde3;border-radius:10px;padding:14px}}
img{{display:block;max-width:100%;max-height:260px;margin:auto;background:#eee}}
h2{{font-size:1.05rem}}dl{{font-size:.82rem;overflow-wrap:anywhere}}
dt{{font-weight:700;margin-top:6px}}dd{{margin-left:0;color:#445}}
</style></head><body><h1>{html.escape(title)}</h1>
<p>seed={SEED}, random preview {len(selected)}</p>
<main class="grid">{''.join(cards)}</main></body></html>\n"""


def validate_records(records: list[dict[str, Any]], expected: int | None = None) -> None:
    if expected is not None and len(records) != expected:
        raise RuntimeError(f"Record count {len(records):,} != {expected:,}")
    for field in ("id", "image_path"):
        values = [item[field] for item in records]
        if len(values) != len(set(values)):
            raise RuntimeError(f"Duplicate {field}")
    if any(not item["ground_truth"].strip() for item in records):
        raise RuntimeError("Empty ground_truth")
    for item in records:
        expected_size = None
        if "width" in item and "height" in item:
            expected_size = (item["width"], item["height"])
        elif "crop_width" in item and "crop_height" in item:
            expected_size = (item["crop_width"], item["crop_height"])
        validate_image(Path(item["image_path"]), expected_size)


def ensure_outputs_absent(paths: list[Path]) -> None:
    conflicts = [path for path in paths if path.exists()]
    if conflicts:
        raise SystemExit(
            "Refusing to overwrite existing outputs: "
            + ", ".join(str(path) for path in conflicts)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this script's nine final-evaluation output files.",
    )
    mode.add_argument(
        "--refresh-previews",
        action="store_true",
        help="Rebuild only the three standalone HTML previews from existing manifests.",
    )
    args = parser.parse_args()
    outputs = {
        "aihub_jsonl": BENCHMARK_DIR / "eval_handwriting_2000.jsonl",
        "aihub_stats": BENCHMARK_DIR / "eval_handwriting_2000_stats.json",
        "aihub_preview": BENCHMARK_DIR / "sample_preview_2000.html",
        "normal_jsonl": BENCHMARK_DIR / "eval_synthetic_normal_2000.jsonl",
        "normal_stats": BENCHMARK_DIR / "eval_synthetic_normal_2000_stats.json",
        "normal_preview": BENCHMARK_DIR / "sample_preview_synthetic_normal.html",
        "error_jsonl": BENCHMARK_DIR / "eval_synthetic_error_2000.jsonl",
        "error_stats": BENCHMARK_DIR / "eval_synthetic_error_2000_stats.json",
        "error_preview": BENCHMARK_DIR / "sample_preview_synthetic_error.html",
    }
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    if args.refresh_previews:
        preview_specs = (
            (
                outputs["aihub_jsonl"],
                outputs["aihub_preview"],
                "AI Hub 실제 손글씨 평가셋 2,000",
                False,
            ),
            (
                outputs["normal_jsonl"],
                outputs["normal_preview"],
                "합성 정상 문장 hold-out 평가셋",
                False,
            ),
            (
                outputs["error_jsonl"],
                outputs["error_preview"],
                "합성 오류 문장 hold-out 평가셋",
                True,
            ),
        )
        for manifest, preview, title, is_error in preview_specs:
            atomic_write_text(
                preview,
                create_preview(read_jsonl(manifest), title, error=is_error),
            )
            print(f"Refreshed: {preview}", flush=True)
        return 0
    if not args.overwrite:
        ensure_outputs_absent(list(outputs.values()))

    print("Selecting AI Hub final 2,000...", flush=True)
    aihub_source = read_jsonl(AIHUB_4000)
    aihub_records, aihub_diagnostics = select_aihub_2000(aihub_source)
    validate_records(aihub_records, expected=2000)
    aihub_stats = build_aihub_stats(aihub_records, aihub_diagnostics)

    print("Preparing synthetic hold-out sets...", flush=True)
    synthetic_rows = read_jsonl(SYNTHETIC_METADATA)
    image_hashes, train_image_hashes = prepare_synthetic_hashes(synthetic_rows)
    train_texts = {
        str(row.get(field) or "")
        for row in synthetic_rows
        if row["split"] == "train"
        for field in ("text", "rendered_text", "reference_text")
    }

    partitions: dict[bool, list[dict[str, Any]]] = {True: [], False: []}
    rejections: dict[bool, Counter[str]] = {True: Counter(), False: Counter()}
    for clean in (True, False):
        for row in synthetic_rows:
            accepted, reason = synthetic_candidate_filter(
                row,
                train_texts,
                image_hashes,
                train_image_hashes,
                clean=clean,
            )
            if accepted:
                partitions[clean].append(row)
            else:
                rejections[clean][reason] += 1

    clean_selected = select_clean_synthetic(partitions[True], 2000)
    # Error hold-out has fewer than 2,000 records. Keep every valid unique
    # non-training record rather than padding with train or duplicating samples.
    error_selected = sorted(
        partitions[False], key=lambda row: (row["split"], row["sid"])
    )[:2000]

    clean_records = [
        make_synthetic_record(row, index, image_hashes[row["sid"]], clean=True)
        for index, row in enumerate(clean_selected, start=1)
    ]
    error_records = [
        make_synthetic_record(row, index, image_hashes[row["sid"]], clean=False)
        for index, row in enumerate(error_selected, start=1)
    ]
    validate_records(clean_records, expected=2000)
    validate_records(error_records)

    clean_stats = build_synthetic_stats(
        clean_records,
        clean=True,
        candidate_count=len(partitions[True]),
        rejection_counts=rejections[True],
        target_count=2000,
        train_text_overlap_count=sum(
            item["ground_truth"] in train_texts for item in clean_records
        ),
        train_image_overlap_count=sum(
            item["image_sha256"] in train_image_hashes for item in clean_records
        ),
    )
    error_stats = build_synthetic_stats(
        error_records,
        clean=False,
        candidate_count=len(partitions[False]),
        rejection_counts=rejections[False],
        target_count=2000,
        train_text_overlap_count=sum(
            item["ground_truth"] in train_texts for item in error_records
        ),
        train_image_overlap_count=sum(
            item["image_sha256"] in train_image_hashes for item in error_records
        ),
    )

    write_jsonl(outputs["aihub_jsonl"], aihub_records)
    atomic_write_json(outputs["aihub_stats"], aihub_stats)
    atomic_write_text(
        outputs["aihub_preview"],
        create_preview(aihub_records, "AI Hub 실제 손글씨 평가셋 2,000"),
    )
    write_jsonl(outputs["normal_jsonl"], clean_records)
    atomic_write_json(outputs["normal_stats"], clean_stats)
    atomic_write_text(
        outputs["normal_preview"],
        create_preview(clean_records, "합성 정상 문장 hold-out 평가셋"),
    )
    write_jsonl(outputs["error_jsonl"], error_records)
    atomic_write_json(outputs["error_stats"], error_stats)
    atomic_write_text(
        outputs["error_preview"],
        create_preview(error_records, "합성 오류 문장 hold-out 평가셋", error=True),
    )

    # Confirm the three manifests are mutually separate and the original stays.
    if not AIHUB_4000.is_file() or len(read_jsonl(AIHUB_4000)) != 4000:
        raise RuntimeError("The preserved AI Hub 4,000 manifest changed")
    print(
        f"Completed: AI Hub={len(aihub_records):,}, "
        f"synthetic normal={len(clean_records):,}, "
        f"synthetic error={len(error_records):,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
