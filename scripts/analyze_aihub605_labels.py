#!/usr/bin/env python3
"""Analyze every AI Hub 605 Validation label without modifying source data."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_LABELS_ROOT = Path(
    "data/053.대용량_손글씨_OCR_데이터/01.데이터/2.Validation/"
    "라벨링데이터/VL/라벨/HW-OCR/4.Validation"
)
DEFAULT_OUTPUT_DIR = Path("results/aihub605_label_analysis")

PUNCTUATION_RE = re.compile(r"[.!?。！？…,:;\"'“”‘’()\[\]{}~·ㆍ—–-]")
SENTENCE_PUNCTUATION_RE = re.compile(r"[.!?。！？…]")
HANGUL_RE = re.compile(r"[가-힣]")
DIGIT_RE = re.compile(r"[0-9]")
WHITESPACE_RE = re.compile(r"\s")
EMAIL_RE = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")
LONG_DIGIT_RE = re.compile(r"\d{7,}")

# These endings only nominate candidates. They do not prove that a row is a sentence.
PREDICATE_ENDINGS = (
    "습니다",
    "습니까",
    "입니다",
    "였습니다",
    "있습니다",
    "없습니다",
    "하였다",
    "했다",
    "한다",
    "된다",
    "이다",
    "였다",
    "있다",
    "없다",
    "된다",
    "하며",
    "이며",
    "면서",
    "한다면",
    "하였다면",
    "하고",
    "되고",
    "라고",
    "다고",
    "한다",
    "하는",
    "하여",
    "해서",
    "되어",
    "돼서",
    "된다",
    "된다면",
    "해요",
    "예요",
    "이에요",
    "군요",
    "네요",
)

GRAMMATICAL_ENDINGS = (
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "에",
    "에서",
    "에게",
    "께",
    "으로",
    "로",
    "와",
    "과",
    "도",
    "만",
    "부터",
    "까지",
    "보다",
    "처럼",
    "하며",
    "이며",
    "하고",
    "라고",
    "다고",
    "지만",
    "므로",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def display_value(value: Any) -> str:
    if value is None:
        return "<MISSING>"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.fmean(values), 4) if values else 0.0


def safe_median(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.median(values), 4) if values else 0.0


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def category_from_label_path(label_path: Any, json_path: Path) -> str:
    if isinstance(label_path, str) and label_path:
        parts = Path(label_path).parts
        if "4.Validation" in parts:
            index = parts.index("4.Validation")
            suffix = parts[index + 1 :]
            if suffix:
                return "/".join(suffix)
    parent_parts = json_path.parent.parts
    if "4.Validation" in parent_parts:
        index = parent_parts.index("4.Validation")
        return "/".join(parent_parts[index + 1 :])
    return "<UNKNOWN>"


def clean_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def char_length(text: str) -> int:
    return len(text)


def nonspace_length(text: str) -> int:
    return len(WHITESPACE_RE.sub("", text))


def has_predicate_ending(token: str) -> bool:
    bare = SENTENCE_PUNCTUATION_RE.sub("", token.strip())
    return any(bare.endswith(ending) for ending in PREDICATE_ENDINGS)


def has_grammatical_ending(token: str) -> bool:
    bare = SENTENCE_PUNCTUATION_RE.sub("", token.strip())
    return any(bare.endswith(ending) for ending in GRAMMATICAL_ENDINGS)


@dataclass
class Box:
    text: str
    box_id: Any
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def cx(self) -> float:
        return (self.xmin + self.xmax) / 2

    @property
    def cy(self) -> float:
        return (self.ymin + self.ymax) / 2

    @property
    def height(self) -> float:
        return max(1.0, self.ymax - self.ymin)


def parse_box(raw: Any) -> Box | None:
    if not isinstance(raw, dict):
        return None
    text = clean_text(raw.get("data"))
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
    return Box(
        text=text,
        box_id=raw.get("id"),
        xmin=min(xs),
        ymin=min(ys),
        xmax=max(xs),
        ymax=max(ys),
    )


def reconstruct_lines(boxes: list[Box]) -> list[list[Box]]:
    """Group boxes by y-center, then order each row from left to right."""
    nonempty = [box for box in boxes if box.text]
    if not nonempty:
        return []
    median_height = statistics.median(box.height for box in nonempty)
    center_tolerance = max(10.0, median_height * 0.60)
    rows: list[dict[str, Any]] = []

    for box in sorted(nonempty, key=lambda value: (value.cy, value.cx)):
        eligible = [
            (abs(box.cy - row["center"]), row)
            for row in rows
            if abs(box.cy - row["center"]) <= center_tolerance
        ]
        if eligible:
            _, row = min(eligible, key=lambda item: item[0])
            row["boxes"].append(box)
            row["center"] = statistics.fmean(item.cy for item in row["boxes"])
        else:
            rows.append({"center": box.cy, "boxes": [box]})

    rows.sort(key=lambda row: row["center"])
    return [sorted(row["boxes"], key=lambda value: (value.xmin, value.box_id or 0)) for row in rows]


def bbox_candidate_reasons(text: str) -> list[str]:
    reasons = []
    if char_length(text) >= 10:
        reasons.append("length>=10")
    if WHITESPACE_RE.search(text):
        reasons.append("whitespace")
    if PUNCTUATION_RE.search(text):
        reasons.append("punctuation")
    return reasons


def sentence_like_line(tokens: list[str]) -> tuple[bool, list[str]]:
    text = " ".join(tokens)
    compact_length = nonspace_length(text)
    hangul_count = len(HANGUL_RE.findall(text))
    digit_count = len(DIGIT_RE.findall(text))
    denominator = max(1, compact_length)
    predicate_count = sum(has_predicate_ending(token) for token in tokens)
    grammatical_count = sum(has_grammatical_ending(token) for token in tokens)

    reasons = []
    if len(tokens) >= 3:
        reasons.append("tokens>=3")
    if compact_length >= 10:
        reasons.append("length>=10")
    if hangul_count / denominator >= 0.60:
        reasons.append("hangul_ratio>=0.60")
    if digit_count / denominator <= 0.20:
        reasons.append("digit_ratio<=0.20")
    if predicate_count:
        reasons.append("predicate_ending")
    if grammatical_count >= 2:
        reasons.append("grammatical_endings>=2")
    if SENTENCE_PUNCTUATION_RE.search(text):
        reasons.append("sentence_punctuation")

    required = (
        len(tokens) >= 3
        and compact_length >= 10
        and hangul_count / denominator >= 0.60
        and digit_count / denominator <= 0.20
        and (
            (predicate_count >= 1 and grammatical_count >= 2)
            or bool(SENTENCE_PUNCTUATION_RE.search(text))
        )
    )
    return required, reasons


def eligible_free_bbox(text: str) -> bool:
    """Conservative label-only filter used only to assess 4,000-crop feasibility."""
    compact = WHITESPACE_RE.sub("", text)
    if len(compact) < 2 or not HANGUL_RE.search(compact):
        return False
    if EMAIL_RE.search(text) or LONG_DIGIT_RE.search(compact):
        return False
    digit_count = len(DIGIT_RE.findall(compact))
    if digit_count / max(1, len(compact)) > 0.20:
        return False
    return True


def greedy_unique_selection_count(image_texts: dict[str, set[str]]) -> int:
    """Construct a deterministic one-bbox-per-image, unique-GT selection."""
    text_frequency = Counter(text for texts in image_texts.values() for text in texts)
    used_texts: set[str] = set()
    selected = 0
    for identifier, texts in sorted(image_texts.items(), key=lambda item: (len(item[1]), item[0])):
        choices = sorted(texts, key=lambda text: (text_frequency[text], -len(text), text))
        choice = next((text for text in choices if text not in used_texts), None)
        if choice is not None:
            used_texts.add(choice)
            selected += 1
    return selected


@dataclass
class CategoryStats:
    json_count: int = 0
    bbox_counts: list[int] = field(default_factory=list)
    bbox_lengths: list[int] = field(default_factory=list)
    empty_bbox_count: int = 0
    invalid_coordinate_count: int = 0
    out_of_bounds_count: int = 0
    whitespace_bbox_count: int = 0
    punctuation_bbox_count: int = 0
    sentence_punctuation_bbox_count: int = 0
    long_bbox_count: int = 0
    bbox_candidate_count: int = 0
    line_count: int = 0
    multi_bbox_line_count: int = 0
    sentence_like_line_count: int = 0
    sentence_like_line_lengths: list[int] = field(default_factory=list)
    field_counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    bbox_candidates: list[dict[str, Any]] = field(default_factory=list)
    line_candidates: list[dict[str, Any]] = field(default_factory=list)
    longest_bbox: dict[str, Any] | None = None
    longest_line: dict[str, Any] | None = None

    def update_longest_bbox(self, record: dict[str, Any]) -> None:
        if self.longest_bbox is None or record["length"] > self.longest_bbox["length"]:
            self.longest_bbox = record

    def update_longest_line(self, record: dict[str, Any]) -> None:
        if self.longest_line is None or record["length"] > self.longest_line["length"]:
            self.longest_line = record


def sample_records(records: list[dict[str, Any]], seed: int, limit: int = 20) -> list[dict[str, Any]]:
    if len(records) <= limit:
        return records
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), limit))
    return [records[index] for index in indices]


def compact_category_stats(stats: CategoryStats, seed: int) -> dict[str, Any]:
    bbox_total = len(stats.bbox_lengths)
    return {
        "json_count": stats.json_count,
        "bbox_total": bbox_total,
        "bbox_per_json_mean": safe_mean(stats.bbox_counts),
        "bbox_per_json_median": safe_median(stats.bbox_counts),
        "bbox_data_length_mean": safe_mean(stats.bbox_lengths),
        "bbox_data_length_median": safe_median(stats.bbox_lengths),
        "bbox_data_length_max": max(stats.bbox_lengths, default=0),
        "empty_bbox_count": stats.empty_bbox_count,
        "invalid_coordinate_count": stats.invalid_coordinate_count,
        "out_of_bounds_count": stats.out_of_bounds_count,
        "whitespace_bbox_count": stats.whitespace_bbox_count,
        "whitespace_bbox_ratio": ratio(stats.whitespace_bbox_count, bbox_total),
        "punctuation_bbox_count": stats.punctuation_bbox_count,
        "punctuation_bbox_ratio": ratio(stats.punctuation_bbox_count, bbox_total),
        "sentence_punctuation_bbox_count": stats.sentence_punctuation_bbox_count,
        "sentence_punctuation_bbox_ratio": ratio(stats.sentence_punctuation_bbox_count, bbox_total),
        "length_ge_10_bbox_count": stats.long_bbox_count,
        "length_ge_10_bbox_ratio": ratio(stats.long_bbox_count, bbox_total),
        "bbox_candidate_count": stats.bbox_candidate_count,
        "bbox_candidate_ratio": ratio(stats.bbox_candidate_count, bbox_total),
        "line_count": stats.line_count,
        "multi_bbox_line_count": stats.multi_bbox_line_count,
        "sentence_like_line_candidate_count": stats.sentence_like_line_count,
        "sentence_like_line_candidate_ratio": ratio(stats.sentence_like_line_count, stats.line_count),
        "sentence_like_line_length_mean": safe_mean(stats.sentence_like_line_lengths),
        "sentence_like_line_length_max": max(stats.sentence_like_line_lengths, default=0),
        "field_distributions": {
            key: counter_to_dict(value) for key, value in sorted(stats.field_counts.items())
        },
        "longest_bbox": stats.longest_bbox,
        "longest_reconstructed_line": stats.longest_line,
        "bbox_candidate_examples_20": sample_records(stats.bbox_candidates, seed),
        "sentence_like_line_examples_20": sample_records(stats.line_candidates, seed),
    }


def add_field_distributions(
    category_stats: CategoryStats,
    dataset: dict[str, Any],
    images: dict[str, Any],
) -> None:
    fields = {
        "Dataset.label_path": dataset.get("label_path"),
        "Dataset.src_path": dataset.get("src_path"),
        "Images.written_content": images.get("written_content"),
        "Images.media_type": images.get("media_type"),
        "Images.pen_type": images.get("pen_type"),
        "Images.writer_age": images.get("writer_age"),
        "Images.writer_sex": images.get("writer_sex"),
        "Images.type": images.get("type"),
        "Images.application_field": images.get("application_field"),
        "Images.acquisition_location": images.get(
            "acquisition_location", images.get("acquistion_location")
        ),
    }
    for key, value in fields.items():
        category_stats.field_counts[key][display_value(value)] += 1


def analyze(labels_root: Path, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    json_files = sorted(labels_root.rglob("*.json"))
    categories: dict[str, CategoryStats] = defaultdict(CategoryStats)
    malformed_files: list[dict[str, str]] = []
    top_level_key_sets: Counter[str] = Counter()
    dataset_key_presence: Counter[str] = Counter()
    images_key_presence: Counter[str] = Counter()
    bbox_key_presence: Counter[str] = Counter()
    identifier_counts: Counter[str] = Counter()
    filename_identifier_match_count = 0
    label_src_path_match_count = 0
    free_eligible_counts: Counter[str] = Counter()
    free_eligible_texts: dict[str, Counter[str]] = defaultdict(Counter)
    free_image_texts_by_category: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    all_bbox_candidates: list[dict[str, Any]] = []
    all_line_candidates: list[dict[str, Any]] = []

    for json_path in json_files:
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            malformed_files.append({"json_path": str(json_path), "error": str(exc)})
            continue

        if not isinstance(payload, dict):
            malformed_files.append({"json_path": str(json_path), "error": "top level is not an object"})
            continue

        dataset = payload.get("Dataset") if isinstance(payload.get("Dataset"), dict) else {}
        images = payload.get("Images") if isinstance(payload.get("Images"), dict) else {}
        raw_boxes = payload.get("bbox") if isinstance(payload.get("bbox"), list) else []
        category = category_from_label_path(dataset.get("label_path"), json_path)
        stats = categories[category]
        stats.json_count += 1
        stats.bbox_counts.append(len(raw_boxes))
        add_field_distributions(stats, dataset, images)

        top_level_key_sets[", ".join(sorted(payload.keys()))] += 1
        dataset_key_presence.update(dataset.keys())
        images_key_presence.update(images.keys())

        identifier = display_value(images.get("identifier"))
        identifier_counts[identifier] += 1
        if json_path.stem == identifier:
            filename_identifier_match_count += 1
        if dataset.get("label_path") == dataset.get("src_path"):
            label_src_path_match_count += 1
        width = images.get("width")
        height = images.get("height")
        parsed_boxes: list[Box] = []

        for raw_box in raw_boxes:
            if isinstance(raw_box, dict):
                bbox_key_presence.update(raw_box.keys())
            text = clean_text(raw_box.get("data")) if isinstance(raw_box, dict) else ""
            text_len = char_length(text)
            stats.bbox_lengths.append(text_len)
            if not text:
                stats.empty_bbox_count += 1
            if WHITESPACE_RE.search(text):
                stats.whitespace_bbox_count += 1
            if PUNCTUATION_RE.search(text):
                stats.punctuation_bbox_count += 1
            if SENTENCE_PUNCTUATION_RE.search(text):
                stats.sentence_punctuation_bbox_count += 1
            if text_len >= 10:
                stats.long_bbox_count += 1
            if category.endswith("/R.Free") and eligible_free_bbox(text):
                free_eligible_counts[category] += 1
                free_eligible_texts[category][text] += 1
                free_image_texts_by_category[category][identifier].add(text)

            record = {
                "category": category,
                "identifier": identifier,
                "json_path": str(json_path),
                "bbox_id": raw_box.get("id") if isinstance(raw_box, dict) else None,
                "text": text,
                "length": text_len,
            }
            stats.update_longest_bbox(record)
            reasons = bbox_candidate_reasons(text)
            if reasons:
                candidate = {**record, "reasons": reasons}
                stats.bbox_candidate_count += 1
                stats.bbox_candidates.append(candidate)
                all_bbox_candidates.append(candidate)

            parsed = parse_box(raw_box)
            if parsed is None:
                stats.invalid_coordinate_count += 1
                continue
            parsed_boxes.append(parsed)
            if (
                isinstance(width, (int, float))
                and isinstance(height, (int, float))
                and (
                    parsed.xmin < 0
                    or parsed.ymin < 0
                    or parsed.xmax > width
                    or parsed.ymax > height
                    or parsed.xmax <= parsed.xmin
                    or parsed.ymax <= parsed.ymin
                )
            ):
                stats.out_of_bounds_count += 1

        lines = reconstruct_lines(parsed_boxes)
        stats.line_count += len(lines)
        stats.multi_bbox_line_count += sum(len(line) >= 2 for line in lines)
        for line_index, line in enumerate(lines, start=1):
            tokens = [box.text for box in line]
            line_text = " ".join(tokens)
            line_record = {
                "category": category,
                "identifier": identifier,
                "json_path": str(json_path),
                "line_index": line_index,
                "bbox_ids": [box.box_id for box in line],
                "bbox_count": len(line),
                "text": line_text,
                "length": char_length(line_text),
            }
            stats.update_longest_line(line_record)
            is_candidate, reasons = sentence_like_line(tokens)
            if is_candidate:
                candidate = {**line_record, "reasons": reasons}
                stats.sentence_like_line_count += 1
                stats.sentence_like_line_lengths.append(char_length(line_text))
                stats.line_candidates.append(candidate)
                all_line_candidates.append(candidate)

    category_output = {
        category: compact_category_stats(stats, seed)
        for category, stats in sorted(categories.items())
    }
    parsed_count = sum(stats.json_count for stats in categories.values())
    combined_free_image_texts: dict[str, set[str]] = {}
    for category_images in free_image_texts_by_category.values():
        combined_free_image_texts.update(category_images)
    combined_free_text_counts = Counter()
    for counts in free_eligible_texts.values():
        combined_free_text_counts.update(counts)
    feasibility_by_category = {}
    for category in sorted(free_image_texts_by_category):
        text_counts = free_eligible_texts[category]
        image_texts = free_image_texts_by_category[category]
        feasibility_by_category[category] = {
            "eligible_bbox_count": free_eligible_counts[category],
            "eligible_unique_ground_truth_count": len(text_counts),
            "eligible_duplicate_ground_truth_occurrences": sum(
                count - 1 for count in text_counts.values() if count > 1
            ),
            "images_with_eligible_bbox": len(image_texts),
            "greedy_unique_gt_one_bbox_per_image_count": greedy_unique_selection_count(image_texts),
        }
    output = {
        "analysis_version": 1,
        "labels_root": str(labels_root),
        "seed": seed,
        "json_files_discovered": len(json_files),
        "json_files_parsed": parsed_count,
        "malformed_file_count": len(malformed_files),
        "malformed_files": malformed_files,
        "schema": {
            "top_level_key_sets": counter_to_dict(top_level_key_sets),
            "Dataset_key_presence": counter_to_dict(dataset_key_presence),
            "Images_key_presence": counter_to_dict(images_key_presence),
            "bbox_key_presence": counter_to_dict(bbox_key_presence),
        },
        "linkage_checks": {
            "unique_identifier_count": len(identifier_counts),
            "duplicate_identifier_count": sum(count - 1 for count in identifier_counts.values() if count > 1),
            "json_filename_stem_equals_Images_identifier_count": filename_identifier_match_count,
            "Dataset_label_path_equals_src_path_count": label_src_path_match_count,
            "image_extension_distribution": counter_to_dict(
                Counter(
                    key
                    for stats in categories.values()
                    for key, count in stats.field_counts["Images.type"].items()
                    for _ in range(count)
                )
            ),
        },
        "bbox_crop_feasibility": {
            "filter": (
                "R.Free만 사용; 공백 제거 길이 2 이상; 한글 포함; 숫자 비율 20% 이하; "
                "이메일 및 7자리 이상 연속 숫자 제외. 이는 라벨 단계의 보수적 1차 필터이며 "
                "개인정보 완전 탐지를 보장하지 않는다."
            ),
            "by_category": feasibility_by_category,
            "combined": {
                "eligible_bbox_count": sum(free_eligible_counts.values()),
                "eligible_unique_ground_truth_count": len(combined_free_text_counts),
                "eligible_duplicate_ground_truth_occurrences": sum(
                    count - 1 for count in combined_free_text_counts.values() if count > 1
                ),
                "images_with_eligible_bbox": len(combined_free_image_texts),
                "greedy_unique_gt_one_bbox_per_image_count": greedy_unique_selection_count(
                    combined_free_image_texts
                ),
            },
        },
        "category_json_counts": {
            category: stats.json_count for category, stats in sorted(categories.items())
        },
        "categories": category_output,
        "automatic_candidate_warning": (
            "문장형 후보는 좌표 기반 행 재구성과 휴리스틱 결과이며 실제 자연어 문장임을 보장하지 않는다. "
            "예시를 사람이 검토해야 한다."
        ),
    }
    return output, all_bbox_candidates, all_line_candidates


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def markdown_report(analysis: dict[str, Any]) -> str:
    lines = [
        "# AI Hub 605 Validation 라벨 자동 분석",
        "",
        "> 이 보고서의 문장형 후보는 휴리스틱 결과다. 실제 예시를 확인하기 전에는 문장으로 확정하지 않는다.",
        "",
        "## 전체",
        "",
        f"- 발견 JSON: {analysis['json_files_discovered']:,}",
        f"- 정상 파싱 JSON: {analysis['json_files_parsed']:,}",
        f"- 파싱 실패 JSON: {analysis['malformed_file_count']:,}",
        "",
        "## Category별 통계",
        "",
        "| category | JSON | bbox | bbox/JSON 평균 | bbox 문자열 평균 | 최대 길이 | 공백 bbox | 문장부호 bbox | 문장형 행 후보 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category, stats in analysis["categories"].items():
        lines.append(
            f"| {category} | {stats['json_count']:,} | {stats['bbox_total']:,} | "
            f"{stats['bbox_per_json_mean']:.2f} | {stats['bbox_data_length_mean']:.2f} | "
            f"{stats['bbox_data_length_max']:,} | {stats['whitespace_bbox_count']:,} "
            f"({stats['whitespace_bbox_ratio']:.2%}) | {stats['sentence_punctuation_bbox_count']:,} "
            f"({stats['sentence_punctuation_bbox_ratio']:.2%}) | "
            f"{stats['sentence_like_line_candidate_count']:,} |"
        )

    linkage = analysis["linkage_checks"]
    feasibility = analysis["bbox_crop_feasibility"]
    combined = feasibility["combined"]
    lines.extend(
        [
            "",
            "## 원본 이미지 연결 규칙 점검",
            "",
            f"- 고유 `Images.identifier`: {linkage['unique_identifier_count']:,}",
            f"- 중복 identifier: {linkage['duplicate_identifier_count']:,}",
            f"- JSON 파일명 stem과 identifier 일치: {linkage['json_filename_stem_equals_Images_identifier_count']:,}",
            f"- `Dataset.label_path`와 `Dataset.src_path` 일치: {linkage['Dataset_label_path_equals_src_path_count']:,}",
            f"- 라벨상 이미지 확장자: {json.dumps(linkage['image_extension_distribution'], ensure_ascii=False)}",
            "- 따라서 예상 파일명은 `<Images.identifier>.<Images.type>`이지만, 원천 이미지 다운로드 후 실제 경로와 파일명을 다시 검증해야 한다.",
            "",
            "## bbox crop 4,000개 구성 가능성",
            "",
            f"- 라벨 1차 필터: {feasibility['filter']}",
            f"- 두 R.Free 유형의 적격 bbox: {combined['eligible_bbox_count']:,}",
            f"- 적격 고유 Ground Truth: {combined['eligible_unique_ground_truth_count']:,}",
            f"- 적격 bbox가 있는 원본 이미지: {combined['images_with_eligible_bbox']:,}",
            f"- 원본당 최대 1개, Ground Truth 중복 없이 탐욕적으로 선택 가능했던 수: {combined['greedy_unique_gt_one_bbox_per_image_count']:,}",
            "",
            "| category | 적격 bbox | 고유 GT | 적격 원본 | 원본당 1개·고유 GT 선택 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for category, values in feasibility["by_category"].items():
        lines.append(
            f"| {category} | {values['eligible_bbox_count']:,} | "
            f"{values['eligible_unique_ground_truth_count']:,} | "
            f"{values['images_with_eligible_bbox']:,} | "
            f"{values['greedy_unique_gt_one_bbox_per_image_count']:,} |"
        )
    likely_sentence_category = analysis["categories"].get("T.Tablet/R.Free", {})
    lines.extend(
        [
            "",
            "## 분석 단계 판단",
            "",
            f"- 자연 문장형 행은 실질적으로 `T.Tablet/R.Free`에서만 확인된다. 이 유형은 "
            f"{likely_sentence_category.get('json_count', 0):,}개 이미지, 좌표 재구성 전체 행 "
            f"{likely_sentence_category.get('line_count', 0):,}개, 문장형 자동 후보 "
            f"{likely_sentence_category.get('sentence_like_line_candidate_count', 0):,}개다.",
            "- 전체 행을 모두 문장으로 간주해도 4,000개에 못 미치며 실제로는 문장 조각과 단어 목록이 섞여 있다.",
            "- 따라서 이번 4,000개 평가는 문장 단위보다 R.Free의 bbox 단어/어절 crop 방식이 적합하다.",
            "- bbox crop을 만들고 이미지 무결성을 검사하려면 VS.zip 원천 이미지 다운로드가 필요하다.",
            "- O.Form에는 이름·주소·연락처형 값이 있으므로 최종 평가셋 후보에서 제외하는 것이 안전하다.",
        ]
    )

    for category, stats in analysis["categories"].items():
        lines.extend(["", f"## {category}", ""])
        lines.extend(
            [
                f"- JSON: {stats['json_count']:,}",
                f"- bbox: {stats['bbox_total']:,}",
                f"- bbox 평균/중앙 문자열 길이: {stats['bbox_data_length_mean']:.2f} / {stats['bbox_data_length_median']:.2f}",
                f"- 길이 10 이상 bbox: {stats['length_ge_10_bbox_count']:,} ({stats['length_ge_10_bbox_ratio']:.2%})",
                f"- 공백 포함 bbox: {stats['whitespace_bbox_count']:,} ({stats['whitespace_bbox_ratio']:.2%})",
                f"- 일반 문장부호 포함 bbox: {stats['sentence_punctuation_bbox_count']:,} ({stats['sentence_punctuation_bbox_ratio']:.2%})",
                f"- 좌표 기반 전체 행/여러 bbox 행: {stats['line_count']:,} / {stats['multi_bbox_line_count']:,}",
                f"- 문장형 행 자동 후보: {stats['sentence_like_line_candidate_count']:,} ({stats['sentence_like_line_candidate_ratio']:.2%})",
                "",
                "### 문장형 행 자동 후보 예시",
                "",
            ]
        )
        examples = stats["sentence_like_line_examples_20"]
        if examples:
            for example in examples:
                lines.append(f"- `{example['identifier']}`: {example['text']}")
        else:
            lines.append("- 없음")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    labels_root = args.labels_root.resolve()
    output_dir = args.output_dir.resolve()
    if not labels_root.is_dir():
        raise SystemExit(f"Labels root does not exist: {labels_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis, bbox_candidates, line_candidates = analyze(labels_root, args.seed)
    analysis_path = output_dir / "analysis.json"
    bbox_path = output_dir / "bbox_candidates.jsonl"
    line_path = output_dir / "line_candidates.jsonl"
    report_path = output_dir / "report.md"

    write_json(analysis_path, analysis)
    write_jsonl(bbox_path, bbox_candidates)
    write_jsonl(line_path, line_candidates)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(markdown_report(analysis))

    print(f"Analyzed {analysis['json_files_parsed']:,}/{analysis['json_files_discovered']:,} JSON files")
    print(f"Analysis: {analysis_path}")
    print(f"Report: {report_path}")
    print(f"bbox candidates: {len(bbox_candidates):,} -> {bbox_path}")
    print(f"line candidates: {len(line_candidates):,} -> {line_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
