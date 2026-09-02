#!/usr/bin/env python3
"""Create a human-review Preview for line-level GT reconstruction in AI Hub 605 R.Free."""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
import shutil
import statistics
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from build_aihub605_eval import (
    DEFAULT_IMAGES_ROOT,
    DEFAULT_LABELS_ROOT,
    LabelPage,
    atomic_write_json,
    atomic_write_text,
    index_images,
    load_label_pages,
    stable_seed,
    validate_image_matching,
)


DEFAULT_OUTPUT_DIR = Path("data/benchmark/aihub_line_validation_preview")
HANGUL_RE = re.compile(r"[가-힣]")
DIGIT_RE = re.compile(r"\d")
SENTENCE_PUNCTUATION_RE = re.compile(r"[.!?。！？…]")
PREDICATE_ENDINGS = (
    "습니다",
    "습니까",
    "입니다",
    "였습니다",
    "하였다",
    "했다",
    "한다",
    "된다",
    "이다",
    "였다",
    "있다",
    "없다",
    "하며",
    "이며",
    "면서",
    "하고",
    "되고",
    "라고",
    "다고",
    "하는",
    "하여",
    "해서",
    "되어",
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


@dataclass(frozen=True)
class Box:
    text: str
    bbox_id: Any
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
    def width(self) -> float:
        return max(1.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(1.0, self.ymax - self.ymin)


@dataclass
class LineRecord:
    identifier: str
    category: str
    json_path: str
    line_index: int
    tokens: list[str]
    token_bboxes: list[list[int]]
    ground_truth: str
    bbox_ids: list[Any]
    crop_bbox: list[int]
    bbox_count: int
    character_length: int
    hangul_ratio: float
    digit_ratio: float
    predicate_ending_count: int
    grammatical_ending_count: int
    sentence_punctuation: bool
    bbox_id_order_matches_x_order: bool | None
    median_horizontal_gap: float | None
    median_gap_to_height_ratio: float | None
    horizontal_overlap_count: int
    sentence_score: int
    automatic_sentence_candidate: bool
    uncovered_dark_pixel_ratio: float | None = None
    visual_gt_coverage_pass: bool | None = None
    recommended_sentence_candidate: bool = False
    source_image: str = ""
    image_path: str = ""
    source: str = "aihub_real"
    selection_kind: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS_ROOT)
    parser.add_argument("--images-root", type=Path, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preview-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--padding", type=int, default=12)
    return parser.parse_args()


def parse_box(raw: Any) -> Box | None:
    if not isinstance(raw, dict):
        return None
    text = raw.get("data")
    x_values = raw.get("x")
    y_values = raw.get("y")
    if not isinstance(text, str) or not text.strip():
        return None
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
    if max(xs) <= min(xs) or max(ys) <= min(ys):
        return None
    return Box(
        text=text.strip(),
        bbox_id=raw.get("id"),
        xmin=min(xs),
        ymin=min(ys),
        xmax=max(xs),
        ymax=max(ys),
    )


def group_boxes_into_lines(boxes: list[Box]) -> list[list[Box]]:
    """Cluster by y-center using the page's median bbox height as the tolerance."""
    if not boxes:
        return []
    median_height = statistics.median(box.height for box in boxes)
    tolerance = max(8.0, median_height * 0.60)
    rows: list[dict[str, Any]] = []
    for box in sorted(boxes, key=lambda item: (item.cy, item.cx)):
        possible = [
            (abs(box.cy - row["center"]), row)
            for row in rows
            if abs(box.cy - row["center"]) <= tolerance
        ]
        if possible:
            _, row = min(possible, key=lambda item: item[0])
            row["boxes"].append(box)
            row["center"] = statistics.median(item.cy for item in row["boxes"])
        else:
            rows.append({"center": box.cy, "boxes": [box]})
    rows.sort(key=lambda row: row["center"])
    return [sorted(row["boxes"], key=lambda item: (item.xmin, str(item.bbox_id))) for row in rows]


def ends_with(token: str, endings: tuple[str, ...]) -> bool:
    cleaned = SENTENCE_PUNCTUATION_RE.sub("", token.strip())
    return any(cleaned.endswith(ending) for ending in endings)


def numeric_ids_increasing(ids: list[Any]) -> bool | None:
    if not all(isinstance(value, int) for value in ids):
        return None
    return ids == sorted(ids)


def build_line_record(
    page: LabelPage,
    line: list[Box],
    line_index: int,
    padding: int,
) -> LineRecord:
    tokens = [box.text for box in line]
    text = " ".join(tokens)
    compact = re.sub(r"\s", "", text)
    hangul_count = len(HANGUL_RE.findall(compact))
    digit_count = len(DIGIT_RE.findall(compact))
    denominator = max(1, len(compact))
    predicate_count = sum(ends_with(token, PREDICATE_ENDINGS) for token in tokens)
    grammatical_count = sum(ends_with(token, GRAMMATICAL_ENDINGS) for token in tokens)
    sentence_punctuation = bool(SENTENCE_PUNCTUATION_RE.search(text))
    gaps = [right.xmin - left.xmax for left, right in zip(line, line[1:])]
    positive_gaps = [gap for gap in gaps if gap >= 0]
    median_gap = statistics.median(positive_gaps) if positive_gaps else None
    median_height = statistics.median(box.height for box in line)
    normalized_gap = median_gap / median_height if median_gap is not None else None
    overlap_count = sum(gap < 0 for gap in gaps)
    ids = [box.bbox_id for box in line]

    score = 0
    score += 1 if len(tokens) >= 3 else 0
    score += 1 if len(compact) >= 10 else 0
    score += 1 if hangul_count / denominator >= 0.60 else 0
    score += 1 if digit_count / denominator <= 0.20 else 0
    score += min(2, grammatical_count)
    score += 3 if predicate_count else 0
    score += 2 if sentence_punctuation else 0
    score += 1 if normalized_gap is not None and normalized_gap <= 4.0 else 0
    automatic_candidate = (
        len(tokens) >= 3
        and len(compact) >= 10
        and hangul_count / denominator >= 0.60
        and digit_count / denominator <= 0.20
        and ((predicate_count >= 1 and grammatical_count >= 2) or sentence_punctuation)
    )
    crop_bbox = [
        max(0, math.floor(min(box.xmin for box in line)) - padding),
        max(0, math.floor(min(box.ymin for box in line)) - padding),
        min(page.width, math.ceil(max(box.xmax for box in line)) + padding),
        min(page.height, math.ceil(max(box.ymax for box in line)) + padding),
    ]
    return LineRecord(
        identifier=page.identifier,
        category=page.category,
        json_path=str(page.json_path),
        line_index=line_index,
        tokens=tokens,
        token_bboxes=[
            [
                math.floor(box.xmin),
                math.floor(box.ymin),
                math.ceil(box.xmax),
                math.ceil(box.ymax),
            ]
            for box in line
        ],
        ground_truth=text,
        bbox_ids=ids,
        crop_bbox=crop_bbox,
        bbox_count=len(line),
        character_length=len(text),
        hangul_ratio=round(hangul_count / denominator, 6),
        digit_ratio=round(digit_count / denominator, 6),
        predicate_ending_count=predicate_count,
        grammatical_ending_count=grammatical_count,
        sentence_punctuation=sentence_punctuation,
        bbox_id_order_matches_x_order=numeric_ids_increasing(ids),
        median_horizontal_gap=round(median_gap, 4) if median_gap is not None else None,
        median_gap_to_height_ratio=round(normalized_gap, 4) if normalized_gap is not None else None,
        horizontal_overlap_count=overlap_count,
        sentence_score=score,
        automatic_sentence_candidate=automatic_candidate,
    )


def add_visual_gt_coverage(
    by_category: dict[str, list[LineRecord]],
    image_matches: dict[str, Path],
    dark_threshold: int = 200,
    bbox_padding: int = 6,
    maximum_uncovered_ratio: float = 0.05,
) -> dict[str, Any]:
    """Detect visible handwriting not covered by any label bbox on Tablet rows."""
    tablet_records = by_category.get("T.Tablet/R.Free", [])
    by_identifier: dict[str, list[LineRecord]] = defaultdict(list)
    for record in tablet_records:
        by_identifier[record.identifier].append(record)

    no_dark_pixel_count = 0
    for page_index, (identifier, records) in enumerate(sorted(by_identifier.items()), start=1):
        with Image.open(image_matches[identifier]) as image:
            grayscale = image.convert("L")
            page_width, page_height = grayscale.size
            writing_left = max(0, math.floor(page_width * 0.06))
            writing_right = min(page_width, math.ceil(page_width * 0.94))
            for record in records:
                line_top = max(0, min(box[1] for box in record.token_bboxes) - bbox_padding)
                line_bottom = min(
                    page_height, max(box[3] for box in record.token_bboxes) + bbox_padding
                )
                band = np.asarray(
                    grayscale.crop((writing_left, line_top, writing_right, line_bottom))
                )
                dark_pixels = band < dark_threshold
                total_dark = int(dark_pixels.sum())
                if total_dark == 0:
                    record.uncovered_dark_pixel_ratio = None
                    record.visual_gt_coverage_pass = False
                    record.recommended_sentence_candidate = False
                    no_dark_pixel_count += 1
                    continue
                covered = np.zeros(band.shape, dtype=bool)
                for left, top, right, bottom in record.token_bboxes:
                    relative_left = max(0, left - bbox_padding - writing_left)
                    relative_right = min(band.shape[1], right + bbox_padding - writing_left)
                    relative_top = max(0, top - bbox_padding - line_top)
                    relative_bottom = min(band.shape[0], bottom + bbox_padding - line_top)
                    if relative_right > relative_left and relative_bottom > relative_top:
                        covered[relative_top:relative_bottom, relative_left:relative_right] = True
                uncovered_dark = int(np.logical_and(dark_pixels, np.logical_not(covered)).sum())
                uncovered_ratio = uncovered_dark / total_dark
                record.uncovered_dark_pixel_ratio = round(uncovered_ratio, 6)
                record.visual_gt_coverage_pass = uncovered_ratio <= maximum_uncovered_ratio
                record.recommended_sentence_candidate = (
                    record.automatic_sentence_candidate and record.visual_gt_coverage_pass
                )
        if page_index % 100 == 0:
            print(
                f"Tablet visual GT coverage checked: {page_index:,}/{len(by_identifier):,}",
                flush=True,
            )

    covered_records = [record for record in tablet_records if record.visual_gt_coverage_pass]
    return {
        "method": (
            "Tablet 행 높이 구간에서 grayscale<200 픽셀을 필기로 보고, 각 라벨 bbox를 6px 확장한 "
            "영역 밖 필기 픽셀 비율을 계산"
        ),
        "dark_threshold": dark_threshold,
        "bbox_padding_pixels": bbox_padding,
        "maximum_uncovered_dark_pixel_ratio": maximum_uncovered_ratio,
        "tablet_line_count": len(tablet_records),
        "visual_gt_coverage_pass_count": len(covered_records),
        "visual_gt_coverage_fail_count": len(tablet_records) - len(covered_records),
        "no_dark_pixel_count": no_dark_pixel_count,
        "automatic_sentence_candidate_count": sum(
            record.automatic_sentence_candidate for record in tablet_records
        ),
        "recommended_sentence_candidate_count": sum(
            record.recommended_sentence_candidate for record in tablet_records
        ),
    }


def extract_all_lines(
    pages: list[LabelPage], padding: int
) -> tuple[dict[str, list[LineRecord]], dict[str, Any]]:
    by_category: dict[str, list[LineRecord]] = defaultdict(list)
    page_counts: Counter[str] = Counter()
    invalid_bbox_counts: Counter[str] = Counter()
    for page in pages:
        if not page.category.endswith("/R.Free"):
            continue
        page_counts[page.category] += 1
        parsed_boxes = []
        for raw in page.bboxes:
            box = parse_box(raw)
            if box is None:
                invalid_bbox_counts[page.category] += 1
            else:
                parsed_boxes.append(box)
        for line_index, line in enumerate(group_boxes_into_lines(parsed_boxes), start=1):
            by_category[page.category].append(
                build_line_record(page, line, line_index, padding)
            )

    category_diagnostics = {}
    for category, records in sorted(by_category.items()):
        multi = [record for record in records if record.bbox_count >= 2]
        structural = [
            record
            for record in records
            if record.bbox_count >= 3
            and len(re.sub(r"\s", "", record.ground_truth)) >= 10
            and record.hangul_ratio >= 0.60
            and record.digit_ratio <= 0.20
        ]
        monotonic_known = [
            record for record in multi if record.bbox_id_order_matches_x_order is not None
        ]
        category_diagnostics[category] = {
            "json_count": page_counts[category],
            "line_count": len(records),
            "multi_bbox_line_count": len(multi),
            "structural_line_count": len(structural),
            "automatic_sentence_candidate_count": sum(
                record.automatic_sentence_candidate for record in records
            ),
            "bbox_id_order_matches_x_order_count": sum(
                record.bbox_id_order_matches_x_order is True for record in monotonic_known
            ),
            "bbox_id_order_known_count": len(monotonic_known),
            "bbox_id_order_match_ratio": round(
                sum(record.bbox_id_order_matches_x_order is True for record in monotonic_known)
                / max(1, len(monotonic_known)),
                6,
            ),
            "invalid_bbox_count": invalid_bbox_counts[category],
            "bbox_per_line_mean": round(
                statistics.fmean(record.bbox_count for record in records), 4
            ),
            "character_length_mean": round(
                statistics.fmean(record.character_length for record in records), 4
            ),
            "character_length_max": max(record.character_length for record in records),
        }
    diagnostics = {
        "grouping_method": (
            "bbox y-center를 페이지 bbox 높이 중앙값의 0.60배(최소 8px) 허용치로 군집화한 뒤, "
            "각 행을 xmin 오름차순으로 정렬"
        ),
        "ground_truth_reconstruction": "동일 행의 bbox.data를 xmin 순으로 ASCII 공백 하나로 연결",
        "automatic_candidate_warning": (
            "휴리스틱은 검토 후보를 좁히기 위한 것이며 자연스러운 문장이나 정확한 띄어쓰기를 보장하지 않는다."
        ),
        "categories": category_diagnostics,
    }
    return dict(by_category), diagnostics


def choose_preview_records(
    by_category: dict[str, list[LineRecord]], preview_count: int, seed: int
) -> tuple[list[LineRecord], dict[str, int]]:
    categories = sorted(by_category)
    if not categories:
        raise RuntimeError("No R.Free lines were found")
    base = preview_count // len(categories)
    quotas = {category: base for category in categories}
    for category in categories[: preview_count - base * len(categories)]:
        quotas[category] += 1

    selected: list[LineRecord] = []
    selected_keys: set[tuple[str, int]] = set()
    used_identifiers: set[str] = set()
    selection_counts: Counter[str] = Counter()
    for category in categories:
        quota = quotas[category]
        structural = [
            record
            for record in by_category[category]
            if record.bbox_count >= 3
            and len(re.sub(r"\s", "", record.ground_truth)) >= 10
            and record.hangul_ratio >= 0.60
            and record.digit_ratio <= 0.20
        ]
        high_ranked = sorted(
            structural,
            key=lambda record: (
                -int(record.recommended_sentence_candidate),
                -int(record.automatic_sentence_candidate),
                -record.sentence_score,
                -record.bbox_count,
                record.identifier,
                record.line_index,
            ),
        )
        high_target = quota // 2
        category_selected = 0
        for record in high_ranked:
            if category_selected >= high_target:
                break
            key = (record.identifier, record.line_index)
            if record.identifier in used_identifiers or key in selected_keys:
                continue
            record.selection_kind = "high_score"
            selected.append(record)
            selected_keys.add(key)
            used_identifiers.add(record.identifier)
            category_selected += 1
            selection_counts[f"{category}:high_score"] += 1

        remaining = [
            record
            for record in structural
            if (record.identifier, record.line_index) not in selected_keys
            and record.identifier not in used_identifiers
        ]
        random.Random(stable_seed(seed, f"preview:{category}")).shuffle(remaining)
        for record in remaining:
            if category_selected >= quota:
                break
            key = (record.identifier, record.line_index)
            if record.identifier in used_identifiers or key in selected_keys:
                continue
            record.selection_kind = "random_structural"
            selected.append(record)
            selected_keys.add(key)
            used_identifiers.add(record.identifier)
            category_selected += 1
            selection_counts[f"{category}:random_structural"] += 1
        if category_selected != quota:
            raise RuntimeError(
                f"Could select only {category_selected}/{quota} Preview lines from {category}"
            )
    selected.sort(key=lambda record: (record.category, record.selection_kind, record.identifier))
    return selected, dict(sorted(selection_counts.items()))


def create_crops(
    records: list[LineRecord],
    image_matches: dict[str, Path],
    output_dir: Path,
) -> list[LineRecord]:
    final_crop_dir = output_dir / "line_crops"
    if final_crop_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing crop directory: {final_crop_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".line_crops_", dir=output_dir))
    try:
        for sequence, record in enumerate(records, start=1):
            source_path = image_matches[record.identifier]
            crop_name = f"{sequence:03d}_{record.identifier}_line{record.line_index:02d}.png"
            crop_path = staging / crop_name
            try:
                with Image.open(source_path) as source:
                    source.load()
                    crop = source.crop(tuple(record.crop_bbox))
                    if crop.width <= 0 or crop.height <= 0:
                        raise ValueError("empty crop")
                    crop.save(crop_path, format="PNG")
                with Image.open(crop_path) as verification:
                    verification.load()
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                raise RuntimeError(f"Failed to crop {record.identifier}: {exc}") from exc
            record.source_image = str(source_path.resolve())
            record.image_path = str((final_crop_dir / crop_name).resolve())
        staging.rename(final_crop_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return records


def write_jsonl(path: Path, records: list[LineRecord]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    temporary.replace(path)


def preview_html(records: list[LineRecord], output_dir: Path, seed: int) -> str:
    cards = []
    for record in records:
        relative_path = Path(record.image_path).relative_to(output_dir.resolve()).as_posix()
        tokens = " | ".join(record.tokens)
        cards.append(
            "<article class=\"card\">"
            f"<img src=\"{html.escape(relative_path)}\" loading=\"lazy\" "
            f"alt=\"{html.escape(record.ground_truth)}\">"
            f"<h2>{html.escape(record.ground_truth)}</h2>"
            f"<p class=\"tokens\"><strong>bbox.data:</strong> {html.escape(tokens)}</p>"
            "<fieldset><legend>사람 검토</legend>"
            "<label><input type=\"checkbox\"> 자연스러운 한 문장/행</label> "
            "<label><input type=\"checkbox\"> GT·띄어쓰기 정확</label> "
            "<label><input type=\"checkbox\"> crop 완전</label>"
            "</fieldset>"
            f"<dl><dt>category</dt><dd>{html.escape(record.category)}</dd>"
            f"<dt>selection</dt><dd>{html.escape(record.selection_kind)}</dd>"
            f"<dt>자동 문장 후보 / 점수</dt><dd>{record.automatic_sentence_candidate} / {record.sentence_score}</dd>"
            f"<dt>미라벨 필기 비율 / GT coverage</dt><dd>{record.uncovered_dark_pixel_ratio} / {record.visual_gt_coverage_pass}</dd>"
            f"<dt>최종 자동 추천 후보</dt><dd>{record.recommended_sentence_candidate}</dd>"
            f"<dt>bbox 수 / IDs</dt><dd>{record.bbox_count} / {html.escape(json.dumps(record.bbox_ids))}</dd>"
            f"<dt>ID 순서=좌→우</dt><dd>{record.bbox_id_order_matches_x_order}</dd>"
            f"<dt>중앙 gap / 높이 비율</dt><dd>{record.median_horizontal_gap} / {record.median_gap_to_height_ratio}</dd>"
            f"<dt>source</dt><dd>{html.escape(record.source_image)}</dd></dl>"
            "</article>"
        )
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Hub 605 R.Free line validation</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f3f5f7;color:#17202a}}
.notice{{background:#fff5cc;border:1px solid #e2c455;padding:12px;border-radius:8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px;margin-top:20px}}
.card{{background:white;border:1px solid #d8dde3;border-radius:10px;padding:14px;box-shadow:0 2px 8px #0001}}
img{{display:block;max-width:100%;max-height:240px;margin:auto;background:#eee}}
h2{{font-size:1.08rem;margin:12px 0}}.tokens{{font-size:.9rem;overflow-wrap:anywhere}}
fieldset{{border:1px solid #ccd3da;margin:12px 0}}dl{{font-size:.82rem;overflow-wrap:anywhere}}
dt{{font-weight:700;margin-top:6px}}dd{{margin-left:0;color:#445}}
</style></head><body>
<h1>AI Hub 605 R.Free line crop Preview — {len(records)}개</h1>
<p class="notice">체크박스는 브라우저에서 임시 검토용이며 자동 저장되지 않습니다. 각 crop이 한 문장/행인지, bbox.data를 공백으로 연결한 GT가 실제 이미지와 일치하는지 확인하세요.</p>
<p>source=<code>aihub_real</code>, seed={seed}</p>
<main class="grid">{''.join(cards)}</main></body></html>\n"""


def main() -> int:
    args = parse_args()
    if not 30 <= args.preview_count <= 50:
        raise SystemExit("--preview-count must be between 30 and 50")
    labels_root = args.labels_root.resolve()
    images_root = args.images_root.resolve()
    output_dir = args.output_dir.resolve()
    if not labels_root.is_dir():
        raise SystemExit(f"Labels root does not exist: {labels_root}")
    if not images_root.is_dir():
        raise SystemExit(f"Images root does not exist: {images_root}")
    artifact_paths = (
        output_dir / "line_preview_50.jsonl",
        output_dir / "line_grouping_diagnostics.json",
        output_dir / "sample_preview_lines.html",
    )
    existing = [path for path in artifact_paths if path.exists()]
    if existing:
        raise SystemExit(
            "Refusing to overwrite existing Preview artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    print("Loading Validation labels...", flush=True)
    pages, label_errors = load_label_pages(labels_root)
    rfree_pages = [page for page in pages if page.category.endswith("/R.Free")]
    image_index, image_suffixes = index_images(images_root)
    print(
        f"R.Free labels={len(rfree_pages):,}, images={sum(len(v) for v in image_index.values()):,}",
        flush=True,
    )
    match_report, image_matches, passed = validate_image_matching(
        rfree_pages, label_errors, image_index, image_suffixes
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "image_match_report.json", match_report)
    if not passed:
        raise SystemExit("Image/label matching failed; line grouping was not started")

    print("Grouping R.Free bboxes into lines...", flush=True)
    by_category, diagnostics = extract_all_lines(rfree_pages, args.padding)
    diagnostics["visual_gt_coverage"] = add_visual_gt_coverage(
        by_category, image_matches
    )
    selected, selection_counts = choose_preview_records(
        by_category, args.preview_count, args.seed
    )
    selected = create_crops(selected, image_matches, output_dir)
    diagnostics["preview"] = {
        "preview_count": len(selected),
        "selection_counts": selection_counts,
        "unique_source_image_count": len({record.identifier for record in selected}),
        "source": "aihub_real",
        "seed": args.seed,
        "padding_pixels": args.padding,
    }
    diagnostics["image_label_matching_summary"] = {
        key: match_report[key]
        for key in (
            "passed",
            "label_json_count",
            "indexed_image_count",
            "unique_match_count",
            "missing_identifier_count",
            "orphan_image_stem_count",
            "dimension_mismatch_count",
            "unreadable_image_count",
        )
    }
    write_jsonl(output_dir / "line_preview_50.jsonl", selected)
    atomic_write_json(output_dir / "line_grouping_diagnostics.json", diagnostics)
    atomic_write_text(
        output_dir / "sample_preview_lines.html",
        preview_html(selected, output_dir, args.seed),
    )
    print(f"Preview completed: {output_dir / 'sample_preview_lines.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
