#!/usr/bin/env python3
"""Build one paper-oriented report from all completed handwriting evaluations."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from run_error_smoke_test import MODEL_SPECS, canonical_text, levenshtein


ROOT = Path(__file__).resolve().parents[1]
REAL_RESULTS_DIR = ROOT / "results" / "handwriting_eval"
SYNTHETIC_RESULTS_DIR = ROOT / "results" / "synthetic_normal_eval"
ERROR_RESULTS_DIR = ROOT / "results" / "error_handwriting_eval"
ERROR_MANIFEST = ROOT / "data" / "benchmark" / "eval_synthetic_error_2000.jsonl"
OUTPUT_JSON = ROOT / "results" / "final_evaluation_summary.json"
OUTPUT_MARKDOWN = ROOT / "results" / "최종_평가_결과.md"

EXPECTED_REAL_COUNT = 2_000
EXPECTED_SYNTHETIC_COUNT = 2_000
EXPECTED_ERROR_COUNT = 1_825
INVALID_MODELS = {
    "minicpm-v46": (
        "프롬프트 반복과 깨진 다국어 출력이 다수 발생하여 추론 설정 수정 후 "
        "재평가가 필요함"
    )
}


@dataclass(frozen=True)
class OcrTrack:
    key: str
    title: str
    results_dir: Path
    expected_count: int


OCR_TRACKS = (
    OcrTrack(
        key="aihub_real",
        title="AI Hub 실제 손글씨",
        results_dir=REAL_RESULTS_DIR,
        expected_count=EXPECTED_REAL_COUNT,
    ),
    OcrTrack(
        key="synthetic_normal",
        title="비오류 합성 손글씨",
        results_dir=SYNTHETIC_RESULTS_DIR,
        expected_count=EXPECTED_SYNTHETIC_COUNT,
    ),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def punctuation_insensitive_text(value: str) -> str:
    """Normalize text while excluding Unicode punctuation from evaluation."""
    normalized = canonical_text(value)
    without_punctuation = "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith("P")
    )
    return re.sub(r"\s+", " ", without_punctuation).strip()


def validate_record_count(
    records: list[dict[str, Any]],
    path: Path,
    expected_count: int,
) -> None:
    if len(records) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} records in {path}, found {len(records)}"
        )
    ids = [str(record["id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate result IDs in {path}")


def summarize_ocr_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_distance = 0
    total_characters = 0
    sample_cers: list[float] = []
    exact_match_count = 0
    total_inference_time = 0.0
    for record in records:
        ground_truth = punctuation_insensitive_text(str(record["ground_truth"]))
        prediction = punctuation_insensitive_text(str(record["prediction"]))
        distance = levenshtein(ground_truth, prediction)
        total_distance += distance
        total_characters += len(ground_truth)
        sample_cers.append(distance / max(1, len(ground_truth)))
        exact_match_count += int(prediction == ground_truth)
        total_inference_time += float(record["inference_time"])
    sample_count = len(records)
    corpus_cer = total_distance / max(1, total_characters)
    return {
        "sample_count": sample_count,
        "mean_cer": sum(sample_cers) / sample_count if sample_count else 0.0,
        "corpus_cer": corpus_cer,
        "character_accuracy": max(0.0, 1.0 - corpus_cer),
        "exact_match_count": exact_match_count,
        "exact_match_rate": exact_match_count / sample_count if sample_count else 0.0,
        "mean_inference_time": (
            total_inference_time / sample_count if sample_count else 0.0
        ),
        "total_inference_time": total_inference_time,
    }


def single_correction_variants(
    ground_truth: str,
    corrected_text: str,
) -> list[str]:
    """Return counterfactuals that apply one aligned correction at a time."""
    matcher = difflib.SequenceMatcher(
        a=ground_truth,
        b=corrected_text,
        autojunk=False,
    )
    variants = []
    for tag, gt_start, gt_end, corrected_start, corrected_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        variants.append(
            ground_truth[:gt_start]
            + corrected_text[corrected_start:corrected_end]
            + ground_truth[gt_end:]
        )
    return variants


def classify_error_prediction(
    sample: dict[str, Any],
    record: dict[str, Any],
) -> dict[str, Any]:
    ground_truth = punctuation_insensitive_text(str(sample["ground_truth"]))
    corrected_text = punctuation_insensitive_text(str(sample["corrected_text"]))
    prediction = punctuation_insensitive_text(str(record["prediction"]))
    if ground_truth == corrected_text:
        raise RuntimeError(
            f"Error sample {sample['id']} becomes unchanged after normalization"
        )

    distance_to_ground_truth = levenshtein(ground_truth, prediction)
    distance_to_corrected = levenshtein(corrected_text, prediction)
    if prediction == ground_truth:
        classification = "error_preserved"
    else:
        variants = single_correction_variants(ground_truth, corrected_text)
        correction_detected = any(
            levenshtein(variant, prediction) < distance_to_ground_truth
            for variant in variants
        )
        classification = (
            "over_corrected" if correction_detected else "other_recognition_error"
        )

    return {
        "classification": classification,
        "ground_truth": ground_truth,
        "corrected_text": corrected_text,
        "prediction": prediction,
        "distance_to_ground_truth": distance_to_ground_truth,
        "distance_to_corrected": distance_to_corrected,
        "cer": distance_to_ground_truth / max(1, len(ground_truth)),
        "exact_corrected_match": prediction == corrected_text,
        "punctuation_only_difference": (
            prediction == ground_truth
            and canonical_text(str(record["prediction"]))
            != canonical_text(str(sample["ground_truth"]))
        ),
    }


def summarize_error_records(
    samples: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    if [sample["id"] for sample in samples] != [record["id"] for record in records]:
        raise RuntimeError("Error result IDs do not match the evaluation manifest order")

    counts = {
        "error_preserved": 0,
        "over_corrected": 0,
        "other_recognition_error": 0,
    }
    total_distance = 0
    total_characters = 0
    total_inference_time = 0.0
    sample_cers: list[float] = []
    exact_corrected_count = 0
    punctuation_only_preserved_count = 0
    partial_overcorrection_count = 0
    by_error_type: dict[str, dict[str, int]] = {}

    for sample, record in zip(samples, records):
        scored = classify_error_prediction(sample, record)
        classification = str(scored["classification"])
        counts[classification] += 1
        total_distance += int(scored["distance_to_ground_truth"])
        total_characters += len(str(scored["ground_truth"]))
        sample_cers.append(float(scored["cer"]))
        total_inference_time += float(record["inference_time"])
        exact_corrected_count += int(scored["exact_corrected_match"])
        punctuation_only_preserved_count += int(scored["punctuation_only_difference"])
        partial_overcorrection_count += int(
            classification == "over_corrected"
            and not scored["exact_corrected_match"]
        )
        for error_type in sample["error_type"]:
            type_counts = by_error_type.setdefault(
                str(error_type),
                {
                    "sample_count": 0,
                    "error_preserved": 0,
                    "over_corrected": 0,
                    "other_recognition_error": 0,
                },
            )
            type_counts["sample_count"] += 1
            type_counts[classification] += 1

    sample_count = len(records)
    corpus_cer = total_distance / max(1, total_characters)
    summaries_by_type: dict[str, dict[str, Any]] = {}
    for error_type, type_counts in sorted(by_error_type.items()):
        type_total = type_counts["sample_count"]
        summaries_by_type[error_type] = {
            **type_counts,
            "error_preserved_rate": type_counts["error_preserved"] / type_total,
            "over_corrected_rate": type_counts["over_corrected"] / type_total,
            "other_recognition_error_rate": (
                type_counts["other_recognition_error"] / type_total
            ),
        }

    return {
        "sample_count": sample_count,
        "mean_cer": sum(sample_cers) / sample_count if sample_count else 0.0,
        "corpus_cer": corpus_cer,
        "character_accuracy": max(0.0, 1.0 - corpus_cer),
        "error_preserved": counts["error_preserved"],
        "error_preserved_rate": counts["error_preserved"] / sample_count,
        "over_corrected": counts["over_corrected"],
        "over_corrected_rate": counts["over_corrected"] / sample_count,
        "other_recognition_error": counts["other_recognition_error"],
        "other_recognition_error_rate": (
            counts["other_recognition_error"] / sample_count
        ),
        "exact_corrected_count": exact_corrected_count,
        "exact_corrected_rate": exact_corrected_count / sample_count,
        "partial_overcorrection_count": partial_overcorrection_count,
        "punctuation_only_preserved_count": punctuation_only_preserved_count,
        "mean_inference_time": total_inference_time / sample_count,
        "total_inference_time": total_inference_time,
        "by_error_type": summaries_by_type,
    }


def with_model_metadata(model_alias: str, summary: dict[str, Any]) -> dict[str, Any]:
    spec = MODEL_SPECS[model_alias]
    invalid_reason = INVALID_MODELS.get(model_alias)
    return {
        "model_alias": model_alias,
        "model_name": spec.model_id,
        "valid": invalid_reason is None,
        "invalid_reason": invalid_reason,
        **summary,
    }


def result_path(results_dir: Path, model_alias: str) -> Path:
    return results_dir / MODEL_SPECS[model_alias].output_name


def build_payload() -> dict[str, Any]:
    ocr_tracks: dict[str, Any] = {}
    ocr_records_by_model: dict[str, list[dict[str, Any]]] = {
        model_alias: [] for model_alias in MODEL_SPECS
    }
    for track in OCR_TRACKS:
        model_summaries = []
        for model_alias in MODEL_SPECS:
            path = result_path(track.results_dir, model_alias)
            records = read_jsonl(path)
            validate_record_count(records, path, track.expected_count)
            ocr_records_by_model[model_alias].extend(records)
            model_summaries.append(
                with_model_metadata(model_alias, summarize_ocr_records(records))
            )
        ocr_tracks[track.key] = {
            "title": track.title,
            "sample_count": track.expected_count,
            "models": model_summaries,
        }

    combined_models = [
        with_model_metadata(
            model_alias,
            summarize_ocr_records(ocr_records_by_model[model_alias]),
        )
        for model_alias in MODEL_SPECS
    ]
    ocr_tracks["combined_general_ocr"] = {
        "title": "일반 OCR 통합",
        "sample_count": EXPECTED_REAL_COUNT + EXPECTED_SYNTHETIC_COUNT,
        "models": combined_models,
    }

    error_samples = read_jsonl(ERROR_MANIFEST)
    validate_record_count(error_samples, ERROR_MANIFEST, EXPECTED_ERROR_COUNT)
    error_models = []
    for model_alias in MODEL_SPECS:
        path = result_path(ERROR_RESULTS_DIR, model_alias)
        records = read_jsonl(path)
        validate_record_count(records, path, EXPECTED_ERROR_COUNT)
        error_models.append(
            with_model_metadata(
                model_alias,
                summarize_error_records(error_samples, records),
            )
        )

    return {
        "generated_on": date.today().isoformat(),
        "scoring": {
            "text_normalization": (
                "Unicode NFC, zero-width/BOM removal, whitespace collapse, and "
                "removal of all Unicode punctuation (category P*)"
            ),
            "spacing": "preserved after collapsing consecutive whitespace",
            "error_classification_precedence": [
                "normalized prediction equals error ground truth => error_preserved",
                "prediction moves closer to any one-correction counterfactual than to error ground truth => over_corrected",
                "otherwise => other_recognition_error",
            ],
            "overcorrection_scope": (
                "At least one injected error corrected; full corrected-sentence "
                "match is not required. A simultaneous unrelated OCR error does "
                "not cancel the over-correction label."
            ),
            "cer": "punctuation-insensitive character error rate",
        },
        "model_validity": {
            model_alias: {
                "valid": model_alias not in INVALID_MODELS,
                "reason": INVALID_MODELS.get(model_alias),
            }
            for model_alias in MODEL_SPECS
        },
        "tracks": {
            **ocr_tracks,
            "synthetic_error": {
                "title": "오류 주입 합성 손글씨",
                "sample_count": EXPECTED_ERROR_COUNT,
                "models": error_models,
            },
        },
    }


def model_label(summary: dict[str, Any]) -> str:
    label = str(summary["model_name"])
    return f"{label} ⚠" if not summary["valid"] else label


def percent(value: float) -> str:
    return f"{value:.2%}"


def find_model(track: dict[str, Any], model_alias: str) -> dict[str, Any]:
    return next(
        summary
        for summary in track["models"]
        if summary["model_alias"] == model_alias
    )


def render_ocr_table(track: dict[str, Any]) -> list[str]:
    lines = [
        f"### {track['title']} ({track['sample_count']:,}건)",
        "",
        "| 모델 | Mean CER | Corpus CER | 문자 정확도 | 완전 일치 | 평균 추론 시간 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in track["models"]:
        lines.append(
            f"| {model_label(summary)} | {summary['mean_cer']:.4f} | "
            f"{summary['corpus_cer']:.4f} | "
            f"{percent(summary['character_accuracy'])} | "
            f"{summary['exact_match_count']} "
            f"({percent(summary['exact_match_rate'])}) | "
            f"{summary['mean_inference_time']:.3f}s |"
        )
    return lines


def render_error_table(track: dict[str, Any]) -> list[str]:
    lines = [
        f"### {track['title']} ({track['sample_count']:,}건)",
        "",
        "| 모델 | Mean CER | 오류 보존 | 과교정(자동 검출) | 기타 문자 인식 오류 | 교정문 전체 일치 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in track["models"]:
        lines.append(
            f"| {model_label(summary)} | {summary['mean_cer']:.4f} | "
            f"{summary['error_preserved']} "
            f"({percent(summary['error_preserved_rate'])}) | "
            f"{summary['over_corrected']} "
            f"({percent(summary['over_corrected_rate'])}) | "
            f"{summary['other_recognition_error']} "
            f"({percent(summary['other_recognition_error_rate'])}) | "
            f"{summary['exact_corrected_count']} "
            f"({percent(summary['exact_corrected_rate'])}) |"
        )
    return lines


def render_compact_table(payload: dict[str, Any]) -> list[str]:
    tracks = payload["tracks"]
    real = tracks["aihub_real"]
    synthetic = tracks["synthetic_normal"]
    combined = tracks["combined_general_ocr"]
    error = tracks["synthetic_error"]
    lines = [
        "## 논문용 핵심 표",
        "",
        "| 모델 | 실제 CER | 합성 CER | 통합 CER | 오류 보존 | 과교정(자동 검출) | 기타 인식 오류 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_alias in MODEL_SPECS:
        real_summary = find_model(real, model_alias)
        synthetic_summary = find_model(synthetic, model_alias)
        combined_summary = find_model(combined, model_alias)
        error_summary = find_model(error, model_alias)
        lines.append(
            f"| {model_label(real_summary)} | {real_summary['mean_cer']:.4f} | "
            f"{synthetic_summary['mean_cer']:.4f} | "
            f"{combined_summary['mean_cer']:.4f} | "
            f"{percent(error_summary['error_preserved_rate'])} | "
            f"{percent(error_summary['over_corrected_rate'])} | "
            f"{percent(error_summary['other_recognition_error_rate'])} |"
        )
    return lines


def render_markdown(payload: dict[str, Any]) -> str:
    tracks = payload["tracks"]
    lines = [
        "# 최종 손글씨 평가 결과",
        "",
        f"- 생성일: `{payload['generated_on']}`",
        "- 일반 OCR: AI Hub 실제 손글씨 2,000건과 비오류 합성 손글씨 2,000건",
        "- 오류 보존: 오류 주입 합성 손글씨 1,825건",
        "- 모든 수치는 문장부호를 제외하고 다시 계산함. 띄어쓰기는 유지함.",
        "- ⚠ 표시는 추론 출력 이상으로 최종 비교에서 제외해야 하는 모델임.",
        "",
        *render_compact_table(payload),
        "",
        "## 일반 OCR 상세 결과",
        "",
    ]
    for key in ("aihub_real", "synthetic_normal", "combined_general_ocr"):
        lines.extend(render_ocr_table(tracks[key]))
        lines.append("")

    lines.extend(
        [
            "## 오류 보존 상세 결과",
            "",
            *render_error_table(tracks["synthetic_error"]),
            "",
            "과교정은 교정문 전체 일치뿐 아니라 주입 오류 중 하나라도 "
            "주석된 교정 방향으로 수정한 경우를 포함한다. 따라서 `교정문 전체 "
            "일치`는 과교정의 부분집합이다.",
            "주석된 교정형과 다른 자연스러운 제3의 표현은 문자열 기반으로 "
            "확정할 수 없으므로 자동 과교정 수치에 포함하지 않았다. 해당 수치는 "
            "재현 가능한 자동 검출 하한이며, 제3의 표현까지 포함하려면 별도 수동 "
            "판정이 필요하다.",
            "",
            "## 판정 기준",
            "",
            "1. 문장부호를 제거한 출력이 오류 포함 정답과 일치하면 `오류 보존`.",
            "2. 그 외에 출력이 오류 중 하나를 주석된 교정 방향으로 수정하면 `과교정`.",
            "3. 나머지 문자 누락·추가·오인식은 `기타 문자 인식 오류`.",
            "4. 여러 오류 중 하나만 고치거나 다른 OCR 오류가 동시에 발생해도 "
            "과교정을 우선 판정함.",
            "",
            "문장부호 차이만 있는 예측은 오류 보존 또는 완전 일치로 처리한다. "
            "띄어쓰기 차이는 평가에 포함한다.",
            "",
            "## 해석 시 주의사항",
            "",
            "- 오류 유형별 성능은 한 문장에 여러 오류 유형이 함께 존재하므로 "
            "단순 표본 분할로 해석하면 안 됨.",
            "- MiniCPM-V-4.6은 세 트랙 모두 프롬프트 반복과 깨진 출력이 많아 "
            "현재 수치를 모델 성능으로 해석할 수 없음.",
            "- `results/error_handwriting_eval`의 기존 요약은 교정문 전체 일치만 "
            "과교정으로 보던 이전 기준이므로 본 보고서의 재집계 수치를 사용해야 함.",
            "- 본 보고서의 과교정은 주석된 교정 방향을 자동 검출한 수치임. "
            "주석과 다른 자연스러운 재표현은 수동 판정 전까지 기타 문자 인식 "
            "오류에 남아 있음.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=OUTPUT_MARKDOWN)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_payload()
    output_json = args.output_json.expanduser().resolve()
    output_markdown = args.output_markdown.expanduser().resolve()
    atomic_write_text(
        output_json,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    atomic_write_text(output_markdown, render_markdown(payload))
    print(f"JSON: {output_json}")
    print(f"Markdown: {output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
