#!/usr/bin/env python3
"""Build a side-by-side HTML preview of error smoke-test predictions."""

from __future__ import annotations

import argparse
import base64
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "smoke_test"
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "smoke_results_preview.html"

MODEL_FILES = (
    ("qwen35-9b", "Qwen3.5-9B", "qwen35_9b_results.jsonl"),
    ("qwen35-4b", "Qwen3.5-4B", "qwen35_4b_results.jsonl"),
    ("qwen3-vl-4b", "Qwen3-VL-4B", "qwen3_vl_4b_results.jsonl"),
    ("internvl3-8b", "InternVL3-8B", "internvl3_8b_results.jsonl"),
    ("ministral3-8b", "Ministral3-8B", "ministral3_8b_results.jsonl"),
    ("minicpm-v46", "MiniCPM-V-4.6", "minicpm_v46_results.jsonl"),
)

CLASS_LABELS = {
    "error_preserved": "오류 보존",
    "over_corrected": "과교정",
    "recognition_error": "인식 오류",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_results(results_dir: Path) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {}
    expected_ids: list[str] | None = None

    for model_key, _label, filename in MODEL_FILES:
        path = results_dir / filename
        if not path.is_file():
            raise SystemExit(f"Missing result file: {path}")
        rows = read_jsonl(path)
        ids = [str(row["id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"Duplicate IDs in: {path}")
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise SystemExit(f"ID order differs in: {path}")
        results[model_key] = rows

    return results


def image_data_uri(image_path: str) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise SystemExit(f"Missing preview image: {path}")
    mime_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def render_summary(results: dict[str, list[dict[str, Any]]]) -> str:
    rows = []
    for model_key, label, _filename in MODEL_FILES:
        model_rows = results[model_key]
        counts = Counter(row["classification"] for row in model_rows)
        mean_cer = sum(float(row["cer"]) for row in model_rows) / len(model_rows)
        rows.append(
            "<tr>"
            f"<th>{html.escape(label)}</th>"
            f"<td>{counts['error_preserved']}</td>"
            f"<td>{counts['over_corrected']}</td>"
            f"<td>{counts['recognition_error']}</td>"
            f"<td>{mean_cer:.4f}</td>"
            "</tr>"
        )
    return "".join(rows)


def render_prediction(
    model_key: str,
    label: str,
    row: dict[str, Any],
) -> str:
    classification = str(row["classification"])
    class_label = CLASS_LABELS.get(classification, classification)
    prediction = html.escape(str(row["prediction"]))
    return (
        f'<div class="prediction {html.escape(classification)}" '
        f'data-model="{html.escape(model_key)}" '
        f'data-classification="{html.escape(classification)}">'
        f'<div class="prediction-head"><strong>{html.escape(label)}</strong>'
        f'<span class="badge">{html.escape(class_label)}</span>'
        f'<span class="cer">CER {float(row["cer"]):.3f}</span></div>'
        f'<div class="text">{prediction}</div></div>'
    )


def render_cards(
    results: dict[str, list[dict[str, Any]]],
) -> str:
    first_model_key = MODEL_FILES[0][0]
    cards = []
    for index, reference in enumerate(results[first_model_key]):
        predictions = "".join(
            render_prediction(model_key, label, results[model_key][index])
            for model_key, label, _filename in MODEL_FILES
        )
        error_types = ", ".join(str(item) for item in reference["error_type"])
        image_src = image_data_uri(str(reference["image_path"]))
        cards.append(
            '<article class="sample">'
            '<div class="sample-head">'
            f'<div><span class="sample-number">#{index + 1}</span> '
            f'<strong>ID {html.escape(str(reference["id"]))}</strong></div>'
            f'<div class="error-types">{html.escape(error_types)}</div></div>'
            '<div class="reference-grid">'
            f'<a href="{html.escape(image_src)}" target="_blank">'
            f'<img src="{html.escape(image_src)}" loading="lazy" '
            f'alt="{html.escape(str(reference["ground_truth"]))}"></a>'
            '<div class="references">'
            f'<div><strong>원문(오류 포함)</strong><p>{html.escape(str(reference["ground_truth"]))}</p></div>'
            f'<div><strong>교정문</strong><p>{html.escape(str(reference["corrected_text"]))}</p></div>'
            '</div></div>'
            f'<div class="predictions">{predictions}</div>'
            '</article>'
        )
    return "".join(cards)


def build_html(
    results: dict[str, list[dict[str, Any]]],
    output: Path,
) -> str:
    model_options = "".join(
        f'<option value="{html.escape(model_key)}">{html.escape(label)}</option>'
        for model_key, label, _filename in MODEL_FILES
    )
    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>오류 손글씨 smoke test 결과</title>
<style>
:root {{ color-scheme: light; font-family: system-ui, sans-serif; }}
body {{ margin: 0; background: #f4f5f7; color: #1d2433; }}
header {{ position: sticky; top: 0; z-index: 10; padding: 16px 24px; background: #fff; border-bottom: 1px solid #dfe3e8; }}
h1 {{ margin: 0 0 10px; font-size: 22px; }}
.controls {{ display: flex; flex-wrap: wrap; gap: 8px; }}
input, select {{ padding: 8px 10px; border: 1px solid #b9c1cc; border-radius: 6px; background: #fff; }}
#count {{ align-self: center; color: #5b6575; }}
main {{ max-width: 1500px; margin: 20px auto; padding: 0 16px 40px; }}
.summary, .sample {{ background: #fff; border: 1px solid #dfe3e8; border-radius: 10px; box-shadow: 0 2px 8px #18223512; }}
.summary {{ margin-bottom: 18px; padding: 14px; overflow-x: auto; }}
table {{ border-collapse: collapse; min-width: 650px; width: 100%; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid #e7eaf0; text-align: right; }}
th:first-child {{ text-align: left; }}
.sample {{ margin-bottom: 16px; padding: 16px; }}
.sample-head, .prediction-head {{ display: flex; align-items: center; gap: 8px; }}
.sample-head {{ justify-content: space-between; margin-bottom: 12px; }}
.sample-number {{ color: #667085; }}
.error-types {{ color: #667085; font-size: 13px; }}
.reference-grid {{ display: grid; grid-template-columns: minmax(260px, 38%) 1fr; gap: 16px; margin-bottom: 12px; }}
img {{ display: block; width: 100%; max-height: 260px; object-fit: contain; border: 1px solid #e1e5eb; border-radius: 6px; background: #fafafa; }}
.references {{ display: grid; gap: 8px; }}
.references div {{ padding: 10px; background: #f7f8fa; border-radius: 6px; }}
.references p {{ margin: 5px 0 0; font-size: 17px; white-space: pre-wrap; }}
.predictions {{ display: grid; gap: 6px; }}
.prediction {{ padding: 9px 11px; border-left: 5px solid #98a2b3; background: #f8f9fb; border-radius: 5px; }}
.prediction.error_preserved {{ border-color: #16a34a; background: #f0fdf4; }}
.prediction.over_corrected {{ border-color: #dc2626; background: #fff1f2; }}
.prediction.recognition_error {{ border-color: #d97706; background: #fffbeb; }}
.badge {{ padding: 2px 7px; border-radius: 999px; background: #fff; font-size: 12px; }}
.cer {{ margin-left: auto; color: #667085; font-size: 12px; }}
.text {{ margin-top: 5px; white-space: pre-wrap; overflow-wrap: anywhere; }}
.hidden {{ display: none; }}
@media (max-width: 760px) {{
  header {{ position: static; padding: 14px; }}
  .reference-grid {{ grid-template-columns: 1fr; }}
  .sample-head {{ align-items: flex-start; flex-direction: column; }}
}}
</style>
</head>
<body>
<header>
  <h1>오류 손글씨 smoke test 결과</h1>
  <div class="controls">
    <input id="query" type="search" placeholder="ID 또는 문장 검색">
    <select id="model"><option value="">모든 모델</option>{model_options}</select>
    <select id="classification">
      <option value="">모든 판정</option>
      <option value="error_preserved">오류 보존</option>
      <option value="over_corrected">과교정</option>
      <option value="recognition_error">인식 오류</option>
    </select>
    <span id="count"></span>
  </div>
</header>
<main>
  <section class="summary">
    <table>
      <thead><tr><th>모델</th><th>오류 보존</th><th>과교정</th><th>인식 오류</th><th>평균 CER</th></tr></thead>
      <tbody>{render_summary(results)}</tbody>
    </table>
  </section>
  <section id="samples">{render_cards(results)}</section>
</main>
<script>
const query = document.querySelector('#query');
const model = document.querySelector('#model');
const classification = document.querySelector('#classification');
const cards = [...document.querySelectorAll('.sample')];
const count = document.querySelector('#count');
function applyFilters() {{
  const needle = query.value.trim().toLocaleLowerCase();
  let visible = 0;
  for (const card of cards) {{
    const textMatch = !needle || card.textContent.toLocaleLowerCase().includes(needle);
    const predictionRows = [...card.querySelectorAll('.prediction')];
    const resultMatch = predictionRows.some(row =>
      (!model.value || row.dataset.model === model.value) &&
      (!classification.value || row.dataset.classification === classification.value)
    );
    card.classList.toggle('hidden', !(textMatch && resultMatch));
    if (textMatch && resultMatch) visible += 1;
  }}
  count.textContent = `${{visible}} / ${{cards.length}}개`;
}}
query.addEventListener('input', applyFilters);
model.addEventListener('change', applyFilters);
classification.addEventListener('change', applyFilters);
applyFilters();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an HTML preview comparing all smoke-test predictions."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Result directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"HTML output path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output = args.output.resolve()
    results = load_results(results_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(build_html(results, output), encoding="utf-8")
    temporary.replace(output)
    sample_count = len(results[MODEL_FILES[0][0]])
    print(f"Preview: {output}")
    print(f"Samples: {sample_count}, models: {len(MODEL_FILES)}")


if __name__ == "__main__":
    main()
