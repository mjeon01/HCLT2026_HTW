---
base_model: Qwen/Qwen3.5-4B
library_name: peft
pipeline_tag: image-text-to-text
language:
- ko
license: apache-2.0
tags:
- ocr
- handwriting
- korean
- vision-language-model
- lora
---

# 한국어 필기체 OCR 과교정 완화

<p align="center">
  <a href="https://huggingface.co/Mun2/qwen3.5-4b-korean-handwriting-ocr-overcorrection">
    <img src="https://img.shields.io/badge/Hugging%20Face-Model-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face model">
  </a>
  <img src="https://img.shields.io/badge/Base-Qwen3.5--4B-7B61FF" alt="Qwen3.5-4B">
  <img src="https://img.shields.io/badge/Language-Korean-0A84FF" alt="Korean">
</p>

한국어 필기체 OCR 모델이 이미지에 적힌 오류를 임의로 고쳐 쓰는 **과교정(overcorrection)** 현상을 측정하고 완화하기 위한 데이터 구축·학습·평가 파이프라인입니다. `Qwen/Qwen3.5-4B`에 정상 문장 20,000회와 오류 포함 문장 20,000회를 동일 비율로 노출하고, vision projector와 LoRA를 함께 학습했습니다.

- 🤗 **학습 모델:** [Mun2/qwen3.5-4b-korean-handwriting-ocr-overcorrection](https://huggingface.co/Mun2/qwen3.5-4b-korean-handwriting-ocr-overcorrection)
- 💻 **프로젝트 코드:** [davemaxuell/HCLT2026_HTW](https://github.com/davemaxuell/HCLT2026_HTW)
- ⚙️ **학습 설정:** [`configs/overcorrection_50_50_40000.yaml`](configs/overcorrection_50_50_40000.yaml)
- 📊 **평가 요약:** [`results/final_evaluation_summary.json`](results/final_evaluation_summary.json)

## 핵심 아이디어

일반 OCR은 정답 문장을 자연스럽게 복원하는 것만으로 충분하지만, 오류 보존 OCR은 이미지에 실제로 쓰인 비표준 표현까지 그대로 전사해야 합니다. 이 프로젝트는 다음 원칙으로 학습합니다.

1. 정상 문장과 오류 포함 문장을 50:50으로 구성합니다.
2. 오류 데이터에서도 교정문이 아니라 이미지에 렌더링된 원문을 정답으로 사용합니다.
3. 프롬프트 토큰은 `-100`으로 마스킹하고 assistant 응답에만 loss를 계산합니다.
4. vision encoder는 동결하고 vision merger(projector)와 language-model LoRA만 학습합니다.

사용한 instruction은 다음과 같습니다.

> 이미지에 작성된 한국어 문장을 맞춤법이나 문법을 수정하지 말고 그대로 전사하세요.

## 파이프라인

### 1. ONE-DM 기반 합성 필기체 구축

합성 한국어 문장과 필체 샘플에서 content·style 특징을 분리하고, style-content fusion을 거쳐 다양한 합성 필기체 이미지를 생성합니다.

<p align="center">
  <img src="assets/ONE-DM기반%20합성데이터구축.png" width="100%" alt="ONE-DM 기반 합성 필기체 데이터 구축 파이프라인">
</p>

### 2. Qwen3.5-4B VLM fine-tuning

합성 필기체 이미지와 전사 instruction을 입력으로 사용합니다. vision encoder는 동결하고 vision merger와 attention·MLP의 LoRA 파라미터를 학습합니다.

<p align="center">
  <img src="assets/VLM%20FINE-TUNING.png" width="480" alt="Qwen3.5-4B VLM fine-tuning 파이프라인">
</p>

## 학습 구성

| 항목 | 설정 |
| --- | --- |
| 베이스 모델 | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B) |
| 학습 방식 | vision projector + LoRA |
| 데이터 노출 | 정상 20,000 + 오류 포함 20,000 |
| 최적화 스텝 | 2,500 |
| 유효 배치 크기 | 16 (`batch_size=1`, accumulation 16) |
| 학습률 | projector `1e-3`, LoRA `1e-4` |
| LoRA | rank 64, alpha 128, dropout 0.05 |
| 정밀도 | BF16 |
| seed | 42 |
| 검증 loss | 0.08388 |

원본 pool은 정상 33,899개와 오류 포함 16,432개입니다. 오류 포함 데이터는 20,000회 노출을 맞추기 위해 seed 42로 결정론적으로 재표집했습니다. 학습이 완료된 체크포인트에는 LoRA adapter, vision projector, processor가 포함됩니다.

## 평가 결과

CER은 Unicode NFC 정규화, 연속 공백 축약, 문장부호 제거 후 계산한 문장별 평균 character error rate이며 낮을수록 좋습니다.

| 평가 데이터 | 샘플 수 | Base mean CER | Fine-tuned mean CER |
| --- | ---: | ---: | ---: |
| AI Hub 실제 필기체 | 2,000 | 0.1438 | 0.1967 |
| 정상 합성 필기체 | 2,000 | 0.0214 | 0.0036* |
| 오류 포함 합성 필기체 | 1,825 | 0.0563 | 0.0182* |

오류 포함 합성 필기체의 문장 단위 진단 결과입니다.

| 오류 보존 | 과교정 | 기타 인식 오류 |
| ---: | ---: | ---: |
| 62.30%* | 22.63%* | 15.07%* |

> `*` 합성 fine-tuned 값은 과거 생성 결과에서 첫 역할 토큰 이후의 반복 꼬리를 제거해 재계산한 진단 수치입니다. 현재 평가 코드는 chat EOS를 올바르게 설정하지만, 논문용 최종 수치는 전체 재추론으로 확정해야 합니다.

AI Hub fine-tuned 결과의 corpus CER은 `0.1859`, exact match는 `156/2,000`(7.8%)입니다. 합성 도메인에서는 성능이 개선되었지만 실제 필기체 CER은 악화되었습니다. 이는 합성 데이터 중심 학습에 따른 도메인 과적합 또는 부분적 망각 가능성을 보여주며, 실제 필기체 혼합 학습과 projector/LoRA 학습률 ablation이 필요합니다.

## 빠른 시작

Python 3.10+ 가상환경을 권장합니다. CUDA 환경에 맞는 PyTorch를 설치한 뒤 필요한 패키지를 설치합니다.

```bash
python -m pip install Pillow numpy kiwipiepy PyYAML transformers peft
```

### 모델 다운로드

```bash
hf download \
  Mun2/qwen3.5-4b-korean-handwriting-ocr-overcorrection \
  --local-dir outputs/huggingface/overcorrection-50-50
```

> **중요:** 이 저장소는 전체 4B 베이스 가중치가 아니라 LoRA adapter, vision projector, processor를 제공합니다. `Qwen/Qwen3.5-4B`에 `adapter/`와 `projector.pt`를 모두 적용해야 하며, adapter만 로드하면 학습된 모델과 동일하지 않습니다.

이 저장소의 평가 스크립트는 Hugging Face 체크포인트 디렉터리를 바로 사용할 수 있습니다. 베이스 모델 `Qwen/Qwen3.5-4B`도 로컬 캐시에 준비되어 있어야 합니다.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_overcorrection_aihub.py \
  --config configs/overcorrection_50_50_40000.yaml \
  --checkpoint outputs/huggingface/overcorrection-50-50 \
  --results-dir results/overcorrection/hf_aihub
```

### 데이터 준비와 학습

경로와 샘플 수를 바꾸기 전 각 스크립트의 `--help`를 먼저 확인하세요.

```bash
python scripts/prepare_overcorrection_data.py --help
python scripts/build_final_eval_sets.py

CUDA_VISIBLE_DEVICES=0 python scripts/train_overcorrection.py \
  --config configs/overcorrection_50_50_40000.yaml
```

### 평가

```bash
# 합성 필기체
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_overcorrection.py \
  --config configs/overcorrection_50_50_40000.yaml \
  --results-dir results/overcorrection/50_50_40000

# AI Hub 실제 필기체
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_overcorrection_aihub.py \
  --config configs/overcorrection_50_50_40000.yaml \
  --results-dir results/overcorrection/50_50_40000_aihub
```

## 저장소 구조

```text
assets/     파이프라인 도판
configs/    실험별 YAML 설정
scripts/    데이터 준비, 학습, 평가, 보고서 생성 코드
results/    공개 가능한 요약 지표 JSON
```

원본 데이터, 생성 이미지, 체크포인트, 원시 예측 JSONL과 로그는 저장소에 포함하지 않습니다.

## 데이터와 재현성

- AI Hub 원본 데이터는 재배포하지 않으며 `AIHUB_APIKEY`를 통해 별도로 내려받아야 합니다.
- 원본 `data/053.대용량_손글씨_OCR_데이터/`는 수정하지 않습니다.
- 데이터 선택과 재표집에는 명시적 seed 42를 사용합니다.
- 모델은 Apache-2.0 라이선스의 `Qwen/Qwen3.5-4B`를 기반으로 합니다. 데이터 사용 조건은 각 원본 데이터 제공처의 정책을 따릅니다.

## 결과 파일

- 베이스 모델 종합 평가: [`results/final_evaluation_summary.json`](results/final_evaluation_summary.json)
- fine-tuned AI Hub 지표: [`results/overcorrection/50_50_40000_aihub/metrics.json`](results/overcorrection/50_50_40000_aihub/metrics.json)
- fine-tuned 합성 진단 지표: [`results/overcorrection/50_50_40000/diagnostic_metrics.json`](results/overcorrection/50_50_40000/diagnostic_metrics.json)

원시 예측과 평가 로그는 용량 및 데이터 노출 문제로 Git에서 제외합니다.
