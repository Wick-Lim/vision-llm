# Vision LLM

토크나이저 없이, 텍스트를 이미지로 렌더링하여 처리하는 비전 기반 언어 모델 PoC.

## 핵심 아이디어

문자는 사람이 눈으로 보기 위해 만든 시각적 기호다. 모델도 텍스트를 "읽는" 대신 "보게" 만든다.

```
텍스트 → [렌더링] → 이미지 → [Encoder] → 잠재 벡터 → [Think] → [Decoder] → 이미지 → [OCR] → 텍스트
```

## 파이프라인

| 단계 | 설명 | 구현 |
|------|------|------|
| 1. 렌더링 | 텍스트를 256×64 그레이스케일 이미지로 변환 | `src/renderer.py` |
| 2. 추론 | CNN Encoder → Think MLP → CNN Decoder | `src/model.py` |
| 3. 복원 | 출력 이미지에서 OCR로 텍스트 추출 | `pytesseract` |

## 아키텍처

```
VisionLLM (~5M params)
├── Encoder: Conv2d ×4 (1→32→64→128→256) → FC → latent (256-dim)
├── ThinkLayer: 3-layer MLP + residual connection
└── Decoder: FC → ConvTranspose2d ×4 (256→128→64→32→1) → Sigmoid
```

- **입력**: `[B, 1, 64, 256]` 그레이스케일 이미지
- **잠재 공간**: 256차원 벡터
- **출력**: `[B, 1, 64, 256]` 그레이스케일 이미지
- **손실 함수**: MSE (0.7) + 1-SSIM (0.3)

## 학습 태스크

| 태스크 | 입력 | 목표 출력 | 목적 |
|--------|------|-----------|------|
| `echo` | `hello` | `hello` | 인코더-디코더 파이프라인 검증 |
| `uppercase` | `hello` | `HELLO` | 잠재 공간에서의 시각적 변환 학습 |

## 설치

```bash
pip install -e .
```

OCR을 위해 Tesseract가 필요하다:

```bash
# macOS
brew install tesseract

# Ubuntu
sudo apt install tesseract-ocr
```

## 사용법

### 학습

```bash
# Echo 태스크 (기본)
python -m src.train --task echo --epochs 50

# Uppercase 태스크
python -m src.train --task uppercase --epochs 50

# 전체 옵션
python -m src.train --task echo --epochs 100 --batch-size 128 --lr 1e-3 --dataset-size 20000
```

### 데모

```bash
python scripts/demo.py "hello" --checkpoint checkpoints/best_echo.pt
```

`demo_input.png`과 `demo_output.png`이 생성된다.

## 프로젝트 구조

```
vision-llm/
├── pyproject.toml        # 의존성 및 프로젝트 메타데이터
├── src/
│   ├── renderer.py       # 텍스트 → 이미지 렌더링
│   ├── model.py          # VisionLLM (Encoder + Think + Decoder)
│   ├── dataset.py        # Echo / Uppercase 데이터셋 생성기
│   └── train.py          # 학습 루프
├── scripts/
│   └── demo.py           # 엔드투엔드 파이프라인 데모
└── checkpoints/          # 학습된 모델 가중치
```

## 이 접근의 장점

- **토크나이저 불필요**: 어떤 언어든 렌더링만 되면 동일하게 처리. 한국어, 아랍어, 수식, 악보도 같은 파이프라인.
- **텍스트-이미지 경계 소멸**: 입출력이 모두 이미지이므로 멀티모달 처리가 구조적으로 자연스럽다.

## 한계 및 향후 방향

- **스케일**: 현재 CNN + MLP 구조는 패턴 매칭 수준. ViT 기반으로 확장 필요.
- **해상도**: 고정 256×64 이미지에 긴 텍스트는 담기 어렵다. 가변 해상도 또는 이미지 분할 필요.
- **OCR 병목**: 출력 이미지 품질이 전체 성능의 상한선. 학습 초기에는 MSE/SSIM으로 평가.

## 참고 논문

- [PIXEL (Rust et al., 2023)](https://arxiv.org/abs/2207.06991) — 텍스트를 렌더링된 이미지로 처리하는 BERT급 모델
- [Pix2Struct (Lee et al., 2023)](https://arxiv.org/abs/2210.03347) — 스크린샷을 구조적 출력으로 변환하는 사전학습 모델
