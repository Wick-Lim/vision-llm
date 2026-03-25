# Vision LLM

벡터 경로 기반 시각 언어 모델. 토크나이저 없이, 폰트의 기하학적 경로(베지어 곡선)를 직접 학습하는 확산 모델.

## 핵심 아이디어

문자는 사람이 눈으로 보기 위해 만든 시각적 기호다. 폰트 파일 안에서 "ㅎ"은 "위에 원, 아래에 줄 두 개"라는 기하학적 구조로 기술된다. 이 모델은 픽셀이 아닌 **벡터 경로** — 점 좌표, 베지어 곡선, 직선 — 를 직접 보고 생성한다.

```
텍스트 → [벡터 추출] → 경로 텐서 → [인코더] → 조건 벡터 → [확산 모델] → 경로 텐서 → [렌더러] → 이미지 → [OCR] → 텍스트
```

### 왜 래스터가 아닌 벡터인가

| | 래스터 (픽셀) | 벡터 (경로) |
|---|---|---|
| "한" 표현 | 16,384 픽셀값 (대부분 빈 배경) | 38개 좌표/제어점 |
| 해상도 | 고정, 올리면 연산량 제곱 증가 | 독립적, 좌표점 수에만 비례 |
| 폰트 변경 | 완전히 다른 픽셀 배열 | 동일한 구조, 다른 좌표값 |
| 본질 | 원본 정보를 버린 결과물 | 원본 그 자체 |

## 파이프라인

| 단계 | 설명 | 구현 |
|------|------|------|
| 1. 벡터 추출 | 폰트에서 글리프 베지어 경로 추출, `[max_len, 8]` 텐서 변환 | `src/vectorizer.py` |
| 2. 인코딩 | 1D-CNN으로 경로 텐서에서 조건 벡터 추출 | `src/encoder.py` |
| 3. 생성 | 1D U-Net 확산 모델이 조건 벡터로부터 응답 경로 생성 | `src/diffusion.py` |
| 4. 렌더링 | 생성된 벡터 경로를 이미지로 렌더링 | `src/renderer.py` |
| 5. OCR | 렌더링된 이미지에서 텍스트 추출 | `pytesseract` |

## 아키텍처

```
PathEncoder (~190K params)
└── Conv1d ×3 (8→64→128→256) → AdaptivePool → FC → 조건 벡터 [256]

UNet1d (~10.6M params)  — 1D 시퀀스 확산 모델
├── Input: 경로 텐서 [B, 512, 8] + 조건 벡터 concat
├── Down: ConvBlock1d ×3 (128→256→512) + Downsample
├── Mid: ConvBlock1d + SelfAttention1d
├── Up: ConvBlock1d ×3 + Upsample + skip connections
├── Time: Sinusoidal → MLP 임베딩
└── Output: 예측 노이즈 [B, 512, 8]
```

**텐서 포맷** — 각 행 `[cmd, x, y, cx1, cy1, cx2, cy2, 0]`:
- cmd: 0=pad, 1=moveTo, 2=lineTo, 3=curveTo, 4=closePath
- 좌표: 글리프 bbox 기준 `[-0.5, 0.5]` 정규화

## 설치

```bash
pip install -e ".[dev]"
```

OCR을 위해 Tesseract 필요:

```bash
# macOS
brew install tesseract

# Ubuntu
sudo apt install tesseract-ocr
```

## 사용법

### 학습

```bash
# 한글 글리프 에코 태스크 (기본)
python -m src.train --epochs 100

# 옵션
python -m src.train --epochs 100 --batch-size 32 --lr 1e-3 --max-len 512 --device cpu
```

### 데모

```bash
python scripts/demo.py "가" --checkpoint checkpoints/best.pt
```

`demo_input.png`(입력 글리프 렌더링)과 `demo_output.png`(생성된 경로 렌더링)이 저장된다.

## 프로젝트 구조

```
vision-llm/
├── pyproject.toml           # 의존성 (torch, fonttools, Pillow, pytesseract)
├── src/
│   ├── vectorizer.py        # 폰트 글리프 → 벡터 경로 추출/텐서 변환
│   ├── encoder.py           # 1D-CNN 조건 벡터 인코더
│   ├── diffusion.py         # 1D U-Net + 노이즈 스케줄러 + DDIM 샘플러
│   ├── renderer.py          # 벡터 경로 → PIL 이미지 렌더링
│   ├── dataset.py           # 한글/ASCII 글리프 에코 데이터셋
│   └── train.py             # 확산 모델 학습 루프
├── scripts/
│   └── demo.py              # 전체 파이프라인 데모
└── checkpoints/             # 학습된 모델 가중치
```

## 구조적 장점

- **토크나이저 불필요**: 렌더링 가능한 모든 문자 체계를 동일한 파이프라인으로 처리
- **모달리티 경계 소멸**: 입출력이 모두 벡터 경로이므로 텍스트/다이어그램/수식 구분 없음
- **데이터 효율**: 래스터 대비 1~2자릿수 적은 데이터량
- **해상도 독립**: 벡터이므로 어떤 크기로든 선명하게 렌더링 가능

## 로드맵

- **Phase 1** (현재): 글리프 에코 — 확산 모델이 벡터 경로 분포를 학습할 수 있는지 검증
- **Phase 2**: 생성 품질 향상 — U-Net 스케일업, DDIM 스텝 최적화, OCR 연동 평가
- **Phase 3**: 범용 확장 — 다국어 데이터셋, 문단 단위 처리, 텍스트 외 벡터 데이터

## 참고 논문

- [PIXEL (Rust et al., 2023)](https://arxiv.org/abs/2207.06991) — 텍스트를 렌더링된 이미지로 처리하는 BERT급 모델
- [Pix2Struct (Lee et al., 2023)](https://arxiv.org/abs/2210.03347) — 스크린샷→구조 변환 사전학습 모델
- [DeepSVG (Carlier et al., 2020)](https://arxiv.org/abs/2007.11301) — SVG 벡터 경로 직접 생성 모델
- [Im2Vec (Reddy et al., 2021)](https://arxiv.org/abs/2102.02798) — 래스터→벡터 그래픽 변환
