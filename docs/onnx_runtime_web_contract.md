# ONNX Runtime Web Contract

이 문서는 현재 코드베이스의 two-stage deepfake pipeline을 `onnxruntime-web`으로 옮길 때의 실행 경계와 shared inference contract를 고정한다.

## 범위

- ONNX로 export하는 대상
  - `artifacts/onnx/face_detector.onnx`
  - `artifacts/onnx/forensics_adapter.onnx`
- Python이 source of truth로 관리하는 대상
  - `src/pipeline/inference_contract.py`
  - `scripts/prepare_chrome_extension.py`가 생성하는 `extension/models/inference_contract.json`
- JS가 담당하는 대상
  - 이미지 decode
  - detector 입력용 letterbox, resize, tensor 변환
  - detector 출력 후 bbox 선택
  - 원본 이미지 기준 crop
  - classifier 입력용 resize, normalize, `if_boundary` 생성
  - contract JSON을 runtime config에 merge

## Shared Contract Artifact

- 생성 위치: `extension/models/inference_contract.json`
- 생성 주체: `scripts/prepare_chrome_extension.py`
- 원본 정의: `src/pipeline/inference_contract.py`
- popup은 이 파일을 먼저 읽고, 파일이 없거나 읽기 실패하면 기존 JS defaults로 fallback한다.

현재 contract JSON이 담는 필드:

- `classifier`
  - `modelFile`
  - `inputSize`
  - `ifBoundaryLength`
  - `mean`
  - `std`
  - `hfMean`
  - `hfStd`
  - `fakeThreshold`
- `selection`
  - `minConfidence`
  - `minFaceSize`
  - `topK`
- `browser`
  - `margin`
- `detector`
  - `modelFile`

## Detector Contract

### Input

- 이름: export 결과의 첫 번째 input
- dtype: `float32`
- shape: `[1, 3, 640, 640]`
- 색상: RGB
- 값 범위: `0..1`
- 전처리:
  - 원본 이미지를 `640x640` 기준으로 letterbox
  - CHW 순서로 변환

### Output

- 실제 output 이름과 shape는 export 결과를 기준으로 확정한다
- `scripts/export_detector_onnx.py`와 `scripts/verify_onnx_exports.py`가 이를 출력한다
- JS는 bbox와 confidence만 소비한다
- landmarks가 출력에 있더라도 현재 파이프라인에서는 사용하지 않는다

### JS 후처리

- confidence threshold 적용
- `min_face_size` 적용
- confidence 내림차순 정렬
- `top_k` 선택
- 선택된 bbox를 원본 이미지 좌표계로 복원
- bbox 기준으로 margin crop

현재 detector 실행 경로는 `extension/onnx_face_detector.js`에 있고, detector model filename은 contract에도 기록되지만 런타임 detector 모듈 자체는 별도 구현을 유지한다.

## Classifier Contract

### Input

- browser runtime은 loaded classifier의 input metadata를 기준으로 입력 이름과 dtype을 확정한다.
- 현재 popup 구현 규칙:
  - Hugging Face 계열 classifier (`pixel_values` 입력)
    - 입력 이름: `pixel_values`
    - dtype: `float32`
    - shape: `[1, 3, H, W]` (`H/W`는 metadata 우선, 없으면 `224` fallback)
    - `if_boundary` 없음
  - legacy ForensicsAdapter classifier (`image` 입력)
    - 입력 이름: `image`
    - dtype: loaded model metadata 기준 (`tensor(float16)`면 `float16`, 그 외에는 `float32`)
    - shape: `[1, 3, 256, 256]`
    - `if_boundary`
      - dtype: loaded model metadata 기준 (`tensor(float16)`면 `float16`, 그 외에는 `float32`)
      - shape: `[1, ifBoundaryLength]` (`inference_contract.json` 기준, 현재 `256`)
      - 값: 모두 `1.0`

- 현재 repo에서 packaging이 valid fallback으로 선택하는 `forensics_adapter.onnx`는 `image` / `if_boundary` 둘 다 `tensor(float)` 입력을 가진다.

### JS 전처리

- detector에서 선택한 face bbox를 원본 이미지에서 crop
- Python pipeline 기본 margin은 `0.25`지만 browser contract는 현재 `0.4`를 사용한다
- crop 이미지를 `256x256`으로 resize
- RGB 기준 normalize 적용
  - mean: `[0.48145466, 0.4578275, 0.40821073]`
  - std: `[0.26862954, 0.26130258, 0.27577711]`
- CHW 순서 `Float32Array`로 변환
- `if_boundary = ones([1, 256])` 생성 (dtype은 loaded model metadata를 따름)

### Output

- `logits`
  - dtype: model/runtime에 따라 달라질 수 있다
  - shape: `[1, 2]`
- `fake_prob`
  - dtype: model/runtime에 따라 달라질 수 있다
  - shape: `[1]`
- `xray_pred`
  - dtype: model/runtime에 따라 달라질 수 있다
  - shape: `[1, 1, 256, 256]`

## Aggregation Rule

- 얼굴별 `fake_prob`를 계산한다
- 얼굴별 label은 `pred_label_id == 1 && fake_prob >= fake_threshold` 규칙을 따른다
- 이미지 단위 결과는 현재 Python pipeline과 동일하게 `max(fake_prob)` 기준으로 정한다
- `max(fake_prob) >= fake_threshold`이면 `fake`
- 아니면 `real`

## Python Scripts

- detector export
```bash
python scripts/export_detector_onnx.py --out_dir artifacts/onnx
```

- classifier export
```bash
python scripts/export_forensics_adapter_onnx.py --weights_path /path/to/ckpt_best.pth --config_path ForensicsAdapter/config/test.yaml --out_dir artifacts/onnx
```

- parity verification
```bash
python scripts/verify_onnx_exports.py --image /path/to/image --weights_path /path/to/ckpt_best.pth --detector_onnx artifacts/onnx/face_detector.onnx --classifier_onnx artifacts/onnx/forensics_adapter.onnx
```
