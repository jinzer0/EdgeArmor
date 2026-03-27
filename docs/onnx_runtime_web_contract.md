# ONNX Runtime Web Contract

이 문서는 현재 코드베이스의 two-stage deepfake pipeline을 `onnxruntime-web`으로 옮길 때의 실행 경계를 고정한다.

## 범위

- ONNX로 export하는 대상
  - `artifacts/onnx/face_detector.onnx`
  - `artifacts/onnx/forensics_adapter.onnx`
- JS가 담당하는 대상
  - 이미지 decode
  - detector 입력용 letterbox, resize, tensor 변환
  - detector 출력 후 bbox 선택
  - 원본 이미지 기준 crop
  - classifier 입력용 resize, normalize, `if_boundary` 생성
  - 얼굴별 결과를 이미지 단위 결과로 집계

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

## Classifier Contract

### Input

- `image`
  - dtype: `float32`
  - shape: `[1, 3, 256, 256]`
- `if_boundary`
  - dtype: `float32`
  - shape: `[1, 256]`
  - 값: 모두 `1.0`

### JS 전처리

- detector에서 선택한 face bbox를 원본 이미지에서 crop
- 현재 Python pipeline과 동일하게 margin crop을 유지
- crop 이미지를 `256x256`으로 resize
- RGB 기준 normalize 적용
  - mean: `[0.48145466, 0.4578275, 0.40821073]`
  - std: `[0.26862954, 0.26130258, 0.27577711]`
- CHW 순서 `Float32Array`로 변환
- `if_boundary = ones([1, 256])` 생성

### Output

- `logits`
  - dtype: `float32`
  - shape: `[1, 2]`
- `fake_prob`
  - dtype: `float32`
  - shape: `[1]`
- `xray_pred`
  - dtype: `float32`
  - shape: `[1, 1, 256, 256]`

## Aggregation Rule

- 얼굴별 `fake_prob`를 계산한다
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
