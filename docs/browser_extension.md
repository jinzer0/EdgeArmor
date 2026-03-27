# Browser Extension

이 프로젝트는 Chrome Manifest V3 기준의 unpacked extension을 포함한다. 업로드한 이미지를 popup 안에서 바로 two-stage pipeline으로 분석한다.

## 구조

- `extension/manifest.json`
  - 확장 루트 설정
- `extension/popup.html`
  - popup UI
- `extension/popup.js`
  - detector/classifier ONNX 세션 생성
  - JS 전처리, crop, 후처리, 결과 집계
- `extension/popup.css`
  - popup 스타일
- `scripts/prepare_chrome_extension.py`
  - extension 폴더에 모델과 ORT Web 런타임 파일을 배치

## 실행 방식

1. ONNX 모델을 먼저 export한다.
```bash
conda run -n ml3_13 python scripts/export_detector_onnx.py --out_dir artifacts/onnx
conda run -n ml3_13 python scripts/export_forensics_adapter_onnx.py --weights_path ckpt_best.pth --config_path ForensicsAdapter/config/test.yaml --out_dir artifacts/onnx
```

2. ORT Web 의존성을 설치한다.
```bash
npm install
```

3. 확장 로드용 자산을 `extension/` 폴더에 준비한다.
```bash
python scripts/prepare_chrome_extension.py
```

4. Chrome에서 `chrome://extensions`를 열고 `개발자 모드`를 켠다.

5. `압축해제된 확장 프로그램을 로드합니다`를 눌러 `/Users/kjy/Desktop/Codes/university/university_26-1/EdgeArmor/extension` 폴더를 선택한다.

6. 확장 popup에서 이미지를 올리면 detector -> crop -> classifier 순으로 분석한다.

## 현재 JS 파이프라인

- detector 입력
  - `640x640` letterbox
  - RGB `0..1`
  - CHW `Float32Array`
- detector 후처리
  - bbox 원본 좌표계 복원
  - `min_confidence`, `min_face_size`, `top_k`
- classifier 입력
  - detector bbox 기준 margin crop
  - `256x256` resize
  - mean/std normalize
  - `if_boundary = ones([1, 256])`
- 집계
  - 얼굴별 fake probability의 최댓값으로 이미지 판정

## 제약

- `artifacts/onnx/forensics_adapter.onnx`가 약 1.2GB라서 popup 첫 실행과 메모리 사용량이 무겁다
- 현재 구현은 로컬 unpacked extension 기준이다
- Web Store 배포 전에는 모델 크기 축소, 분할, 또는 더 가벼운 classifier가 필요하다
