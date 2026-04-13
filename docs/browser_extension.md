# Browser Extension

이 프로젝트는 Chrome Manifest V3 기준의 unpacked extension을 포함한다. 현재 기본 경로는 popup UI에서 이미지를 고른 뒤 브라우저 내부에서 YOLOv8 face detector ONNX와 classifier ONNX를 실행하고, 결과를 popup에 렌더링하는 방식이다.

## 구조

- `extension/manifest.json`
  - 확장 루트 설정
- `extension/popup.html`
  - popup UI
- `extension/popup.js`
  - 기본 경로에서는 브라우저 내부 ONNX detector/classifier 실행
  - 필요하면 `local_server` 모드로 Python 서버 호출
  - 결과 집계 및 렌더링
- `extension/onnx_face_detector.js`
  - `face_detector.onnx` 로드
  - detector 입력 letterbox 및 출력 bbox 복원
- `extension/popup.css`
  - popup 스타일
- `scripts/prepare_chrome_extension.py`
  - extension 폴더에 detector ONNX, browser classifier ONNX, `inference_contract.json`, ORT 런타임 파일을 배치
- `scripts/run_extension_inference_server.py`
  - 선택적인 로컬 Python 추론 서버 실행

## 실행 방식

1. detector ONNX 모델을 준비한다.
```bash
python scripts/export_detector_onnx.py --out_dir artifacts/onnx
```

2. classifier ONNX 모델을 export한다.
```bash
conda run -n ml3_13 python scripts/export_forensics_adapter_onnx.py --weights_path ckpt_best.pth --config_path ForensicsAdapter/config/test.yaml --out_dir artifacts/onnx
```

classifier export가 끝나면 legacy classifier artifact와 browser-targeted variant가 함께 생성될 수 있다. 다만 extension packaging은 특정 fp16 파일명을 전제로 하지 않고, 사용 가능한 classifier 후보 중에서 **유효한 ONNX만 검사한 뒤** `extension/models/model.onnx`로 복사한다.

3. ORT Web 의존성을 설치한다.
```bash
npm install
```

4. 확장 로드용 자산을 `extension/` 폴더에 준비한다.
```bash
python scripts/prepare_chrome_extension.py
```

이 스크립트는 detector ONNX를 `extension/models/face_detector.onnx`로, 검증된 classifier ONNX를 `extension/models/model.onnx`로, shared runtime contract를 `extension/models/inference_contract.json`으로, ORT Web 런타임 파일을 `extension/vendor/`로 복사 또는 링크한다. invalid ONNX artifact는 packaging 단계에서 제외된다.

5. Chrome에서 `chrome://extensions`를 열고 `개발자 모드`를 켠다.

6. `압축해제된 확장 프로그램을 로드합니다`를 눌러 `/Users/kjy/Desktop/Codes/university/university_26-1/EdgeArmor/extension` 폴더를 선택한다.

7. 확장 popup에서 이미지를 올리고 분석 버튼을 누른다.

## 선택 경로: 로컬 서버 모드

필요하면 `extension/popup.js`의 `runtimeMode`를 `"local_server"`로 바꿔 기존 Python pipeline을 사용할 수 있다.

```bash
conda run -n ml3_13 python scripts/run_extension_inference_server.py --weights_path ckpt_best.pth
```

## 런타임 요구사항

- 기본 경로는 브라우저 내부 YOLOv8 detector ONNX + ONNX Runtime Web classifier 조합이다
- extension manifest에는 `127.0.0.1` 및 `localhost` host permission이 포함된다
- 필요하면 `runtimeMode`를 `local_server`로 바꿔 기존 Python 서버 경로를 사용할 수 있다
- detector export는 `arnabdhar/YOLOv8-Face-Detection` 가중치를 Ultralytics YOLO runtime으로 ONNX 변환한 산출물을 사용한다

## 현재 JS 파이프라인

- popup
  - 파일 선택
  - `face_detector.onnx` 로 얼굴 bbox 검출
  - `inference_contract.json` 로 classifier/selection/browser 기본값 동기화
  - margin crop 후 classifier 입력 크기로 resize
  - ONNX Runtime Web classifier 추론 (`image` / `if_boundary` dtype은 loaded model metadata를 기준으로 결정)
  - 결과를 preview/summary 패널에 렌더링
- 선택 경로
  - 로컬 서버 모드에서는 Python `DeepfakeDetectionPipeline` 결과를 그대로 렌더링

## 제약

- 원본 `artifacts/onnx/forensics_adapter.onnx`는 약 1.2GB라서 extension에서 직접 쓰지 않는다
- extension은 선택된 valid classifier ONNX를 `extension/models/model.onnx`로 복사해 사용한다
- current repo에서는 invalid browser fp16 artifact가 있더라도 `scripts/prepare_chrome_extension.py`가 이를 건너뛰고 valid fallback classifier를 선택한다
- `forensics_adapter.webgpu.fp16.onnx`는 legacy browser-targeted variant로 남아 있을 수 있지만, runtime 기본 전제는 아니다
- detector는 `extension/models/face_detector.onnx`를 ORT Web으로 실행한다
- 현재 구현은 로컬 unpacked extension 기준이다
- Web Store 배포 전에는 모델 크기 축소, 분할, 또는 더 가벼운 classifier가 필요하다
