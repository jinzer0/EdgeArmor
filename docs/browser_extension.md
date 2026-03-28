# Browser Extension

이 프로젝트는 Chrome Manifest V3 기준의 unpacked extension을 포함한다. 현재 기본 경로는 popup UI에서 이미지를 고른 뒤 로컬 Python 추론 서버로 요청을 보내고, 결과를 popup에 렌더링하는 방식이다.

## 구조

- `extension/manifest.json`
  - 확장 루트 설정
- `extension/popup.html`
  - popup UI
- `extension/popup.js`
  - 기본 경로에서는 로컬 추론 서버 호출
  - 기존 browser ONNX 실행 경로는 실험용으로 남아 있음
  - 결과 집계 및 렌더링
- `extension/popup.css`
  - popup 스타일
- `scripts/prepare_chrome_extension.py`
  - extension 폴더에 모델과 ORT Web 런타임 파일을 배치
- `scripts/run_extension_inference_server.py`
  - 로컬 Python 추론 서버 실행

## 실행 방식

1. ONNX 모델을 먼저 export한다.
```bash
conda run -n ml3_13 python scripts/export_detector_onnx.py --out_dir artifacts/onnx
conda run -n ml3_13 python scripts/export_forensics_adapter_onnx.py --weights_path ckpt_best.pth --config_path ForensicsAdapter/config/test.yaml --out_dir artifacts/onnx
```

classifier export가 끝나면 브라우저용 `artifacts/onnx/forensics_adapter.webgpu.fp16.onnx`도 함께 생성된다.

2. ORT Web 의존성을 설치한다.
```bash
npm install
```

3. 확장 로드용 자산을 `extension/` 폴더에 준비한다.
```bash
python scripts/prepare_chrome_extension.py
```

4. 로컬 추론 서버를 실행한다.
```bash
conda run -n ml3_13 python scripts/run_extension_inference_server.py --weights_path ckpt_best.pth
```

5. Chrome에서 `chrome://extensions`를 열고 `개발자 모드`를 켠다.

6. `압축해제된 확장 프로그램을 로드합니다`를 눌러 `/Users/kjy/Desktop/Codes/university/university_26-1/EdgeArmor/extension` 폴더를 선택한다.

7. 확장 popup에서 이미지를 올리고 분석 버튼을 누른다.

## 런타임 요구사항

- 기본 경로는 `http://127.0.0.1:8765/analyze` 로컬 서버 호출이다
- extension manifest에는 `127.0.0.1` 및 `localhost` host permission이 포함된다
- 브라우저 내부 WebGPU classifier 경로는 현재 Chrome/macOS 조합에서 안정적이지 않아 기본 비활성화 대상으로 취급한다

## 현재 JS 파이프라인

- popup
  - 파일 선택
  - 로컬 서버에 원본 이미지 전송
  - 응답 결과를 preview/summary 패널에 렌더링
- 로컬 서버
  - Python `DeepfakeDetectionPipeline` 실행
  - 얼굴 탐지, crop, classifier 추론
  - 얼굴별 결과와 최종 집계 JSON 반환

## 제약

- 원본 `artifacts/onnx/forensics_adapter.onnx`는 약 1.2GB라서 extension에서 직접 쓰지 않는다
- extension은 `artifacts/onnx/forensics_adapter.webgpu.fp16.onnx`를 `extension/models/forensics_adapter.onnx`로 복사해 사용한다
- `forensics_adapter.webgpu.fp16.onnx`도 약 593MB라서 현재 Chrome 146/macOS 26 조합에서는 browser WebGPU 세션 생성 시 브라우저 크래시가 발생할 수 있다
- 그래서 기본 사용 경로는 브라우저 내부 classifier 로드가 아니라 로컬 Python 서버다
- 현재 구현은 로컬 unpacked extension 기준이다
- Web Store 배포 전에는 모델 크기 축소, 분할, 또는 더 가벼운 classifier가 필요하다
