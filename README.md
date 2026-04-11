
# EdgeArmor

[![Platform: Chrome & Edge](https://img.shields.io/badge/Platform-Chrome%20%7C%20Edge-blue.svg)](#quick-start)
[![Status: In Progress](https://img.shields.io/badge/Status-In%20Progress-green.svg)](#)

> 온디바이스 기반 딥페이크 이미지 탐지 및 방어 브라우저 익스텐션
>
> 서버 전송 없이, 브라우저 안에서만 탐지하고 방어합니다.

**[프로젝트 구조 보러가기](#project-structure)**
---
## Highlights

- 실시간 이미지 탐지
- 업로드 이미지 방어 노이즈 주입
- 브라우저 로컬 추론
- Chrome / Edge 지원
---
## Stack
> Work In Progress - 변경될 수 있음!

| Layer | Tech |
| :--- | :--- |
| Frontend | HTML5, CSS3, JavaScript (ES6+), Manifest V3 |
| AI / ML | YOLOv8 ONNX Face Detector, ForensicsAdapter, ONNX Runtime Web |
| Build | Webpack / Vite |
---
## Quick Start
> Work In Progress - 변경될 수 있음!

- Face detector ONNX export
  - `python scripts/export_detector_onnx.py --out_dir artifacts/onnx`
- Classifier ONNX export
  - `conda run -n ml3_13 python scripts/export_forensics_adapter_onnx.py --weights_path ckpt_best.pth --config_path ForensicsAdapter/config/test.yaml --out_dir artifacts/onnx`
- Browser extension
  - `npm install`
  - `python scripts/prepare_chrome_extension.py`
  - prepare 스크립트는 `face_detector.onnx`, browser classifier ONNX, `inference_contract.json`, ORT Web 런타임을 `extension/` 폴더에 배치
  - Chrome `chrome://extensions`에서 `extension/` 폴더를 unpacked extension으로 로드
  - 상세 내용은 `docs/browser_extension.md` 참고

Attribution:

- face detector는 Ultralytics YOLO runtime으로 export한 `arnabdhar/YOLOv8-Face-Detection` 가중치를 사용한다.
---
## Project Structure
> Work In Progress - 변경될 수 있음!

```text
EdgeArmor/
├── README.md
├── package.json
├── package-lock.json
├── docs/
│   ├── browser_extension.md
│   └── onnx_runtime_web_contract.md
├── extension/
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.css
│   ├── popup.js
│   ├── image_utils.js
│   ├── onnx_face_detector.js
│   ├── models/                         # gitignored, extension 실행 필수 산출물
│   │   ├── face_detector.onnx
│   │   ├── inference_contract.json
│   │   ├── model.onnx
│   │   └── ...
│   └── vendor/                         # gitignored, extension 실행 필수 런타임 파일
│       ├── ort.all.min.mjs
│       └── ort-wasm-*.wasm
├── artifacts/
│   └── onnx/                           # gitignored, 원본/변환 모델 저장소
│       ├── face_detector.onnx
│       ├── forensics_adapter.onnx
│       ├── forensics_adapter.webgpu.fp16.onnx
│       └── ...
├── scripts/
│   ├── export_detector_onnx.py
│   ├── export_forensics_adapter_onnx.py
│   ├── prepare_chrome_extension.py
│   ├── quantize_forensics_adapter_onnx.py
│   ├── benchmark_forensics_adapter_onnx.py
│   ├── verify_onnx_exports.py
│   ├── run_extension_inference_server.py
│   └── run_pipeline.py
├── src/
│   ├── detection/
│   │   └── FaceDetection.py
│   └── pipeline/
│       ├── deepfake_pipeline.py
│       ├── face_selector.py
│       ├── forensics_adapter_infer.py
│       └── preprocess.py
├── ForensicsAdapter/
│   ├── config/
│   ├── dataset/
│   ├── model/
│   ├── trainer/
│   ├── train.py
│   └── test.py
├── dataset/
│   └── weight/
│       └── ViT-L-14.pt                # gitignored, ForensicsAdapter 실행 필수
└── ckpt_best.pth                      # gitignored, classifier export/server 실행 필수
```

필수 gitignored 파일 메모:

- `ckpt_best.pth`: Python deepfake classifier 가중치
- `dataset/weight/ViT-L-14.pt`: ForensicsAdapter 백본 로딩에 필요
- `artifacts/onnx/*.onnx`: export, quantize, verify, extension 모델 준비의 입력
- `extension/models/*`: 실제 Chrome extension이 직접 로드하는 모델 복사본
- `extension/vendor/*`: ORT Web runtime 번들

---
## Checklist
> Work In Progress - 변경될 수 있음!


- [ ] 로컬 추론 기반 탐지
- [ ] 이미지 방어 노이즈 주입
- [ ] Chrome / Edge 지원
- [ ] 모델 최적화
- [ ] 배포용 패키징
---
## AI HACK CAMP 2026
### Team: 다찾을건대


## License
