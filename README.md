
# EdgeArmor

[![Platform: Chrome & Edge](https://img.shields.io/badge/Platform-Chrome%20%7C%20Edge-blue.svg)](#quick-start)
[![Status: In Progress](https://img.shields.io/badge/Status-In%20Progress-green.svg)](#)

> 온디바이스 기반 딥페이크 이미지 탐지 및 방어 브라우저 익스텐션
>
> 서버 전송 없이, 브라우저 안에서만 탐지하고 방어합니다.

현재 작업 브랜치: `codex/mediapipe-face-detection`   
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
| AI / ML | MediaPipe Tasks Vision, ONNX Runtime Web |
| Build | Webpack / Vite |
---
## Quick Start
> Work In Progress - 변경될 수 있음!

- Classifier ONNX export
  - `conda run -n ml3_13 python scripts/export_forensics_adapter_onnx.py --weights_path ckpt_best.pth --config_path ForensicsAdapter/config/test.yaml --out_dir artifacts/onnx`
- Browser extension
  - `npm install`
  - `python scripts/prepare_chrome_extension.py`
  - face detector는 MediaPipe 모델을 자동 다운로드해 사용
  - Chrome `chrome://extensions`에서 `extension/` 폴더를 unpacked extension으로 로드
  - 상세 내용은 `docs/browser_extension.md` 참고
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
│   ├── mediapipe_face_detector.js
│   ├── models/                         # gitignored, extension 실행 필수 산출물
│   │   ├── blaze_face_short_range.tflite
│   │   ├── model.onnx
│   │   ├── model_fp16.onnx
│   │   ├── model_int8.onnx
│   │   ├── model_q4f16.onnx
│   │   ├── forensics_adapter.onnx
│   │   └── forensics_adapter_fp16.onnx
│   └── vendor/                         # gitignored, extension 실행 필수 런타임 파일
│       ├── ort.all.min.mjs
│       ├── ort-wasm-*.wasm
│       └── mediapipe/
│           ├── vision_bundle.mjs
│           └── wasm/
│               ├── vision_wasm_internal.js
│               ├── vision_wasm_internal.wasm
│               ├── vision_wasm_module_internal.js
│               ├── vision_wasm_module_internal.wasm
│               ├── vision_wasm_nosimd_internal.js
│               └── vision_wasm_nosimd_internal.wasm
├── artifacts/
│   ├── mediapipe/                      # gitignored, prepare 스크립트가 캐시
│   │   └── blaze_face_short_range.tflite
│   └── onnx/                           # gitignored, 원본/변환 모델 저장소
│       ├── face_detector.onnx
│       ├── forensics_adapter.onnx
│       ├── forensics_adapter.webgpu.fp16.onnx
│       ├── model_bnb4.onnx
│       ├── model_q4.onnx
│       ├── model_quantized.onnx
│       └── model_uint8.onnx
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
- `artifacts/mediapipe/blaze_face_short_range.tflite`: MediaPipe detector 캐시
- `extension/models/*`: 실제 Chrome extension이 직접 로드하는 모델 복사본
- `extension/vendor/*`: ORT Web 및 MediaPipe runtime 번들

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
