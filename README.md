
# EdgeArmor

[![Platform: Chrome & Edge](https://img.shields.io/badge/Platform-Chrome%20%7C%20Edge-blue.svg)](#quick-start)
[![Status: In Progress](https://img.shields.io/badge/Status-In%20Progress-green.svg)](#)

> 온디바이스 기반 딥페이크 이미지 탐지 및 방어 브라우저 익스텐션
>
> 서버 전송 없이, 브라우저 안에서만 탐지하고 방어합니다.

## 현재 작업 브랜치: `codex/extension-runtime`    

**[프로젝트 구조 보러가기](#project-structure)**
---
## Highlights

- 실시간 이미지 탐지
- 얼굴 검출 후 얼굴별 딥페이크 판별
- Chrome popup + 로컬 Python 추론 서버
- Chrome / Edge 지원
---
## Stack
> Work In Progress - 변경될 수 있음!

| Layer | Tech |
| :--- | :--- |
| Frontend | HTML5, CSS3, JavaScript (ES6+), Manifest V3 |
| Browser Runtime | ONNX Runtime Web |
| Python Inference | PyTorch, Ultralytics YOLOv8-Face, Pillow |
| Model Assets | ONNX exports, `ckpt_best.pth`, `ViT-L-14.pt` |
---
## Quick Start
> Work In Progress - 변경될 수 있음!

- ONNX export
  - `conda run -n ml3_13 python scripts/export_detector_onnx.py --out_dir artifacts/onnx`
  - `conda run -n ml3_13 python scripts/export_forensics_adapter_onnx.py --weights_path ckpt_best.pth --config_path ForensicsAdapter/config/test.yaml --out_dir artifacts/onnx`
- Browser extension
  - `npm install`
  - `python scripts/prepare_chrome_extension.py`
  - `conda run -n ml3_13 python scripts/run_extension_inference_server.py --weights_path ckpt_best.pth`
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
│   ├── models/                           # gitignored, extension 실행 필수
│   │   ├── face_detector.onnx
│   │   ├── forensics_adapter.onnx
│   │   ├── forensics_adapter_fp16.onnx
│   │   ├── model.onnx
│   │   ├── model_fp16.onnx
│   │   ├── model_int8.onnx
│   │   ├── model_q4f16.onnx
│   │   └── blaze_face_short_range.tflite
│   └── vendor/                           # gitignored, ORT Web 런타임
│       ├── ort.all.min.mjs
│       ├── ort.wasm.min.mjs
│       ├── ort-wasm-simd-threaded.wasm
│       ├── ort-wasm-simd-threaded.mjs
│       ├── ort-wasm-simd-threaded.jsep.wasm
│       ├── ort-wasm-simd-threaded.jsep.mjs
│       ├── ort-wasm-simd-threaded.asyncify.wasm
│       ├── ort-wasm-simd-threaded.asyncify.mjs
│       ├── ort-wasm-simd-threaded.jspi.wasm
│       └── ort-wasm-simd-threaded.jspi.mjs
├── artifacts/
│   └── onnx/                             # gitignored, export 산출물
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
│       ├── __init__.py
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
│   ├── test.py
│   └── visualize_attention.py
├── dataset/
│   └── weight/
│       └── ViT-L-14.pt                  # gitignored, ForensicsAdapter 실행 필수
├── ckpt_best.pth                        # gitignored, 서버/모델 export 필수
└── things.md
```

필수 gitignored 파일 메모:

- `ckpt_best.pth`: Python 딥페이크 분류기 가중치
- `dataset/weight/ViT-L-14.pt`: ForensicsAdapter 백본 로딩에 필요
- `artifacts/onnx/face_detector.onnx`: 브라우저 detector 실험 경로 및 검증 스크립트 입력
- `artifacts/onnx/forensics_adapter.onnx`: 원본 classifier ONNX
- `artifacts/onnx/forensics_adapter.webgpu.fp16.onnx`: extension 배포용 classifier 복사 원본
- `extension/models/*`: popup이 실제로 로드하는 모델 복사본
- `extension/vendor/*`: ORT Web wasm/runtime 파일


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
