
# EdgeArmor

[![Platform: Chrome & Edge](https://img.shields.io/badge/Platform-Chrome%20%7C%20Edge-blue.svg)](#quick-start)
[![Status: In Progress](https://img.shields.io/badge/Status-In%20Progress-green.svg)](#)

> 온디바이스 기반 딥페이크 이미지 탐지 및 방어 브라우저 익스텐션
>
> 서버 전송 없이, 브라우저 안에서만 탐지하고 방어합니다.
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
| AI / ML | TensorFlow.js, ONNX Runtime Web |
| Build | Webpack / Vite |
---
## Quick Start
> Work In Progress - 변경될 수 있음!

- ONNX export
  - `conda run -n ml3_13 python scripts/export_detector_onnx.py --out_dir artifacts/onnx`
  - `conda run -n ml3_13 python scripts/export_forensics_adapter_onnx.py --weights_path ckpt_best.pth --config_path ForensicsAdapter/config/test.yaml --out_dir artifacts/onnx`
- Browser extension
  - `npm install`
  - `python scripts/prepare_chrome_extension.py`
  - Chrome `chrome://extensions`에서 `extension/` 폴더를 unpacked extension으로 로드
  - 상세 내용은 `docs/browser_extension.md` 참고
---
## Project Structure
> Work In Progress - 변경될 수 있음!



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
