# face detect -> crop -> deepfake detect 2-stage pipeline 구현 메모

## 목표

입력 이미지 1장을 받아서 아래 순서로 처리하는 파이프라인을 만든다.

1. 얼굴 검출
2. 얼굴 crop
3. crop된 얼굴별 deepfake 판별
4. 얼굴별 결과를 합쳐 최종 이미지 단위 결과 반환

---

## 현재 코드베이스 기준 핵심 분석

### 1. 얼굴 검출은 이미 바로 쓸 수 있다

- `src/detection/FaceDetection.py`
- `FaceDetector.detect_faces()`는 bbox와 confidence를 반환한다.
- `FaceDetector.crop_and_align_faces()`는 bbox 기준으로 얼굴 crop을 만든다.
- 다만 현재 detection 결과의 `landmarks`는 항상 `None`이라서, 정렬(`_align_face`)은 사실상 동작하지 않는다.
- 즉 지금 상태는 "face detect + margin crop"까지는 바로 가능하고, "정렬된 face crop"은 아직 완성형은 아니다.

### 2. ForensicsAdapter는 단일 이미지 inference wrapper가 없다

- `ForensicsAdapter/test.py`는 데이터셋 평가 스크립트다.
- 현재 구조는 JSON 기반 dataset loader를 통해 `data_dict`를 만든 뒤 `DS` 모델에 넣는 방식이다.
- 즉 지금은 "crop된 얼굴 이미지 1장"을 바로 넣어 추론하는 entrypoint가 없다.

### 3. DS 모델은 단순히 image tensor 하나만 받지 않는다

- `ForensicsAdapter/model/ds.py`
- `DS.forward()`는 `data_dict['image']` 외에도 `data_dict['if_boundary']`, `data_dict['label']` 등을 참조한다.
- 학습 모드가 아니면 `patch_label`, `clip_patch_label`은 사실상 필요 없지만, `if_boundary`와 `label`은 여전히 들어가야 안전하다.

### 4. 현재 dataset이 해주던 전처리를 inference용으로 따로 빼야 한다

- `ForensicsAdapter/dataset/abstract_dataset.py`
- dataset은 아래 작업을 자동으로 해준다.
  - 이미지 resize
  - tensor 변환
  - normalize
  - mask 기반 `if_boundary` 생성
- 하지만 우리가 만들 파이프라인은 dataset JSON을 거치지 않으므로, crop face 이미지를 직접 `data_dict`로 바꿔주는 별도 어댑터가 필요하다.

### 5. device 선택 로직이 분리되어 있다

- `src/detection/FaceDetection.py`는 `mps -> cuda -> cpu` 순서로 device를 고른다.
- `ForensicsAdapter/config/test.yaml`은 `device: 'cuda:0'`으로 고정돼 있다.
- `ForensicsAdapter/test.py`는 또 별도 `device = torch.device("cuda" if ... else "cpu")`를 쓴다.
- 파이프라인에서는 detector와 deepfake model이 같은 device 정책을 쓰도록 통일해야 한다.

### 6. 체크포인트는 레포에 없다

- 현재 레포 안에서 `.pth` 파일은 보이지 않는다.
- 따라서 실제 추론 파이프라인을 돌리려면 deepfake detector weight path를 외부에서 주입받는 구조가 필요하다.

---

## 구현 방향

가장 안전한 구현 방향은 아래처럼 "탐지기"와 "딥페이크 추론기"를 분리한 뒤, 마지막에 orchestration 레이어로 묶는 것이다.

1. `FaceDetector`는 그대로 사용
2. `ForensicsAdapter`용 단일 이미지 inference wrapper 추가
3. 두 모듈을 연결하는 pipeline 클래스 추가
4. CLI 또는 테스트 스크립트 추가

이 방향이 좋은 이유는 기존 학습 코드와 추론 파이프라인을 분리할 수 있고, 나중에 web app이나 extension에서도 wrapper만 재사용할 수 있기 때문이다.

---

## step by step 구현 계획

### Step 1. ForensicsAdapter 단일 이미지 추론 래퍼를 만든다

추천 파일:

- `src/pipeline/forensics_adapter_infer.py`

여기서 해야 할 일:

1. `ForensicsAdapter/config/test.yaml`을 읽는다.
2. 실행 가능한 실제 device를 다시 결정한다.
3. `DS(...)`를 생성한다.
4. 외부에서 받은 checkpoint path로 weight를 load한다.
5. `predict(face_image)` 형태의 메서드를 노출한다.

이 래퍼가 중요한 이유:

- 현재 `ForensicsAdapter/test.py`는 dataset loader 중심이라서 재사용성이 낮다.
- 우리가 필요한 것은 "PIL 이미지 1장 -> score 1개" 인터페이스다.

이 파일에서 만들 최소 인터페이스 예시:

```python
class ForensicsAdapterInfer:
    def __init__(self, config_path, weights_path, device=None):
        ...

    def predict(self, face_image):
        ...
        return {
            "fake_prob": ...,
            "pred_label": ...,
            "xray_pred": ...,
        }
```

주의할 점:

- `DS.forward(..., inference=True)`를 써야 한다.
- 이때 `label`은 dummy 값으로라도 넣어야 한다.
- `if_boundary`도 넣어야 한다.

권장 처리:

- `label`: `torch.LongTensor([0])`
- `if_boundary`: shape이 맞는 all-ones tensor
- `mask`, `landmark`, `xray`, `patch_label`, `clip_patch_label`: `None` 또는 dummy

이렇게 가는 이유:

- inference 경로에서는 patch label 계열이 사실상 쓰이지 않는다.
- 반면 `if_boundary`는 `DS.forward()` 내부에서 바로 `.to(self.device)`가 호출되므로 빠지면 안 된다.

### Step 2. face crop 이미지를 DS 입력 형태로 바꾸는 전처리 함수를 만든다

추천 파일:

- `src/pipeline/preprocess.py`

여기서 해야 할 일:

1. PIL face image를 RGB로 보정
2. `resolution=256`으로 resize
3. tensor 변환
4. `mean/std` normalize 적용
5. batch 차원 추가
6. 최소 `data_dict` 구성

중요한 기준:

- 전처리는 반드시 `ForensicsAdapter/dataset/abstract_dataset.py`와 최대한 동일하게 맞춘다.
- 특히 normalize 값은 `ForensicsAdapter/config/test.yaml` 기준을 그대로 써야 한다.

최소 `data_dict` 예시:

```python
{
    "image": image_tensor,
    "label": dummy_label,
    "landmark": None,
    "mask": None,
    "xray": None,
    "patch_label": None,
    "clip_patch_label": None,
    "if_boundary": if_boundary_tensor,
}
```

권장 shape:

- `image`: `[1, 3, 256, 256]`
- `label`: `[1]`
- `if_boundary`: `[1, 256]`

왜 `if_boundary`가 `[1, 256]`이냐:

- `MaskPostXrayProcess`는 16x16 patch 기준으로 `if_boundaries`를 사용한다.
- 즉 patch 개수 256개에 해당하는 boundary mask가 필요하다.

### Step 3. 얼굴 선택 정책을 먼저 정한다

추천 파일:

- `src/pipeline/face_selector.py`

현재 `FaceDetector.detect_faces()`는 얼굴을 여러 개 반환할 수 있다. 그래서 pipeline에서 아래 정책을 명확히 정해야 한다.

권장 정책:

1. confidence threshold 이하 bbox 제거
2. 너무 작은 얼굴 제거
3. 남은 얼굴 중 상위 `top_k`만 사용

초기 추천값:

- `min_confidence = 0.5`
- `min_face_size = 64`
- `top_k = 3`

얼굴 정렬 관련 판단:

- 현재는 landmark가 없어서 정렬 효과를 기대하기 어렵다.
- 따라서 1차 구현에서는 `crop + resize`만으로 충분하다.
- 나중에 detector를 landmark 지원 모델로 바꾸면 정렬 단계를 강화하면 된다.

### Step 4. 전체 파이프라인 클래스를 만든다

추천 파일:

- `src/pipeline/deepfake_pipeline.py`

이 파일이 실제 orchestration 레이어다.

권장 흐름:

1. 입력 이미지 로드
2. face detection 수행
3. bbox 필터링
4. face crop 생성
5. 각 crop에 대해 ForensicsAdapter inference 수행
6. 얼굴별 결과 집계
7. 최종 결과 반환

권장 인터페이스:

```python
class DeepfakeDetectionPipeline:
    def __init__(self, detector, deepfake_model, min_confidence=0.5, top_k=3):
        ...

    def predict(self, image):
        ...
        return {
            "num_faces": ...,
            "faces": [...],
            "image_fake_prob": ...,
            "image_pred_label": ...,
        }
```

얼굴별 결과 예시:

```python
{
    "bbox": [x1, y1, x2, y2],
    "det_confidence": 0.91,
    "fake_prob": 0.82,
    "pred_label": "fake",
}
```

### Step 5. 이미지 단위 최종 score 집계 규칙을 정한다

이 부분은 구현 전에 반드시 정해야 한다.

가장 실용적인 초기 정책:

1. 얼굴이 0개면 `status=no_face` 반환
2. 얼굴이 1개면 그 결과를 최종 결과로 사용
3. 얼굴이 여러 개면 가장 큰 얼굴 또는 가장 높은 fake score 얼굴을 최종 결과로 사용

초기 추천:

- 기본 집계는 `max(fake_prob)` 사용

이유:

- 딥페이크 이미지에서는 한 얼굴만 조작돼도 전체 이미지를 fake로 보는 쪽이 보수적이다.
- 브라우저 확장이나 실사용 환경에서도 `max` 정책이 이해하기 쉽다.

추가로 같이 반환하면 좋은 값:

- `max_fake_prob`
- `mean_fake_prob`
- `selected_face_index`

### Step 6. no-face / multi-face / 실패 케이스를 명시적으로 처리한다

권장 예외 처리:

1. 얼굴이 없으면 예외 대신 정상 응답 반환
2. crop 중 실패한 얼굴은 skip
3. 모든 얼굴 추론이 실패하면 `status=failed`
4. 모델 weight가 없으면 초기화 단계에서 명확한 에러 발생

반환 형식 예시:

```python
{
    "status": "ok",
    "image_fake_prob": 0.82,
    "num_faces": 2,
    "faces": [...]
}
```

또는

```python
{
    "status": "no_face",
    "image_fake_prob": None,
    "num_faces": 0,
    "faces": []
}
```

### Step 7. 실행용 스크립트를 하나 만든다

추천 파일:

- `scripts/run_pipeline.py`

이 스크립트에서 할 일:

1. 입력 이미지 path 받기
2. checkpoint path 받기
3. pipeline 초기화
4. 결과 출력

추천 CLI 예시:

```bash
python scripts/run_pipeline.py \
  --image path/to/image.jpg \
  --weights path/to/ckpt_best.pth
```

출력은 처음에는 JSON 비슷한 dict print만 해도 충분하다.

### Step 8. 디버깅용 산출물을 함께 저장한다

초기 구현에서는 아래 저장 옵션을 넣는 것이 좋다.

추천 출력:

1. 원본 이미지에 bbox 그린 시각화
2. crop된 face 이미지들
3. 얼굴별 fake score

추천 경로:

- `outputs/debug/`

이걸 해두면 아래 문제를 바로 확인할 수 있다.

- 얼굴 검출이 엉뚱한 위치를 잡는지
- crop margin이 너무 큰지
- 작은 얼굴에서 score가 불안정한지

---

## 가장 중요한 구현 포인트

### 1. `ForensicsAdapter/test.py`를 직접 재사용하기보다는 로직만 가져오는 게 낫다

이유:

- 지금 파일은 dataset 평가 스크립트라서 파이프라인용 진입점으로는 무겁다.
- 필요한 것은 모델 생성, weight load, single-image inference뿐이다.

### 2. inference용 `data_dict`를 별도로 정의해야 한다

현재 모델 구조상 이것이 핵심이다.

- `image`만 넣는 식으로는 바로 못 쓴다.
- 최소 입력 스펙을 wrapper에서 강제로 맞춰줘야 한다.

### 3. detector와 deepfake model의 device를 하나로 통일해야 한다

권장 함수:

```python
def resolve_device():
    ...
```

이 함수를 만들어 두 모델이 같은 장치에서 동작하도록 맞추는 편이 안전하다.

### 4. 첫 버전은 "정렬"보다 "안정적인 crop"이 더 중요하다

현재 detector가 landmark를 채우지 않기 때문에, 정렬 품질에 기대기 어렵다.
따라서 1차 목표는 아래 둘이다.

1. 얼굴 bbox를 안정적으로 얻기
2. crop face를 ForensicsAdapter에 안정적으로 넣기

---

## 추천 구현 순서

실제 작업 순서는 아래가 가장 무난하다.

1. `src/pipeline/preprocess.py` 작성
2. `src/pipeline/forensics_adapter_infer.py` 작성
3. 단일 crop face 이미지 1장으로 deepfake inference 확인
4. `src/pipeline/deepfake_pipeline.py` 작성
5. `scripts/run_pipeline.py` 작성
6. debug output 저장 추가
7. multi-face 집계 정책 튜닝

---

## 바로 구현할 때의 최소 TODO

- [ ] `src/pipeline/` 디렉토리 생성
- [ ] `preprocess.py`에서 crop face -> `data_dict` 변환 구현
- [ ] `forensics_adapter_infer.py`에서 DS load + checkpoint load 구현
- [ ] `deepfake_pipeline.py`에서 detect -> crop -> predict 연결
- [ ] `run_pipeline.py`에서 CLI 실행 가능하게 만들기
- [ ] no-face / multi-face 처리 추가
- [ ] debug 이미지 저장 추가

---

## 결론

현재 코드베이스에서는 얼굴 검출 단계는 이미 거의 준비돼 있고, 실제 병목은 `ForensicsAdapter`를 "dataset 평가 코드"에서 "단일 face crop 추론기"로 감싸는 것이다.

즉 구현의 핵심은 아래 한 줄로 정리된다.

`crop된 얼굴 PIL 이미지 -> ForensicsAdapter가 요구하는 최소 data_dict로 변환 -> DS inference -> 얼굴별 score 집계`

이 구조로 가면 원하는 `input image -> face detect -> crop face image input -> deepfake detect` 2-stage pipeline을 가장 적은 변경으로 붙일 수 있다.
