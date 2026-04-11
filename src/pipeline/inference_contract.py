CLASSIFIER_INPUT_RESOLUTION = 256
CLASSIFIER_PATCH_SIZE = 16
CLASSIFIER_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLASSIFIER_STD = (0.26862954, 0.26130258, 0.27577711)
HF_CLASSIFIER_MEAN = (0.5, 0.5, 0.5)
HF_CLASSIFIER_STD = (0.5, 0.5, 0.5)

DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_MIN_FACE_SIZE = 64
DEFAULT_TOP_K = 3
DEFAULT_FAKE_THRESHOLD = 0.5
DEFAULT_PIPELINE_MARGIN = 0.25
EXTENSION_BROWSER_MARGIN = 0.4

CLASSIFIER_MODEL_FILENAME = "model.onnx"
DETECTOR_MODEL_FILENAME = "face_detector.onnx"
EXTENSION_CONTRACT_FILENAME = "inference_contract.json"


def classifier_patch_count(resolution=CLASSIFIER_INPUT_RESOLUTION):
    return (int(resolution) // CLASSIFIER_PATCH_SIZE) ** 2


def compute_face_label(
    fake_prob,
    pred_label_id,
    fake_threshold=DEFAULT_FAKE_THRESHOLD,
):
    return "fake" if int(pred_label_id) == 1 and float(fake_prob) >= float(fake_threshold) else "real"


def build_no_face_result(num_detected_faces):
    return {
        "status": "no_face",
        "image_fake_prob": None,
        "image_pred_label": "real",
        "num_detected_faces": int(num_detected_faces),
        "num_faces": 0,
        "faces": [],
    }


def build_failed_result(num_detected_faces=None, num_failed_faces=None):
    result = {
        "status": "failed",
        "image_fake_prob": None,
        "image_pred_label": "real",
        "num_faces": 0,
        "faces": [],
    }

    if num_detected_faces is not None:
        result["num_detected_faces"] = int(num_detected_faces)
    if num_failed_faces is not None:
        result["num_failed_faces"] = int(num_failed_faces)

    return result


def aggregate_face_predictions(
    face_predictions,
    fake_threshold=DEFAULT_FAKE_THRESHOLD,
    num_detected_faces=None,
    num_failed_faces=None,
):
    if not face_predictions:
        return build_failed_result(
            num_detected_faces=num_detected_faces,
            num_failed_faces=num_failed_faces,
        )

    fake_threshold = float(fake_threshold)
    fake_probs = [float(entry["fake_prob"]) for entry in face_predictions]
    max_fake_prob = max(fake_probs)
    mean_fake_prob = sum(fake_probs) / len(fake_probs)
    selected_face_index = int(fake_probs.index(max_fake_prob))
    image_pred_label = "fake" if max_fake_prob >= fake_threshold else "real"

    result = {
        "status": "ok",
        "image_fake": 1 if image_pred_label == "fake" else 0,
        "image_fake_prob": float(max_fake_prob),
        "image_pred_label": image_pred_label,
        "num_faces": len(face_predictions),
        "faces": face_predictions,
        "summary": {
            "max_fake_prob": float(max_fake_prob),
            "mean_fake_prob": float(mean_fake_prob),
            "selected_face_index": selected_face_index,
            "fake_threshold": fake_threshold,
        },
    }

    if num_detected_faces is not None:
        result["num_detected_faces"] = int(num_detected_faces)
    if num_failed_faces is not None:
        result["num_failed_faces"] = int(num_failed_faces)

    return result


def build_extension_contract():
    return {
        "contractVersion": 1,
        "classifier": {
            "modelFile": CLASSIFIER_MODEL_FILENAME,
            "inputSize": CLASSIFIER_INPUT_RESOLUTION,
            "ifBoundaryLength": classifier_patch_count(),
            "mean": list(CLASSIFIER_MEAN),
            "std": list(CLASSIFIER_STD),
            "hfMean": list(HF_CLASSIFIER_MEAN),
            "hfStd": list(HF_CLASSIFIER_STD),
            "fakeThreshold": DEFAULT_FAKE_THRESHOLD,
        },
        "selection": {
            "minConfidence": DEFAULT_MIN_CONFIDENCE,
            "minFaceSize": DEFAULT_MIN_FACE_SIZE,
            "topK": DEFAULT_TOP_K,
        },
        "browser": {
            "margin": EXTENSION_BROWSER_MARGIN,
        },
        "detector": {
            "modelFile": DETECTOR_MODEL_FILENAME,
        },
    }
