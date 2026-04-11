import importlib
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np


class _DummyImage:
    size = (100, 100)
    width = 100
    height = 100


class _Input:
    def __init__(self, name, input_type="tensor(float)", shape=None):
        self.name = name
        self.type = input_type
        self.shape = shape or [1]


class _Output:
    def __init__(self, name, shape=None):
        self.name = name
        self.shape = shape or [1]


class _Session:
    def __init__(self, inputs, outputs):
        self._inputs = inputs
        self._outputs = outputs

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, *_args, **_kwargs):
        return [np.array([[0.0, 0.0, 10.0, 10.0, 0.9, 0.0]], dtype=np.float32)]


def import_verify_module(record):
    fake_torch = types.SimpleNamespace(manual_seed=lambda seed: record.update(seed=seed))
    fake_onnx = types.SimpleNamespace(
        load=lambda path: {"path": path},
        checker=types.SimpleNamespace(check_model=lambda model: None),
    )
    fake_ort = types.SimpleNamespace(InferenceSession=lambda path, providers=None: object())

    face_detection_module = types.ModuleType("src.detection.FaceDetection")

    class FakeFaceDetector:
        def __init__(self, margin=0.25, device=None):
            record["face_detector_device"] = device
            self.margin = margin
            self.device = device

        def detect_faces(self, image):
            return [{"bbox": (0, 0, 10, 10), "confidence": 0.9, "landmarks": None}]

        def crop_and_align_faces(self, image, detections, margin=None):
            return [_DummyImage()]

    setattr(face_detection_module, "FaceDetector", FakeFaceDetector)

    pipeline_module = types.ModuleType("src.pipeline.deepfake_pipeline")

    class FakePipeline:
        def __init__(self, config_path, weights_path, device=None, **kwargs):
            record["pipeline_device"] = device

        def predict(self, image):
            return {
                "status": "ok",
                "faces": [],
                "num_faces": 0,
                "image_fake_prob": 0.1,
                "image_pred_label": "real",
            }

    setattr(pipeline_module, "DeepfakeDetectionPipeline", FakePipeline)

    selector_module = types.ModuleType("src.pipeline.face_selector")
    setattr(selector_module, "select_faces", lambda detections, **kwargs: detections[:1])

    infer_module = types.ModuleType("src.pipeline.forensics_adapter_infer")

    class FakeInfer:
        def __init__(self, config_path, weights_path, device=None):
            record["infer_device"] = device
            self.config = {
                "resolution": 256,
                "mean": [0.1, 0.2, 0.3],
                "std": [0.4, 0.5, 0.6],
            }

        def predict(self, face_image):
            return {"fake_prob": 0.1, "pred_label": 0, "logits": [1.0, 0.0]}

    setattr(infer_module, "ForensicsAdapterInfer", FakeInfer)

    contract_module = types.ModuleType("src.pipeline.inference_contract")
    setattr(contract_module, "DEFAULT_FAKE_THRESHOLD", 0.5)
    setattr(contract_module, "DEFAULT_MIN_CONFIDENCE", 0.5)
    setattr(contract_module, "DEFAULT_MIN_FACE_SIZE", 64)
    setattr(contract_module, "DEFAULT_PIPELINE_MARGIN", 0.25)
    setattr(contract_module, "DEFAULT_TOP_K", 3)
    setattr(
        contract_module,
        "aggregate_face_predictions",
        lambda faces, **kwargs: {
            "status": "ok",
            "faces": faces,
            "image_fake_prob": 0.1,
            "image_pred_label": "real",
        },
    )
    setattr(contract_module, "compute_face_label", lambda fake_prob, pred_label_id, fake_threshold=0.5: "fake" if pred_label_id == 1 and fake_prob >= fake_threshold else "real")

    preprocess_module = types.ModuleType("src.pipeline.preprocess")
    setattr(preprocess_module, "build_face_image_tensor", lambda *args, **kwargs: types.SimpleNamespace(numpy=lambda: np.zeros((1, 3, 256, 256), dtype=np.float32)))
    setattr(preprocess_module, "create_if_boundary", lambda *args, **kwargs: types.SimpleNamespace(numpy=lambda: np.ones((1, 256), dtype=np.float32)))

    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "onnx": fake_onnx,
            "onnxruntime": fake_ort,
            "src.detection.FaceDetection": face_detection_module,
            "src.pipeline.deepfake_pipeline": pipeline_module,
            "src.pipeline.face_selector": selector_module,
            "src.pipeline.forensics_adapter_infer": infer_module,
            "src.pipeline.inference_contract": contract_module,
            "src.pipeline.preprocess": preprocess_module,
        },
    ):
        sys.modules.pop("scripts.verify_onnx_exports", None)
        module = importlib.import_module("scripts.verify_onnx_exports")
        importlib.reload(module)
        setattr(
            module,
            "importlib",
            types.SimpleNamespace(
                import_module=lambda name: fake_torch if name == "torch" else importlib.import_module(name),
            ),
        )
        return module


class VerifyOnnxExportsDeviceBoundaryTests(unittest.TestCase):
    def test_verify_script_uses_explicit_cpu_device_across_components(self):
        record = {}
        module = import_verify_module(record)

        setattr(module, "parse_args", lambda: types.SimpleNamespace(
            image="image.png",
            weights_path="ckpt_best.pth",
            config_path="config.yaml",
            detector_onnx="detector.onnx",
            classifier_onnx="classifier.onnx",
            seed=0,
            classifier_atol=1e-3,
            min_confidence=0.5,
            min_face_size=64,
            top_k=3,
            margin=0.25,
            fake_threshold=0.5,
        ))
        setattr(module, "load_sessions", lambda detector_path, classifier_path: (
            _Session([_Input("detector_input")], [_Output("detector_output")]),
            _Session([_Input("image"), _Input("if_boundary")], [_Output("logits"), _Output("fake_prob"), _Output("xray_pred")]),
        ))
        setattr(module, "load_image", lambda image_path: _DummyImage())
        setattr(module, "letterbox_image", lambda image: (object(), {"scale": 1.0, "pad_x": 0, "pad_y": 0, "orig_width": 100, "orig_height": 100, "input_size": 640}))
        setattr(module, "image_to_onnx_input", lambda image: np.zeros((1, 3, 640, 640), dtype=np.float32))
        setattr(module, "parse_detector_output", lambda output_array: [{"bbox": [0.0, 0.0, 10.0, 10.0], "confidence": 0.9, "class_id": 0}])
        setattr(module, "scale_detections_back", lambda detections, metadata: [{"bbox": (0, 0, 10, 10), "confidence": 0.9, "landmarks": None}])
        setattr(module, "run_classifier_onnx", lambda session, inputs: {
            "logits": np.array([[1.0, 0.0]], dtype=np.float32),
            "fake_prob": np.array([0.1], dtype=np.float32),
            "xray_pred": np.zeros((1, 1, 256, 256), dtype=np.float32),
        })

        with redirect_stdout(io.StringIO()):
            module.main()

        self.assertEqual(record["face_detector_device"], "cpu")
        self.assertEqual(record["infer_device"], "cpu")
        self.assertEqual(record["pipeline_device"], "cpu")
        self.assertEqual(record["seed"], 0)
