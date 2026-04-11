import importlib
import sys
import types
import unittest
from unittest.mock import patch


def import_deepfake_pipeline(fake_classifier_device):
    created_detectors = []
    face_detection_module = types.ModuleType("src.detection.FaceDetection")

    class FakeFaceDetector:
        def __init__(self, margin=0.25, device=None):
            self.margin = margin
            self.device = device
            created_detectors.append(self)

        def _load_image(self, image):
            return image

        def detect_faces(self, image):
            return []

        def crop_and_align_faces(self, image, detections, margin=None):
            return []

    setattr(face_detection_module, "FaceDetector", FakeFaceDetector)

    infer_module = types.ModuleType("src.pipeline.forensics_adapter_infer")

    class FakeForensicsAdapterInfer:
        def __init__(self, config_path, weights_path, device=None):
            self.config_path = config_path
            self.weights_path = weights_path
            self.requested_device = device
            self.device = fake_classifier_device

        def predict(self, crop):
            return {"fake_prob": 0.1, "pred_label": 0, "logits": [1.0, 0.0]}

    setattr(infer_module, "ForensicsAdapterInfer", FakeForensicsAdapterInfer)

    with patch.dict(
        sys.modules,
        {
            "src.detection.FaceDetection": face_detection_module,
            "src.pipeline.forensics_adapter_infer": infer_module,
        },
    ):
        sys.modules.pop("src.pipeline.deepfake_pipeline", None)
        module = importlib.import_module("src.pipeline.deepfake_pipeline")
        importlib.reload(module)
        return module, face_detection_module, created_detectors


class DeepfakePipelineDeviceBoundaryTests(unittest.TestCase):
    def test_pipeline_builds_default_detector_from_classifier_device(self):
        module, face_detection_module, created_detectors = import_deepfake_pipeline(
            fake_classifier_device="cuda:0",
        )

        pipeline = module.DeepfakeDetectionPipeline(
            config_path="config.yaml",
            weights_path="ckpt_best.pth",
            device="cuda:0",
        )

        self.assertEqual(pipeline.device, "cuda:0")
        self.assertEqual(len(created_detectors), 1)
        self.assertEqual(created_detectors[0].margin, 0.25)
        self.assertEqual(created_detectors[0].device, "cuda:0")
        self.assertFalse(hasattr(face_detection_module, "_DEVICE"))

    def test_pipeline_keeps_injected_detector_untouched(self):
        module, face_detection_module, created_detectors = import_deepfake_pipeline(
            fake_classifier_device="cpu",
        )
        custom_detector = object()

        pipeline = module.DeepfakeDetectionPipeline(
            config_path="config.yaml",
            weights_path="ckpt_best.pth",
            detector=custom_detector,
        )

        self.assertIs(pipeline.detector, custom_detector)
        self.assertEqual(pipeline.device, "cpu")
        self.assertEqual(created_detectors, [])
        self.assertFalse(hasattr(face_detection_module, "_DEVICE"))
