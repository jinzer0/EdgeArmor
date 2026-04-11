import importlib
import sys
import types
import unittest
from unittest.mock import patch


class DummyCrop:
    def __init__(self, width=32, height=24):
        self.width = width
        self.height = height

    def save(self, _path):
        return None


class DummyImage:
    def copy(self):
        return self


def import_deepfake_pipeline(fake_predict):
    face_detection_module = types.ModuleType("src.detection.FaceDetection")

    class FakeFaceDetector:
        def __init__(self, margin=0.25, device=None):
            self.margin = margin
            self.device = device

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
            self.device = device or "cpu"

        def predict(self, crop):
            return fake_predict(crop)

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
        return module


class DeepfakePipelinePredictTests(unittest.TestCase):
    def test_predict_returns_no_face_when_selection_is_empty(self):
        module = import_deepfake_pipeline(lambda crop: {"fake_prob": 0.1, "pred_label": 0, "logits": [1.0, 0.0]})

        class Detector:
            def _load_image(self, image):
                return DummyImage()

            def detect_faces(self, image):
                return [{"bbox": (0, 0, 20, 20), "confidence": 0.9, "landmarks": None}]

            def crop_and_align_faces(self, image, detections, margin=None):
                return [DummyCrop()]

        pipeline = module.DeepfakeDetectionPipeline(
            config_path="config.yaml",
            weights_path="ckpt_best.pth",
            detector=Detector(),
        )

        with patch.object(module, "select_faces", lambda **kwargs: []):
            result = pipeline.predict("image-path")

        self.assertEqual(result["status"], "no_face")
        self.assertEqual(result["num_detected_faces"], 1)
        self.assertEqual(result["num_faces"], 0)
        self.assertEqual(result["faces"], [])

    def test_predict_returns_failed_when_all_face_predictions_raise(self):
        def raising_predict(_crop):
            raise RuntimeError("boom")

        module = import_deepfake_pipeline(raising_predict)

        class Detector:
            def _load_image(self, image):
                return DummyImage()

            def detect_faces(self, image):
                return [{"bbox": (0, 0, 20, 20), "confidence": 0.9, "landmarks": None}]

            def crop_and_align_faces(self, image, detections, margin=None):
                return [DummyCrop()]

        pipeline = module.DeepfakeDetectionPipeline(
            config_path="config.yaml",
            weights_path="ckpt_best.pth",
            detector=Detector(),
        )

        with patch.object(module, "select_faces", lambda **kwargs: [{"bbox": (0, 0, 20, 20), "confidence": 0.9, "landmarks": None}]):
            result = pipeline.predict("image-path")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["num_detected_faces"], 1)
        self.assertEqual(result["num_failed_faces"], 1)
        self.assertEqual(result["num_faces"], 0)

    def test_predict_keeps_public_face_shape_and_debug_path(self):
        module = import_deepfake_pipeline(lambda crop: {"fake_prob": 0.8, "pred_label": 1, "logits": [0.1, 0.9]})
        debug_calls = {}

        class Detector:
            def _load_image(self, image):
                return DummyImage()

            def detect_faces(self, image):
                return [{"bbox": (1, 2, 21, 22), "confidence": 0.95, "landmarks": None}]

            def crop_and_align_faces(self, image, detections, margin=None):
                return [DummyCrop(width=40, height=50)]

        pipeline = module.DeepfakeDetectionPipeline(
            config_path="config.yaml",
            weights_path="ckpt_best.pth",
            detector=Detector(),
        )
        pipeline._draw_debug = lambda image, face_results, output_dir, prefix: debug_calls.update(
            output_dir=output_dir,
            prefix=prefix,
            face_results=face_results,
        )

        selected = [{"bbox": (1, 2, 21, 22), "confidence": 0.95, "landmarks": None}]
        with patch.object(module, "select_faces", lambda **kwargs: selected):
            result = pipeline.predict(
                "image-path",
                save_debug=True,
                debug_dir="outputs/custom",
                debug_prefix="sample",
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["image_pred_label"], "fake")
        self.assertEqual(result["num_faces"], 1)
        self.assertEqual(result["debug_dir"], "outputs/custom")
        self.assertEqual(result["faces"][0]["crop_size"], [40, 50])
        self.assertNotIn("crop", result["faces"][0])
        self.assertEqual(debug_calls["output_dir"], "outputs/custom")
        self.assertEqual(debug_calls["prefix"], "sample")
        self.assertIn("crop", debug_calls["face_results"][0])


if __name__ == "__main__":
    unittest.main()
