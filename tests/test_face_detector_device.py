import importlib
import sys
import types
import unittest
from unittest.mock import patch


class FakeDevice:
    def __init__(self, spec):
        if isinstance(spec, FakeDevice):
            spec = spec.spec
        self.spec = str(spec)
        self.type = self.spec.split(":", 1)[0]

    def __str__(self):
        return self.spec

    def __repr__(self):
        return f"FakeDevice({self.spec!r})"


def build_fake_torch(cuda_available=False, mps_available=False):
    def device(spec):
        return spec if isinstance(spec, FakeDevice) else FakeDevice(spec)

    return types.SimpleNamespace(
        device=device,
        cuda=types.SimpleNamespace(is_available=lambda: cuda_available),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps_available),
        ),
    )


class FakeYOLO:
    def __init__(self, model_path):
        self.model_path = model_path
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


def import_face_detection(cuda_available=False, mps_available=False):
    fake_torch = build_fake_torch(
        cuda_available=cuda_available,
        mps_available=mps_available,
    )
    fake_hub = types.SimpleNamespace(
        hf_hub_download=lambda repo_id, filename: f"/fake/{repo_id}/{filename}",
    )
    fake_ultralytics = types.SimpleNamespace(YOLO=FakeYOLO)

    with patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "huggingface_hub": fake_hub,
            "ultralytics": fake_ultralytics,
        },
    ):
        sys.modules.pop("src.detection.FaceDetection", None)
        module = importlib.import_module("src.detection.FaceDetection")
        importlib.reload(module)
        setattr(module, "_torch_module", lambda: fake_torch)
        setattr(module, "_hf_hub_download", fake_hub.hf_hub_download)
        setattr(module, "_yolo_class", lambda: FakeYOLO)
        return module


class FaceDetectorDeviceTests(unittest.TestCase):
    def test_default_device_prefers_mps_then_cuda_then_cpu(self):
        module = import_face_detection(cuda_available=True, mps_available=True)
        self.assertEqual(str(module.FaceDetector().device), "mps")

        module = import_face_detection(cuda_available=True, mps_available=False)
        self.assertEqual(str(module.FaceDetector().device), "cuda")

        module = import_face_detection(cuda_available=False, mps_available=False)
        self.assertEqual(str(module.FaceDetector().device), "cpu")

    def test_explicit_device_uses_per_device_model_cache(self):
        module = import_face_detection(cuda_available=True, mps_available=False)
        module.FaceDetector._models_by_device = {}

        cpu_detector = module.FaceDetector(device="cpu")
        gpu_detector = module.FaceDetector(device="cuda")

        cpu_model = cpu_detector._get_model()
        gpu_model = gpu_detector._get_model()
        cpu_model_again = module.FaceDetector(device="cpu")._get_model()

        self.assertIs(cpu_model, cpu_model_again)
        self.assertIsNot(cpu_model, gpu_model)
        self.assertEqual(sorted(module.FaceDetector._models_by_device.keys()), ["cpu", "cuda"])
        self.assertEqual(str(cpu_model.moved_to), "cpu")
        self.assertEqual(str(gpu_model.moved_to), "cuda")

    def test_helper_functions_forward_detector_and_device(self):
        module = import_face_detection(cuda_available=False, mps_available=False)
        created = []

        class RecordingDetector:
            def __init__(self, margin=0.25, device=None):
                self.margin = margin
                self.device = device
                created.append(self)

            def detect_faces(self, image):
                return [("detect", image, self.device)]

            def crop_and_align_faces(self, image, detections, margin=None):
                return [("crop", image, detections, margin, self.device)]

            def process_image(self, image, margin=None):
                return [("process", image, margin, self.device)]

        injected = RecordingDetector(device="cuda")

        with patch.object(module, "FaceDetector", RecordingDetector):
            detect_result = module.detect_faces("img-a", device="cpu")
            crop_result = module.crop_and_align_faces("img-b", [1], margin=0.5, device="mps")
            process_result = module.process_image("img-c", margin=0.4, detector=injected)

        self.assertEqual(detect_result, [("detect", "img-a", "cpu")])
        self.assertEqual(crop_result, [("crop", "img-b", [1], 0.5, "mps")])
        self.assertEqual(process_result, [("process", "img-c", 0.4, "cuda")])
        self.assertEqual(len(created), 3)
        self.assertEqual(created[2].margin, 0.5)
