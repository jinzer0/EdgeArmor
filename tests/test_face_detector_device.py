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
        module._torch_module = lambda: fake_torch
        module._hf_hub_download = fake_hub.hf_hub_download
        module._yolo_class = lambda: FakeYOLO
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
