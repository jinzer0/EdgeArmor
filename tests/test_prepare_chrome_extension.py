import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import onnx
from onnx import TensorProto, helper


def create_valid_onnx(path):
    node = helper.make_node("Identity", inputs=["x"], outputs=["y"])
    graph = helper.make_graph(
        [node],
        "test-graph",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph)
    onnx.save(model, path)


class PrepareChromeExtensionTests(unittest.TestCase):
    def test_resolve_classifier_source_skips_invalid_preferred_model(self):
        module = importlib.import_module("scripts.prepare_chrome_extension")
        importlib.reload(module)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            invalid_model = root / "model_q4f16.onnx"
            invalid_model.write_text("not an onnx file", encoding="utf-8")

            valid_legacy = root / "forensics_adapter.onnx"
            create_valid_onnx(valid_legacy)

            with patch.object(module, "ONNX_DIR", temp_dir):
                resolved = module.resolve_classifier_source()

            self.assertEqual(resolved, "forensics_adapter.onnx")

    def test_forced_classifier_must_be_valid(self):
        module = importlib.import_module("scripts.prepare_chrome_extension")
        importlib.reload(module)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            invalid_model = root / "model_q4f16.onnx"
            invalid_model.write_text("not an onnx file", encoding="utf-8")

            with patch.object(module, "ONNX_DIR", temp_dir), patch.dict(
                os.environ,
                {module.CLASSIFIER_ENV_VAR: "model_q4f16.onnx"},
                clear=False,
            ):
                with self.assertRaises(ValueError):
                    module.resolve_classifier_source()


if __name__ == "__main__":
    unittest.main()
