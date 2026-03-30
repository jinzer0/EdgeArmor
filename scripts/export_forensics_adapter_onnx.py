import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import onnx
import onnxruntime as ort
import torch
from torch import nn

from src.pipeline.forensics_adapter_infer import ForensicsAdapterInfer


class ForensicsAdapterExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, if_boundary):
        logits, fake_prob, xray_pred = self.model.forward_export(image, if_boundary)
        return logits, fake_prob, xray_pred


def parse_args():
    parser = argparse.ArgumentParser(description="Export the ForensicsAdapter classifier to ONNX.")
    parser.add_argument("--weights_path", required=True, help="Path to ckpt_best.pth")
    parser.add_argument("--config_path", default="ForensicsAdapter/config/test.yaml", help="Path to YAML config")
    parser.add_argument("--out_dir", default="artifacts/onnx", help="Directory for exported ONNX files")
    parser.add_argument("--filename", default="forensics_adapter_fp16.onnx", help="Exported ONNX filename")
    parser.add_argument(
        "--browser_filename",
        default="forensics_adapter.webgpu.fp16.onnx",
        help="Browser-targeted FP16 ONNX filename",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for export-time module initialization")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def collect_contract(model_path):
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    inputs = []
    for value in model.graph.input:
        dims = []
        for dim in value.type.tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.dim_value else dim.dim_param)
        inputs.append({"name": value.name, "shape": dims})

    outputs = []
    for value in model.graph.output:
        dims = []
        for dim in value.type.tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.dim_value else dim.dim_param)
        outputs.append({"name": value.name, "shape": dims})

    return {
        "path": model_path,
        "inputs": inputs,
        "outputs": outputs,
    }


def export_browser_model(source_path, output_path):
    from onnx import helper
    from onnxconverter_common import float16

    model = onnx.load(source_path)
    model = float16.convert_float_to_float16(
        model,
        keep_io_types=False,
        disable_shape_infer=True,
        op_block_list=[],
    )

    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        for attribute in node.attribute:
            if attribute.name == "to" and helper.get_attribute_value(attribute) == onnx.TensorProto.FLOAT:
                attribute.i = onnx.TensorProto.FLOAT16

    onnx.checker.check_model(model)
    onnx.save_model(model, output_path)

    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    return {
        "path": output_path,
        "inputs": [{"name": value.name, "type": value.type} for value in session.get_inputs()],
        "outputs": [{"name": value.name, "type": value.type} for value in session.get_outputs()],
    }


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    torch.manual_seed(args.seed)

    infer = ForensicsAdapterInfer(
        config_path=args.config_path,
        weights_path=args.weights_path,
        device="cpu",
    )
    wrapper = ForensicsAdapterExportWrapper(infer.model).eval()

    resolution = int(infer.config["resolution"])
    patch_num = (resolution // 16) ** 2
    image = torch.zeros(1, 3, resolution, resolution, dtype=torch.float32)
    if_boundary = torch.ones(1, patch_num, dtype=torch.float32)

    output_path = os.path.join(args.out_dir, args.filename)
    torch.onnx.export(
        wrapper,
        (image, if_boundary),
        output_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["image", "if_boundary"],
        output_names=["logits", "fake_prob", "xray_pred"],
        dynamo=False,
    )

    contract = collect_contract(output_path)
    browser_output_path = os.path.join(args.out_dir, args.browser_filename)
    browser_contract = export_browser_model(output_path, browser_output_path)
    contract["config"] = {
        "resolution": resolution,
        "patch_num": patch_num,
        "mean": infer.config["mean"],
        "std": infer.config["std"],
    }
    contract["browser_webgpu_fp16"] = browser_contract
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
