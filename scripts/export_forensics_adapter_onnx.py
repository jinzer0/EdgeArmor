import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import onnx
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
    parser.add_argument("--filename", default="forensics_adapter.onnx", help="Exported ONNX filename")
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
    contract["config"] = {
        "resolution": resolution,
        "patch_num": patch_num,
        "mean": infer.config["mean"],
        "std": infer.config["std"],
    }
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
