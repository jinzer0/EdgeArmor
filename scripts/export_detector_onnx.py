import argparse
import json
import os
import shutil
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from huggingface_hub import hf_hub_download
import onnx
from ultralytics import YOLO

from src.detection.FaceDetection import MODEL_FILENAME, MODEL_REPO_ID


def parse_args():
    parser = argparse.ArgumentParser(description="Export the YOLOv8 face detector to ONNX.")
    parser.add_argument("--out_dir", default="artifacts/onnx", help="Directory for exported ONNX files")
    parser.add_argument("--filename", default="face_detector.onnx", help="Exported ONNX filename")
    parser.add_argument("--imgsz", type=int, default=640, help="Static export image size")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_export_path(export_result):
    if isinstance(export_result, (list, tuple)) and export_result:
        return str(export_result[0])
    return str(export_result)


def build_contract(model_path):
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

    model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    model = YOLO(model_path)

    export_result = model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=1,
        dynamic=False,
        simplify=True,
        opset=args.opset,
        nms=True,
        half=False,
    )

    exported_path = normalize_export_path(export_result)
    target_path = os.path.join(args.out_dir, args.filename)
    if os.path.abspath(exported_path) != os.path.abspath(target_path):
        shutil.copy2(exported_path, target_path)
    else:
        target_path = exported_path

    contract = build_contract(target_path)
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
