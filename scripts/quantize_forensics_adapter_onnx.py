import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def parse_args():
    parser = argparse.ArgumentParser(description="Create INT8 ONNX model for forensics adapter.")
    parser.add_argument(
        "--source",
        default="artifacts/onnx/forensics_adapter.onnx",
        help="Source FP32 ONNX path",
    )
    parser.add_argument(
        "--output",
        default="artifacts/onnx/forensics_adapter.int8.onnx",
        help="Output INT8 ONNX path",
    )
    parser.add_argument(
        "--weight_type",
        default="qint8",
        choices=["qint8", "quint8"],
        help="Quantized weight type",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip generation when output already exists",
    )
    return parser.parse_args()


def ensure_parent_dir(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def build_contract(source_path, output_path, weight_type, skipped, file_sizes):
    return {
        "source": source_path,
        "output": output_path,
        "method": "onnxruntime.quantization.quantize_dynamic",
        "weight_type": weight_type,
        "skipped": skipped,
        "source_size_mb": round(file_sizes[0], 4),
        "output_size_mb": round(file_sizes[1], 4),
    }


def main():
    args = parse_args()

    source_path = args.source
    output_path = args.output

    if not os.path.isfile(source_path):
        raise SystemExit(f"Source model not found: {source_path}")

    if args.skip_existing and os.path.isfile(output_path):
        source_size = os.path.getsize(source_path) / (1024 * 1024)
        output_size = os.path.getsize(output_path) / (1024 * 1024)
        contract = build_contract(
            source_path,
            output_path,
            args.weight_type,
            True,
            (source_size, output_size),
        )
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        return

    ensure_parent_dir(output_path)
    onnx_model = onnx.load(source_path)
    onnx.checker.check_model(onnx_model)

    weight_type = QuantType.QInt8 if args.weight_type == "qint8" else QuantType.QUInt8
    quantize_dynamic(
        source_path,
        output_path,
        weight_type=weight_type,
    )

    quantized_model = onnx.load(output_path)
    onnx.checker.check_model(quantized_model)

    source_size = os.path.getsize(source_path) / (1024 * 1024)
    output_size = os.path.getsize(output_path) / (1024 * 1024)
    contract = build_contract(
        source_path,
        output_path,
        args.weight_type,
        False,
        (source_size, output_size),
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
