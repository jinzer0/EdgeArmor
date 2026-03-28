import argparse
import json
import os
import statistics
import time

import numpy as np
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark ONNX models for forensics adapter.")
    parser.add_argument(
        "--fp32",
        default="artifacts/onnx/forensics_adapter_fp16.onnx",
        help="FP32 ONNX path",
    )
    parser.add_argument(
        "--fp16",
        default="artifacts/onnx/forensics_adapter.webgpu.fp16.onnx",
        help="FP16 ONNX path",
    )
    parser.add_argument(
        "--int8",
        default="artifacts/onnx/forensics_adapter.int8.onnx",
        help="INT8 ONNX path",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=128,
        help="Inference iterations per model",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=16,
        help="Warmup iterations before timing",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print output diff against FP32 model",
    )
    parser.add_argument(
        "--json",
        default="",
        help="Optional JSON output path",
    )
    return parser.parse_args()


def ensure_file(path):
    if not os.path.isfile(path):
        raise SystemExit(f"Model not found: {path}")


def onnx_type_to_numpy(type_name):
    if type_name in ("tensor(float16)", "float16"):
        return np.float16
    if type_name in ("tensor(float)", "float", "tensor(double)"):
        return np.float32 if type_name != "tensor(double)" else np.float64
    if type_name in ("tensor(int64)", "int64"):
        return np.int64
    if type_name in ("tensor(int32)", "int32"):
        return np.int32
    if type_name in ("tensor(bool)", "bool"):
        return np.bool_
    return np.float32


def resolve_shape(shape):
    dims = []
    for dim in shape:
        if isinstance(dim, int):
            dims.append(dim)
        else:
            dims.append(1)
    return tuple(max(1, int(d)) for d in dims)


def build_reference_inputs(input_meta):
    inputs = {}
    rng = np.random.RandomState(0)
    for item in input_meta:
        np_type = onnx_type_to_numpy(item["type"])
        shape = resolve_shape(item["shape"])
        if np.issubdtype(np_type, np.floating):
            data = (rng.rand(*shape).astype(np.float32) * 2.0) - 1.0
            if np_type != np.float32:
                data = data.astype(np_type)
        elif np_type == np.bool_:
            data = rng.rand(*shape) > 0.5
        else:
            data = rng.randint(0, 2, size=shape).astype(np_type)
        inputs[item["name"]] = data
    return inputs


def cast_inputs(reference_inputs, ort_inputs):
    inputs = {}
    for item in ort_inputs:
        target_type = onnx_type_to_numpy(item.type)
        value = np.asarray(reference_inputs[item.name])
        if np_type_is_float(input_type=item.type):
            inputs[item.name] = value.astype(target_type)
        else:
            inputs[item.name] = value.astype(target_type)
    return inputs


def np_type_is_float(input_type):
    return onnx_type_to_numpy(input_type) in (np.float16, np.float32, np.float64)


def run_once(session, inputs):
    return session.run(None, inputs)


def benchmark(session, inputs, iterations, warmup):
    for _ in range(max(0, warmup)):
        session.run(None, inputs)

    timings = []
    for _ in range(max(1, iterations)):
        start = time.perf_counter()
        session.run(None, inputs)
        end = time.perf_counter()
        timings.append((end - start) * 1000.0)

    return {
        "count": len(timings),
        "mean_ms": float(statistics.mean(timings)),
        "median_ms": float(statistics.median(timings)),
        "p90_ms": float(np.percentile(timings, 90)),
        "p95_ms": float(np.percentile(timings, 95)),
        "min_ms": float(min(timings)),
        "max_ms": float(max(timings)),
        "std_ms": float(statistics.pstdev(timings)),
        "throughput_fps": float(1000.0 / statistics.mean(timings)),
    }


def output_diff(fp32_outputs, candidate_outputs):
    diffs = []
    for left, right in zip(fp32_outputs, candidate_outputs):
        left_np = np.asarray(left).astype(np.float32)
        right_np = np.asarray(right).astype(np.float32)
        diffs.append(float(np.max(np.abs(left_np - right_np))))
    return {
        "max_abs_diff": float(max(diffs)),
        "per_output_max_abs_diff": diffs,
    }


def evaluate_model(path, reference_inputs, iterations, warmup):
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inputs = cast_inputs(reference_inputs, session.get_inputs())
    stats = benchmark(session, inputs, iterations, warmup)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    return {
        "session": session,
        "inputs": inputs,
        "path": path,
        "size_mb": round(size_mb, 4),
        "timings_ms": stats,
        "output_count": len(session.get_outputs()),
        "provider": session.get_providers(),
        "input_contract": [
            {
                "name": item.name,
                "shape": item.shape,
                "type": item.type,
            }
            for item in session.get_inputs()
        ],
        "output_contract": [
            {
                "name": item.name,
                "shape": item.shape,
                "type": item.type,
            }
            for item in session.get_outputs()
        ],
    }


def main():
    args = parse_args()

    for path in [args.fp32, args.fp16, args.int8]:
        ensure_file(path)

    fp32_session = ort.InferenceSession(args.fp32, providers=["CPUExecutionProvider"])
    fp32_input_meta = [
        {
            "name": item.name,
            "shape": item.shape,
            "type": "tensor(float)",
        }
        for item in fp32_session.get_inputs()
    ]
    reference_inputs = build_reference_inputs(fp32_input_meta)
    fp32_session = None

    model_defs = [
        ("fp32", args.fp32),
        ("fp16", args.fp16),
        ("int8", args.int8),
    ]

    results = {}
    baseline_outputs = None

    for tag, path in model_defs:
        evaluated = evaluate_model(path, reference_inputs, args.iterations, args.warmup)
        session = evaluated.pop("session")
        inputs = evaluated.pop("inputs")

        timings = evaluated.pop("timings_ms")
        timings["iterations"] = args.iterations
        timings["warmup"] = args.warmup
        evaluated["timings_ms"] = timings

        if tag == "fp32":
            baseline_outputs = run_once(session, inputs)
        elif args.compare and baseline_outputs is not None:
            candidate = run_once(session, inputs)
            evaluated["diff_to_fp32"] = output_diff(baseline_outputs, candidate)

        evaluated["mean_latency_ms"] = timings["mean_ms"]
        evaluated["throughput_fps"] = timings["throughput_fps"]
        results[tag] = evaluated

    ranking = sorted(
        [(tag, data["timings_ms"]["mean_ms"]) for tag, data in results.items()],
        key=lambda item: item[1],
    )

    report = {
        "iterations": args.iterations,
        "warmup": args.warmup,
        "provider": ["CPUExecutionProvider"],
        "models": results,
        "ranking_by_mean_latency": [
            {
                "model": tag,
                "rank": idx + 1,
                "mean_ms": value,
            }
            for idx, (tag, value) in enumerate(ranking)
        ],
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
