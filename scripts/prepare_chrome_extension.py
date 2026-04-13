import json
import os
import shutil
import sys

import onnx


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EXTENSION_DIR = os.path.join(PROJECT_ROOT, "extension")
MODELS_DIR = os.path.join(EXTENSION_DIR, "models")
VENDOR_DIR = os.path.join(EXTENSION_DIR, "vendor")
ONNX_DIR = os.path.join(PROJECT_ROOT, "artifacts", "onnx")
ORT_DIST_DIR = os.path.join(PROJECT_ROOT, "node_modules", "onnxruntime-web", "dist")

from src.pipeline.inference_contract import (
    CLASSIFIER_MODEL_FILENAME,
    DETECTOR_MODEL_FILENAME,
    EXTENSION_CONTRACT_FILENAME,
    build_extension_contract,
)

CLASSIFIER_ENV_VAR = "EDGEARMOR_CLASSIFIER_ONNX"


HF_CLASSIFIER_PREFER_ORDER = [
    "model_q4f16.onnx",
    "model_q4.onnx",
    "model_int8.onnx",
    "model_quantized.onnx",
    "model_uint8.onnx",
    "model_fp16.onnx",
    "model_bnb4.onnx",
]

LEGACY_CLASSIFIER_PREFER_ORDER = [
    "forensics_adapter.onnx",
    "forensics_adapter.webgpu.fp16.onnx",
    "forensics_adapter_fp16.onnx",
]


def is_valid_onnx_file(model_path):
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
    except Exception:
        return False
    return True

def resolve_classifier_source():
    forced = os.environ.get(CLASSIFIER_ENV_VAR)
    if forced:
        forced_path = os.path.join(ONNX_DIR, forced)
        if not os.path.exists(forced_path):
            raise FileNotFoundError(f"Forced classifier ONNX not found: {forced}")
        if not is_valid_onnx_file(forced_path):
            raise ValueError(f"Forced classifier ONNX is invalid: {forced}")
        return forced

    hf_candidates = [
        filename
        for filename in os.listdir(ONNX_DIR)
        if filename.startswith("model_") and filename.endswith(".onnx")
    ]
    if hf_candidates:
        for filename in HF_CLASSIFIER_PREFER_ORDER:
            candidate = os.path.join(ONNX_DIR, filename)
            if os.path.exists(candidate) and is_valid_onnx_file(candidate):
                return filename

        valid_hf_candidates = [
            filename
            for filename in hf_candidates
            if is_valid_onnx_file(os.path.join(ONNX_DIR, filename))
        ]
        if valid_hf_candidates:
            smallest_model = min(
                valid_hf_candidates,
                key=lambda filename: os.path.getsize(os.path.join(ONNX_DIR, filename)),
            )
            return smallest_model

    for filename in LEGACY_CLASSIFIER_PREFER_ORDER:
        candidate = os.path.join(ONNX_DIR, filename)
        if os.path.exists(candidate) and is_valid_onnx_file(candidate):
            return filename

    raise FileNotFoundError(
        "No suitable valid classifier ONNX found in artifacts/onnx. Provide a valid model_*.onnx or keep a valid legacy classifier artifact."
    )

VENDOR_FILES = [
    "ort.all.min.mjs",
    "ort-wasm-simd-threaded.wasm",
    "ort-wasm-simd-threaded.mjs",
    "ort-wasm-simd-threaded.jsep.wasm",
    "ort-wasm-simd-threaded.jsep.mjs",
    "ort-wasm-simd-threaded.asyncify.wasm",
    "ort-wasm-simd-threaded.asyncify.mjs",
    "ort-wasm-simd-threaded.jspi.wasm",
    "ort-wasm-simd-threaded.jspi.mjs",
]

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def remove_existing(path):
    if os.path.lexists(path):
        os.remove(path)


def link_or_copy(source_path, target_path):
    remove_existing(target_path)
    try:
        os.link(source_path, target_path)
    except OSError:
        shutil.copy2(source_path, target_path)


def copy_file(source_path, target_path):
    remove_existing(target_path)
    shutil.copy2(source_path, target_path)


def validate_inputs():
    if not os.path.isdir(EXTENSION_DIR):
        raise FileNotFoundError(f"Extension directory not found: {EXTENSION_DIR}")
    if not os.path.isdir(ONNX_DIR):
        raise FileNotFoundError(f"ONNX artifacts directory not found: {ONNX_DIR}")
    if not os.path.isdir(ORT_DIST_DIR):
        raise FileNotFoundError(
            "onnxruntime-web dist directory not found. Run `npm install` first."
        )
    missing_models = [
        model_file["source"]
        for model_file in get_model_files()
        if not os.path.exists(os.path.join(ONNX_DIR, model_file["source"]))
    ]
    if missing_models:
        raise FileNotFoundError(
            "Missing ONNX models: "
            + ", ".join(missing_models)
            + ". Export the browser-safe classifier first."
        )


def prepare_models():
    ensure_dir(MODELS_DIR)
    remove_existing(os.path.join(MODELS_DIR, "blaze_face_short_range.tflite"))
    remove_existing(os.path.join(MODELS_DIR, "forensics_adapter_fp16.onnx"))
    for model_file in get_model_files():
        source_path = os.path.join(ONNX_DIR, model_file["source"])
        target_path = os.path.join(MODELS_DIR, model_file["target"])
        link_or_copy(source_path, target_path)


def prepare_contract():
    ensure_dir(MODELS_DIR)
    target_path = os.path.join(MODELS_DIR, EXTENSION_CONTRACT_FILENAME)
    remove_existing(target_path)
    with open(target_path, "w", encoding="utf-8") as file:
        json.dump(build_extension_contract(), file, ensure_ascii=False, indent=2)
        file.write("\n")


def get_model_files():
    classifier_source = resolve_classifier_source()

    return [
        {
            "source": DETECTOR_MODEL_FILENAME,
            "target": DETECTOR_MODEL_FILENAME,
        },
        {
            "source": classifier_source,
            "target": CLASSIFIER_MODEL_FILENAME,
        },
    ]


def prepare_ort_vendor():
    ensure_dir(VENDOR_DIR)
    for filename in VENDOR_FILES:
        source_path = os.path.join(ORT_DIST_DIR, filename)
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Missing ORT runtime file: {source_path}")
        target_path = os.path.join(VENDOR_DIR, filename)
        copy_file(source_path, target_path)


def prepare_vendor():
    prepare_ort_vendor()
    shutil.rmtree(os.path.join(VENDOR_DIR, "mediapipe"), ignore_errors=True)


def main():
    validate_inputs()
    prepare_models()
    prepare_contract()
    prepare_vendor()
    print("Prepared Chrome extension folder:")
    print(EXTENSION_DIR)


if __name__ == "__main__":
    main()
