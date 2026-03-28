import os
import shutil


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
EXTENSION_DIR = os.path.join(PROJECT_ROOT, "extension")
MODELS_DIR = os.path.join(EXTENSION_DIR, "models")
VENDOR_DIR = os.path.join(EXTENSION_DIR, "vendor")
ONNX_DIR = os.path.join(PROJECT_ROOT, "artifacts", "onnx")
ORT_DIST_DIR = os.path.join(PROJECT_ROOT, "node_modules", "onnxruntime-web", "dist")

MODEL_FILES = [
    {
        "source": "face_detector.onnx",
        "target": "face_detector.onnx",
    },
    {
        "source": "forensics_adapter.webgpu.fp16.onnx",
        "target": "forensics_adapter.onnx",
    },
]

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
        for model_file in MODEL_FILES
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
    for model_file in MODEL_FILES:
        source_path = os.path.join(ONNX_DIR, model_file["source"])
        target_path = os.path.join(MODELS_DIR, model_file["target"])
        link_or_copy(source_path, target_path)


def prepare_vendor():
    ensure_dir(VENDOR_DIR)
    for filename in VENDOR_FILES:
        source_path = os.path.join(ORT_DIST_DIR, filename)
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Missing ORT runtime file: {source_path}")
        target_path = os.path.join(VENDOR_DIR, filename)
        copy_file(source_path, target_path)


def main():
    validate_inputs()
    prepare_models()
    prepare_vendor()
    print("Prepared Chrome extension folder:")
    print(EXTENSION_DIR)


if __name__ == "__main__":
    main()
