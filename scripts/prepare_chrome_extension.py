import os
import shutil
import urllib.request


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
EXTENSION_DIR = os.path.join(PROJECT_ROOT, "extension")
MODELS_DIR = os.path.join(EXTENSION_DIR, "models")
VENDOR_DIR = os.path.join(EXTENSION_DIR, "vendor")
ONNX_DIR = os.path.join(PROJECT_ROOT, "artifacts", "onnx")
MEDIAPIPE_CACHE_DIR = os.path.join(PROJECT_ROOT, "artifacts", "mediapipe")
ORT_DIST_DIR = os.path.join(PROJECT_ROOT, "node_modules", "onnxruntime-web", "dist")
MEDIAPIPE_DIST_DIR = os.path.join(PROJECT_ROOT, "node_modules", "@mediapipe", "tasks-vision")
MEDIAPIPE_VENDOR_DIR = os.path.join(VENDOR_DIR, "mediapipe")
MEDIAPIPE_WASM_DIR = os.path.join(MEDIAPIPE_VENDOR_DIR, "wasm")
CLASSIFIER_ENV_VAR = "EDGEARMOR_CLASSIFIER_ONNX"
MEDIAPIPE_MODEL_FILENAME = "blaze_face_short_range.tflite"
MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


HF_CLASSIFIER_PREFER_ORDER = [
    "model_q4f16.onnx",
    "model_q4.onnx",
    "model_int8.onnx",
    "model_quantized.onnx",
    "model_uint8.onnx",
    "model_fp16.onnx",
    "model_bnb4.onnx",
]

def resolve_classifier_source():
    forced = os.environ.get(CLASSIFIER_ENV_VAR)
    if forced:
        forced_path = os.path.join(ONNX_DIR, forced)
        if not os.path.exists(forced_path):
            raise FileNotFoundError(f"Forced classifier ONNX not found: {forced}")
        return forced

    hf_candidates = [
        filename
        for filename in os.listdir(ONNX_DIR)
        if filename.startswith("model_") and filename.endswith(".onnx")
    ]
    if hf_candidates:
        for filename in HF_CLASSIFIER_PREFER_ORDER:
            candidate = os.path.join(ONNX_DIR, filename)
            if os.path.exists(candidate):
                return filename

        smallest_model = min(
            hf_candidates,
            key=lambda filename: os.path.getsize(os.path.join(ONNX_DIR, filename)),
        )
        return smallest_model

    legacy_source = "forensics_adapter.webgpu.fp16.onnx"
    legacy_path = os.path.join(ONNX_DIR, legacy_source)
    if os.path.exists(legacy_path):
        return legacy_source

    for filename in ["forensics_adapter.webgpu.fp16.onnx", "forensics_adapter_fp16.onnx"]:
        candidate = os.path.join(ONNX_DIR, filename)
        if os.path.exists(candidate):
            return filename

    raise FileNotFoundError("No suitable classifier ONNX found in artifacts/onnx. Download model_*.onnx or keep forensics_adapter.webgpu.fp16.onnx.")

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

MEDIAPIPE_VENDOR_FILES = [
    ("vision_bundle.mjs", os.path.join(MEDIAPIPE_VENDOR_DIR, "vision_bundle.mjs")),
    ("wasm/vision_wasm_internal.js", os.path.join(MEDIAPIPE_WASM_DIR, "vision_wasm_internal.js")),
    ("wasm/vision_wasm_internal.wasm", os.path.join(MEDIAPIPE_WASM_DIR, "vision_wasm_internal.wasm")),
    ("wasm/vision_wasm_module_internal.js", os.path.join(MEDIAPIPE_WASM_DIR, "vision_wasm_module_internal.js")),
    ("wasm/vision_wasm_module_internal.wasm", os.path.join(MEDIAPIPE_WASM_DIR, "vision_wasm_module_internal.wasm")),
    ("wasm/vision_wasm_nosimd_internal.js", os.path.join(MEDIAPIPE_WASM_DIR, "vision_wasm_nosimd_internal.js")),
    ("wasm/vision_wasm_nosimd_internal.wasm", os.path.join(MEDIAPIPE_WASM_DIR, "vision_wasm_nosimd_internal.wasm")),
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


def download_file(url, target_path):
    ensure_dir(os.path.dirname(target_path))
    temp_path = f"{target_path}.tmp"
    remove_existing(temp_path)
    urllib.request.urlretrieve(url, temp_path)
    remove_existing(target_path)
    os.replace(temp_path, target_path)


def validate_inputs():
    if not os.path.isdir(EXTENSION_DIR):
        raise FileNotFoundError(f"Extension directory not found: {EXTENSION_DIR}")
    if not os.path.isdir(ONNX_DIR):
        raise FileNotFoundError(f"ONNX artifacts directory not found: {ONNX_DIR}")
    if not os.path.isdir(ORT_DIST_DIR):
        raise FileNotFoundError(
            "onnxruntime-web dist directory not found. Run `npm install` first."
        )
    if not os.path.isdir(MEDIAPIPE_DIST_DIR):
        raise FileNotFoundError(
            "@mediapipe/tasks-vision package not found. Run `npm install` first."
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
    remove_existing(os.path.join(MODELS_DIR, "face_detector.onnx"))
    for model_file in get_model_files():
        source_path = os.path.join(ONNX_DIR, model_file["source"])
        target_path = os.path.join(MODELS_DIR, model_file["target"])
        link_or_copy(source_path, target_path)

    prepare_mediapipe_model()


def get_model_files():
    classifier_source = resolve_classifier_source()
    classifier_target = "forensics_adapter_fp16.onnx"
    if classifier_source.startswith("model_"):
        classifier_target = "model.onnx"

    return [
        {
            "source": classifier_source,
            "target": classifier_target,
        },
    ]


def prepare_mediapipe_model():
    cache_path = os.path.join(MEDIAPIPE_CACHE_DIR, MEDIAPIPE_MODEL_FILENAME)
    target_path = os.path.join(MODELS_DIR, MEDIAPIPE_MODEL_FILENAME)
    if not os.path.exists(cache_path):
        print(f"Downloading MediaPipe face detector model: {MEDIAPIPE_MODEL_URL}")
        download_file(MEDIAPIPE_MODEL_URL, cache_path)
    link_or_copy(cache_path, target_path)


def prepare_ort_vendor():
    ensure_dir(VENDOR_DIR)
    for filename in VENDOR_FILES:
        source_path = os.path.join(ORT_DIST_DIR, filename)
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Missing ORT runtime file: {source_path}")
        target_path = os.path.join(VENDOR_DIR, filename)
        copy_file(source_path, target_path)


def prepare_mediapipe_vendor():
    ensure_dir(MEDIAPIPE_VENDOR_DIR)
    ensure_dir(MEDIAPIPE_WASM_DIR)
    for source_name, target_path in MEDIAPIPE_VENDOR_FILES:
        source_path = os.path.join(MEDIAPIPE_DIST_DIR, source_name)
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Missing MediaPipe runtime file: {source_path}")
        copy_file(source_path, target_path)


def prepare_vendor():
    prepare_ort_vendor()
    prepare_mediapipe_vendor()


def main():
    validate_inputs()
    prepare_models()
    prepare_vendor()
    print("Prepared Chrome extension folder:")
    print(EXTENSION_DIR)


if __name__ == "__main__":
    main()
