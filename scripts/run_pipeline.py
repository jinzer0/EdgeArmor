import argparse
import json
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.deepfake_pipeline import DeepfakeDetectionPipeline
from src.pipeline.inference_contract import (
    DEFAULT_FAKE_THRESHOLD,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_FACE_SIZE,
    DEFAULT_PIPELINE_MARGIN,
    DEFAULT_TOP_K,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage deepfake image detection pipeline")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--weights_path", required=True, help="Path to DS checkpoint")
    parser.add_argument("--config_path", default=None, help="Path to config yaml (optional)")
    parser.add_argument("--device", default=None, help="cuda:0, cuda, mps, or cpu")
    parser.add_argument("--min_confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--min_face_size", type=int, default=DEFAULT_MIN_FACE_SIZE)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--margin", type=float, default=DEFAULT_PIPELINE_MARGIN)
    parser.add_argument("--fake_threshold", type=float, default=DEFAULT_FAKE_THRESHOLD)
    parser.add_argument("--save_debug", action="store_true")
    parser.add_argument("--debug_dir", default="outputs/debug")
    parser.add_argument("--debug_prefix", default="sample")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.image):
        print(f"Image not found: {args.image}")
        sys.exit(1)

    pipeline = DeepfakeDetectionPipeline(
        config_path=args.config_path,
        weights_path=args.weights_path,
        device=args.device,
        min_confidence=args.min_confidence,
        min_face_size=args.min_face_size,
        top_k=args.top_k,
        margin=args.margin,
        fake_threshold=args.fake_threshold,
    )

    result = pipeline.predict(
        image=args.image,
        save_debug=args.save_debug,
        debug_dir=args.debug_dir,
        debug_prefix=args.debug_prefix,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
