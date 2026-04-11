import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch

from src.detection.FaceDetection import FaceDetector
from src.pipeline.deepfake_pipeline import DeepfakeDetectionPipeline
from src.pipeline.face_selector import select_faces
from src.pipeline.forensics_adapter_infer import ForensicsAdapterInfer
from src.pipeline.inference_contract import (
    DEFAULT_FAKE_THRESHOLD,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_FACE_SIZE,
    DEFAULT_PIPELINE_MARGIN,
    DEFAULT_TOP_K,
    aggregate_face_predictions,
    compute_face_label,
)
from src.pipeline.preprocess import build_face_image_tensor, create_if_boundary


def parse_args():
    parser = argparse.ArgumentParser(description="Verify ONNX exports against the PyTorch pipeline.")
    parser.add_argument("--image", required=True, help="Input image used for detector/classifier/end-to-end checks")
    parser.add_argument("--weights_path", required=True, help="Path to ckpt_best.pth")
    parser.add_argument("--config_path", default="ForensicsAdapter/config/test.yaml", help="Path to YAML config")
    parser.add_argument("--detector_onnx", default="artifacts/onnx/face_detector.onnx", help="Detector ONNX path")
    parser.add_argument("--classifier_onnx", default="artifacts/onnx/forensics_adapter_fp16.onnx", help="Classifier ONNX path")
    parser.add_argument("--seed", type=int, default=0, help="Seed used for deterministic model initialization")
    parser.add_argument("--classifier_atol", type=float, default=1e-3, help="Classifier parity absolute tolerance")
    parser.add_argument("--min_confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--min_face_size", type=int, default=DEFAULT_MIN_FACE_SIZE)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--margin", type=float, default=DEFAULT_PIPELINE_MARGIN)
    parser.add_argument("--fake_threshold", type=float, default=DEFAULT_FAKE_THRESHOLD)
    return parser.parse_args()


def load_image(image_path):
    return Image.open(image_path).convert("RGB")


def letterbox_image(image, target_size=640, fill=(114, 114, 114)):
    width, height = image.size
    scale = min(target_size / width, target_size / height)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_size, target_size), fill)
    pad_x = (target_size - resized_width) // 2
    pad_y = (target_size - resized_height) // 2
    canvas.paste(resized, (pad_x, pad_y))
    metadata = {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "orig_width": width,
        "orig_height": height,
        "input_size": target_size,
    }
    return canvas, metadata


def image_to_onnx_input(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0)


def parse_detector_output(output_array):
    array = np.asarray(output_array)

    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.squeeze(array, axis=-1)

    if array.ndim != 2:
        raise ValueError(f"Unsupported detector output shape: {array.shape}")

    if array.shape[0] >= 6 and array.shape[1] < 6:
        array = array.T

    if array.shape[1] < 6:
        raise ValueError(f"Detector output must expose at least 6 columns, got {array.shape}")

    detections = []
    for row in array:
        if not np.all(np.isfinite(row[:6])):
            continue
        confidence = float(row[4])
        if confidence <= 0:
            continue
        detections.append(
            {
                "bbox": [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
                "confidence": confidence,
                "class_id": int(row[5]),
            }
        )

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    return detections


def scale_detections_back(detections, metadata):
    scaled = []
    scale = metadata["scale"]
    pad_x = metadata["pad_x"]
    pad_y = metadata["pad_y"]
    width = metadata["orig_width"]
    height = metadata["orig_height"]

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        x1 = max(0.0, min(width, (x1 - pad_x) / scale))
        y1 = max(0.0, min(height, (y1 - pad_y) / scale))
        x2 = max(0.0, min(width, (x2 - pad_x) / scale))
        y2 = max(0.0, min(height, (y2 - pad_y) / scale))
        scaled.append(
            {
                "bbox": (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                "confidence": float(detection["confidence"]),
                "landmarks": None,
            }
        )

    return scaled


def load_sessions(detector_path, classifier_path):
    detector_model = onnx.load(detector_path)
    onnx.checker.check_model(detector_model)
    classifier_model = onnx.load(classifier_path)
    onnx.checker.check_model(classifier_model)

    detector_session = ort.InferenceSession(detector_path, providers=["CPUExecutionProvider"])
    classifier_session = ort.InferenceSession(classifier_path, providers=["CPUExecutionProvider"])
    return detector_session, classifier_session


def compare_detector_outputs(pytorch_detections, onnx_detections):
    top_pt = pytorch_detections[:3]
    top_onnx = onnx_detections[:3]
    pairs = []
    count = min(len(top_pt), len(top_onnx))
    for idx in range(count):
        pt = top_pt[idx]
        ox = top_onnx[idx]
        bbox_delta = [abs(float(a) - float(b)) for a, b in zip(pt["bbox"], ox["bbox"])]
        pairs.append(
            {
                "index": idx,
                "pytorch_confidence": float(pt["confidence"]),
                "onnx_confidence": float(ox["confidence"]),
                "confidence_delta": abs(float(pt["confidence"]) - float(ox["confidence"])),
                "bbox_delta": bbox_delta,
            }
        )

    return {
        "pytorch_count": len(pytorch_detections),
        "onnx_count": len(onnx_detections),
        "top_pairs": pairs,
    }


def build_classifier_onnx_inputs(face_image, config, input_type="tensor(float)"):
    resolution = int(config["resolution"])
    image_tensor = build_face_image_tensor(
        face_image,
        resolution=resolution,
        mean=config["mean"],
        std=config["std"],
        device="cpu",
    )
    if_boundary = create_if_boundary(
        batch_size=1,
        resolution=resolution,
        value=1.0,
        device="cpu",
    )
    image_array = image_tensor.numpy()
    if_boundary_array = if_boundary.numpy()

    if input_type == "tensor(float16)":
        image_array = image_array.astype(np.float16)
        if_boundary_array = if_boundary_array.astype(np.float16)

    return {
        "image": image_array,
        "if_boundary": if_boundary_array,
    }


def run_classifier_onnx(session, inputs):
    feed = {}
    for value in session.get_inputs():
        feed[value.name] = inputs[value.name]
    outputs = session.run(None, feed)
    names = [output.name for output in session.get_outputs()]
    return {name: value for name, value in zip(names, outputs)}


def max_abs_diff(left, right):
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def build_face_crop(image, detector, min_confidence, min_face_size, top_k):
    detections = detector.detect_faces(image)
    selected = select_faces(
        detections=detections,
        min_confidence=min_confidence,
        min_face_size=min_face_size,
        top_k=top_k,
    )
    crops = detector.crop_and_align_faces(image, selected)
    return detections, selected, crops


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    detector_session, classifier_session = load_sessions(args.detector_onnx, args.classifier_onnx)
    image = load_image(args.image)

    face_detector = FaceDetector(margin=args.margin)
    pytorch_detector_detections, pytorch_selected, pytorch_crops = build_face_crop(
        image=image,
        detector=face_detector,
        min_confidence=args.min_confidence,
        min_face_size=args.min_face_size,
        top_k=args.top_k,
    )

    letterboxed, metadata = letterbox_image(image)
    detector_input = image_to_onnx_input(letterboxed)
    detector_feed = {detector_session.get_inputs()[0].name: detector_input}
    detector_outputs = detector_session.run(None, detector_feed)
    detector_output_names = [output.name for output in detector_session.get_outputs()]
    detector_output_map = {name: value for name, value in zip(detector_output_names, detector_outputs)}
    raw_onnx_detections = parse_detector_output(detector_outputs[0])
    onnx_detections = scale_detections_back(raw_onnx_detections, metadata)
    onnx_selected = select_faces(
        detections=onnx_detections,
        min_confidence=args.min_confidence,
        min_face_size=args.min_face_size,
        top_k=args.top_k,
    )
    onnx_crops = face_detector.crop_and_align_faces(image, onnx_selected)

    infer = ForensicsAdapterInfer(
        config_path=args.config_path,
        weights_path=args.weights_path,
        device="cpu",
    )

    if not pytorch_crops:
        raise RuntimeError("No face crop found in the provided image for classifier verification.")

    classifier_input_type = classifier_session.get_inputs()[0].type
    classifier_atol = args.classifier_atol
    if classifier_input_type == "tensor(float16)":
        classifier_atol = max(classifier_atol, 1e-2)
    classifier_inputs = build_classifier_onnx_inputs(
        pytorch_crops[0],
        infer.config,
        input_type=classifier_input_type,
    )
    classifier_outputs = run_classifier_onnx(classifier_session, classifier_inputs)
    pytorch_pred = infer.predict(pytorch_crops[0])

    logits_onnx = np.asarray(classifier_outputs["logits"])
    fake_prob_onnx = np.asarray(classifier_outputs["fake_prob"])
    xray_onnx = np.asarray(classifier_outputs["xray_pred"])

    classifier_report = {
        "logits_max_abs_diff": max_abs_diff(pytorch_pred["logits"], logits_onnx[0]),
        "fake_prob_abs_diff": abs(float(pytorch_pred["fake_prob"]) - float(fake_prob_onnx.reshape(-1)[0])),
        "xray_shape": list(xray_onnx.shape),
        "inputs": {
            "image_shape": list(classifier_inputs["image"].shape),
            "if_boundary_shape": list(classifier_inputs["if_boundary"].shape),
        },
        "atol": classifier_atol,
    }
    classifier_report["within_tolerance"] = (
        classifier_report["logits_max_abs_diff"] <= classifier_atol
        and classifier_report["fake_prob_abs_diff"] <= classifier_atol
    )

    onnx_face_predictions = []
    for idx, crop in enumerate(onnx_crops):
        crop_inputs = build_classifier_onnx_inputs(
            crop,
            infer.config,
            input_type=classifier_input_type,
        )
        crop_outputs = run_classifier_onnx(classifier_session, crop_inputs)
        logits = np.asarray(crop_outputs["logits"]).reshape(1, -1)[0]
        fake_prob = float(np.asarray(crop_outputs["fake_prob"]).reshape(-1)[0])
        pred_label_id = int(np.argmax(logits))
        pred_label = compute_face_label(
            fake_prob=fake_prob,
            pred_label_id=pred_label_id,
            fake_threshold=args.fake_threshold,
        )
        detection = onnx_selected[idx]
        onnx_face_predictions.append(
            {
                "face_index": idx,
                "bbox": list(map(int, detection["bbox"])),
                "det_confidence": float(detection["confidence"]),
                "fake_prob": fake_prob,
                "pred_label": pred_label,
                "pred_label_id": pred_label_id,
                "logits": [float(value) for value in logits.tolist()],
                "crop_size": [crop.width, crop.height],
            }
        )

    onnx_pipeline_result = aggregate_face_predictions(
        onnx_face_predictions,
        fake_threshold=args.fake_threshold,
        num_detected_faces=len(onnx_detections),
        num_failed_faces=0,
    )

    pytorch_pipeline = DeepfakeDetectionPipeline(
        config_path=args.config_path,
        weights_path=args.weights_path,
        device="cpu",
        min_confidence=args.min_confidence,
        min_face_size=args.min_face_size,
        top_k=args.top_k,
        margin=args.margin,
        fake_threshold=args.fake_threshold,
    )
    pytorch_pipeline_result = pytorch_pipeline.predict(image)

    report = {
        "detector_contract": {
            "inputs": [
                {
                    "name": item.name,
                    "shape": item.shape,
                }
                for item in detector_session.get_inputs()
            ],
            "outputs": [
                {
                    "name": item.name,
                    "shape": item.shape,
                }
                for item in detector_session.get_outputs()
            ],
        },
        "classifier_contract": {
            "inputs": [
                {
                    "name": item.name,
                    "shape": item.shape,
                }
                for item in classifier_session.get_inputs()
            ],
            "outputs": [
                {
                    "name": item.name,
                    "shape": item.shape,
                }
                for item in classifier_session.get_outputs()
            ],
        },
        "detector_output_names": detector_output_names,
        "detector_parity": compare_detector_outputs(pytorch_detector_detections, onnx_detections),
        "classifier_parity": classifier_report,
        "end_to_end": {
            "pytorch": pytorch_pipeline_result,
            "onnx": onnx_pipeline_result,
            "image_fake_prob_abs_diff": None
            if pytorch_pipeline_result["image_fake_prob"] is None or onnx_pipeline_result["image_fake_prob"] is None
            else abs(float(pytorch_pipeline_result["image_fake_prob"]) - float(onnx_pipeline_result["image_fake_prob"])),
            "same_label": pytorch_pipeline_result["image_pred_label"] == onnx_pipeline_result["image_pred_label"],
        },
        "raw_detector_output_shapes": {
            name: list(np.asarray(value).shape)
            for name, value in detector_output_map.items()
        },
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not classifier_report["within_tolerance"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
