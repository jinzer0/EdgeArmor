import os
from PIL import Image, ImageDraw

from src.detection.FaceDetection import FaceDetector
from .face_selector import select_faces
from .forensics_adapter_infer import ForensicsAdapterInfer
from .inference_contract import (
    DEFAULT_FAKE_THRESHOLD,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_FACE_SIZE,
    DEFAULT_PIPELINE_MARGIN,
    DEFAULT_TOP_K,
    aggregate_face_predictions,
    build_failed_result,
    build_no_face_result,
    compute_face_label,
)


class DeepfakeDetectionPipeline:
    def __init__(
        self,
        config_path,
        weights_path,
        device=None,
        detector=None,
        min_confidence=DEFAULT_MIN_CONFIDENCE,
        min_face_size=DEFAULT_MIN_FACE_SIZE,
        top_k=DEFAULT_TOP_K,
        margin=DEFAULT_PIPELINE_MARGIN,
        fake_threshold=DEFAULT_FAKE_THRESHOLD,
    ):
        self.detector = detector if detector is not None else FaceDetector(margin=margin)
        self.fake_detector = ForensicsAdapterInfer(
            config_path=config_path,
            weights_path=weights_path,
            device=device,
        )

        try:
            import src.detection.FaceDetection as detector_module

            detector_module._DEVICE = self.fake_detector.device
        except Exception:
            pass

        self.min_confidence = min_confidence
        self.min_face_size = min_face_size
        self.top_k = top_k
        self.fake_threshold = fake_threshold

    def _draw_debug(self, image, face_results, output_dir, prefix):
        os.makedirs(output_dir, exist_ok=True)

        debug_image = image.copy()
        drawer = ImageDraw.Draw(debug_image)

        for idx, face in enumerate(face_results):
            x1, y1, x2, y2 = face["bbox"]
            is_fake = face["pred_label"] == "fake"
            color = "red" if is_fake else "green"
            drawer.rectangle([x1, y1, x2, y2], outline=color, width=3)
            drawer.text(
                (x1, max(0, y1 - 14)),
                f"{idx + 1}. conf:{face['det_confidence']:.2f}, prob:{face['fake_prob']:.3f}",
                fill=color,
            )

        image_output = os.path.join(output_dir, f"{prefix}_faces.png")
        debug_image.save(image_output)

        for idx, face in enumerate(face_results):
            crop = face["crop"]
            crop_output = os.path.join(output_dir, f"{prefix}_face_{idx + 1:02d}.png")
            crop.save(crop_output)

    def predict(
        self,
        image,
        save_debug=False,
        debug_dir="outputs/debug",
        debug_prefix="image",
    ):
        image = self.detector._load_image(image)
        detections = self.detector.detect_faces(image)

        selected = select_faces(
            detections=detections,
            min_confidence=self.min_confidence,
            min_face_size=self.min_face_size,
            top_k=self.top_k,
        )

        if len(selected) == 0:
            return build_no_face_result(num_detected_faces=len(detections))

        crops = self.detector.crop_and_align_faces(image, selected)

        face_results = []
        failed = 0
        for idx, detection in enumerate(selected):
            if idx >= len(crops):
                break

            crop = crops[idx]
            try:
                pred = self.fake_detector.predict(crop)
            except Exception:
                failed += 1
                continue

            fake_prob = pred["fake_prob"]
            pred_label = compute_face_label(
                fake_prob=fake_prob,
                pred_label_id=pred["pred_label"],
                fake_threshold=self.fake_threshold,
            )

            face_results.append(
                {
                    "face_index": idx,
                    "bbox": list(map(int, detection["bbox"])),
                    "det_confidence": float(detection["confidence"]),
                    "fake_prob": float(fake_prob),
                    "pred_label": pred_label,
                    "pred_label_id": int(pred["pred_label"]),
                    "logits": pred["logits"],
                    "crop": crop,
                    "crop_size": [crop.width, crop.height],
                }
            )

        if len(face_results) == 0:
            return build_failed_result(
                num_detected_faces=len(detections),
                num_failed_faces=failed,
            )

        faces_summary = []
        for face in face_results:
            info = dict(face)
            info.pop("crop")
            faces_summary.append(info)

        if save_debug:
            self._draw_debug(image, face_results, debug_dir, debug_prefix)

        result = aggregate_face_predictions(
            face_predictions=faces_summary,
            fake_threshold=self.fake_threshold,
            num_detected_faces=len(detections),
            num_failed_faces=failed,
        )
        result["debug_dir"] = debug_dir if save_debug else None
        return result
