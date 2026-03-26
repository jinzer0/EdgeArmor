import math
import os

from huggingface_hub import hf_hub_download
from PIL import Image
import torch
from ultralytics import YOLO


MODEL_REPO_ID = "arnabdhar/YOLOv8-Face-Detection"
MODEL_FILENAME = "model.pt"

_DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


class FaceDetector:
    _model = None

    def __init__(self, margin=0.25):
        self.margin = margin

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
            cls._model = YOLO(model_path).to(_DEVICE)

        return cls._model

    def _load_image(self, image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, str) or isinstance(image, os.PathLike):
            return Image.open(image).convert("RGB")

        raise ValueError("image must be a PIL.Image.Image or image path")

    def _clamp_bbox(self, x1, y1, x2, y2, width, height):
        x1 = max(0, min(int(x1), width))
        y1 = max(0, min(int(y1), height))
        x2 = max(0, min(int(x2), width))
        y2 = max(0, min(int(y2), height))

        if x2 <= x1:
            x2 = min(width, x1 + 1)
            x1 = max(0, x2 - 1)

        if y2 <= y1:
            y2 = min(height, y1 + 1)
            y1 = max(0, y2 - 1)

        return x1, y1, x2, y2

    def _expand_bbox(self, bbox, margin, image_size):
        image_width, image_height = image_size
        x1, y1, x2, y2 = bbox

        bbox_width = x2 - x1
        bbox_height = y2 - y1

        margin_x = bbox_width * margin
        margin_y = bbox_height * margin

        expanded_x1 = math.floor(x1 - margin_x)
        expanded_y1 = math.floor(y1 - margin_y)
        expanded_x2 = math.ceil(x2 + margin_x)
        expanded_y2 = math.ceil(y2 + margin_y)

        return self._clamp_bbox(
            expanded_x1,
            expanded_y1,
            expanded_x2,
            expanded_y2,
            image_width,
            image_height,
        )

    def _align_face(self, face_image, detection, crop_origin):
        landmarks = detection.get("landmarks")
        if not landmarks or len(landmarks) < 2:
            return face_image

        left_eye = landmarks[0]
        right_eye = landmarks[1]
        crop_x1, crop_y1 = crop_origin

        left_eye_x = left_eye[0] - crop_x1
        left_eye_y = left_eye[1] - crop_y1
        right_eye_x = right_eye[0] - crop_x1
        right_eye_y = right_eye[1] - crop_y1

        if left_eye_x == right_eye_x and left_eye_y == right_eye_y:
            return face_image

        angle = math.degrees(math.atan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x))
        return face_image.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True)

    def detect_faces(self, image):
        image = self._load_image(image)
        model = self._get_model()
        result = model(image, verbose=False)[0]

        detections = []
        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.detach().cpu().tolist()
        confidences = result.boxes.conf.detach().cpu().tolist()

        for bbox, confidence in zip(boxes, confidences):
            x1, y1, x2, y2 = bbox
            detections.append(
                {
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": float(confidence),
                    "landmarks": None,
                }
            )

        return detections

    def crop_and_align_faces(self, image, detections, margin=None):
        image = self._load_image(image)
        faces = []
        margin = self.margin if margin is None else margin

        for detection in detections:
            expanded_bbox = self._expand_bbox(detection["bbox"], margin, image.size)
            x1, y1, x2, y2 = expanded_bbox
            cropped_face = image.crop((x1, y1, x2, y2))
            aligned_face = self._align_face(cropped_face, detection, (x1, y1))
            faces.append(aligned_face)

        return faces

    def process_image(self, image, margin=None):
        detections = self.detect_faces(image)
        return self.crop_and_align_faces(image, detections, margin=margin)

    def __call__(self, image, margin=None):
        return self.process_image(image, margin=margin)


def detect_faces(image):
    detector = FaceDetector()
    return detector.detect_faces(image)


def crop_and_align_faces(image, detections, margin=0.25):
    detector = FaceDetector(margin=margin)
    return detector.crop_and_align_faces(image, detections, margin=margin)


def process_image(image, margin=0.25):
    detector = FaceDetector(margin=margin)
    return detector.process_image(image, margin=margin)


if __name__ == "__main__":
    detector = FaceDetector(margin=0.25)
    faces = detector.process_image("src/detection/1.jpg")

    print("faces:", len(faces))
