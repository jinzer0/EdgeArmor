import math

from huggingface_hub import hf_hub_download
from PIL import Image
import torch
from ultralytics import YOLO


MODEL_REPO_ID = "arnabdhar/YOLOv8-Face-Detection"
MODEL_FILENAME = "model.pt"

_DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
_MODEL = None


def _get_model():
    global _MODEL

    if _MODEL is None:
        model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
        _MODEL = YOLO(model_path).to(_DEVICE)

    return _MODEL


def _clamp_bbox(x1, y1, x2, y2, width, height):
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


def _expand_bbox(bbox, margin, image_size):
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

    return _clamp_bbox(
        expanded_x1,
        expanded_y1,
        expanded_x2,
        expanded_y2,
        image_width,
        image_height,
    )


def _align_face(face_image, detection, crop_origin):
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


def detect_faces(image):
    image = image.convert("RGB")
    model = _get_model()
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


def crop_and_align_faces(image, detections, margin=0.25):
    image = image.convert("RGB")
    faces = []

    for detection in detections:
        expanded_bbox = _expand_bbox(detection["bbox"], margin, image.size)
        x1, y1, x2, y2 = expanded_bbox
        cropped_face = image.crop((x1, y1, x2, y2))
        aligned_face = _align_face(cropped_face, detection, (x1, y1))
        faces.append(aligned_face)

    return faces


def process_image(image, margin=0.25):
    detections = detect_faces(image)
    return crop_and_align_faces(image, detections, margin=margin)


if __name__ == "__main__":
    image = Image.open("src/detection/1.jpg")
    detections = detect_faces(image)
    faces = crop_and_align_faces(image, detections, margin=0.25)

    print("detections:", len(detections))
    print("faces:", len(faces))
