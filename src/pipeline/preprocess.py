import torch
from PIL import Image
from torchvision import transforms as T

from .inference_contract import (
    CLASSIFIER_INPUT_RESOLUTION,
    CLASSIFIER_MEAN,
    CLASSIFIER_STD,
)


def _to_rgb_image(face_image):
    if isinstance(face_image, Image.Image):
        return face_image.convert("RGB")

    raise ValueError("face_image must be a PIL.Image.Image")


def create_if_boundary(batch_size=1, resolution=CLASSIFIER_INPUT_RESOLUTION, value=1.0, device="cpu"):
    patch_num = (resolution // 16) ** 2
    return torch.full((batch_size, patch_num), float(value), dtype=torch.float32, device=device)


def build_face_image_tensor(face_image, resolution=CLASSIFIER_INPUT_RESOLUTION, mean=None, std=None, device="cpu"):
    if mean is None:
        mean = list(CLASSIFIER_MEAN)
    if std is None:
        std = list(CLASSIFIER_STD)

    image = _to_rgb_image(face_image).resize((resolution, resolution))

    preprocess = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    return preprocess(image).unsqueeze(0).to(device)


def build_face_data_dict(face_image, resolution=CLASSIFIER_INPUT_RESOLUTION, mean=None, std=None, device="cpu", label=0, if_boundary_value=1.0):
    image_tensor = build_face_image_tensor(
        face_image,
        resolution=resolution,
        mean=mean,
        std=std,
        device=device,
    )
    if_boundary = create_if_boundary(
        batch_size=1,
        resolution=resolution,
        value=if_boundary_value,
        device=device,
    )

    label = torch.LongTensor([label]).to(device)

    return {
        "image": image_tensor,
        "label": label,
        "landmark": None,
        "mask": None,
        "xray": None,
        "patch_label": None,
        "clip_patch_label": None,
        "if_boundary": if_boundary,
    }
