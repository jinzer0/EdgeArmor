import os
import yaml
import torch
from ForensicsAdapter.model.ds import DS

from .preprocess import build_face_data_dict


DEFAULT_INFERENCE_CONFIG = {
    "clip_model_name": "ViT-L/14",
    "vit_name": "vit_tiny_patch16_224",
    "num_quires": 128,
    "fusion_map": {0: 0, 1: 1, 2: 8, 3: 15},
    "mlp_dim": 256,
    "mlp_out_dim": 128,
    "head_num": 16,
    "resolution": 256,
    "mean": [0.48145466, 0.4578275, 0.40821073],
    "std": [0.26862954, 0.26130258, 0.27577711],
    "device": None,
}


def resolve_device(requested_device=None, config_device=None):
    candidates = [requested_device, config_device]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            device = torch.device(candidate)
        except Exception:
            continue

        if device.type == "cuda" and torch.cuda.is_available():
            return device
        if device.type == "mps" and torch.backends.mps.is_available():
            return device
        if device.type == "cpu":
            return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _extract_state_dict(ckpt, key_candidates=None):
    if not isinstance(ckpt, dict):
        return ckpt

    if key_candidates is None:
        key_candidates = ("state_dict", "model_state_dict", "state")

    for key in key_candidates:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]

    if "module" in ckpt and isinstance(ckpt["module"], dict):
        return ckpt["module"]

    return ckpt


class ForensicsAdapterInfer:
    def __init__(
        self,
        config_path,
        weights_path,
        device=None,
    ):
        self.weights_path = weights_path

        config = DEFAULT_INFERENCE_CONFIG.copy()
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f) or {}
            config.update(file_config)
        self.config = config

        self.device = resolve_device(requested_device=device, config_device=self.config.get("device"))
        self.model = DS(
            clip_name=self.config["clip_model_name"],
            adapter_vit_name=self.config["vit_name"],
            num_quires=self.config["num_quires"],
            fusion_map=self.config["fusion_map"],
            mlp_dim=self.config["mlp_dim"],
            mlp_out_dim=self.config["mlp_out_dim"],
            head_num=self.config["head_num"],
            device=self.device,
        )
        self._load_weights()

        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self):
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")

        ckpt = torch.load(self.weights_path, map_location=self.device)
        state_dict = _extract_state_dict(ckpt)
        self.model.load_state_dict(state_dict, strict=False)

    def predict(self, face_image, label=0):
        data_dict = build_face_data_dict(
            face_image,
            resolution=int(self.config["resolution"]),
            mean=self.config["mean"],
            std=self.config["std"],
            device=self.device,
            label=label,
        )

        with torch.no_grad():
            pred_dict = self.model(data_dict, inference=True)

        prob = pred_dict["prob"].detach().squeeze(0).float().cpu().item()
        cls = pred_dict["cls"].detach().squeeze(0).float().cpu()
        pred_label = int(torch.argmax(cls).item())

        return {
            "fake_prob": float(prob),
            "pred_label": pred_label,
            "logits": [float(cls[0].item()), float(cls[1].item())],
        }
