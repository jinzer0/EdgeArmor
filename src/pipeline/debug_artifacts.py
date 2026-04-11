import os

from PIL import ImageDraw


def save_debug_artifacts(image, face_results, output_dir, prefix):
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
