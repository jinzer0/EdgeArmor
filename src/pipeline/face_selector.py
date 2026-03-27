def select_faces(detections, min_confidence=0.5, min_face_size=64, top_k=3):
    filtered = []

    for detection in detections:
        confidence = float(detection.get("confidence", 0.0))
        if confidence < min_confidence:
            continue

        x1, y1, x2, y2 = detection["bbox"]
        width = x2 - x1
        height = y2 - y1
        if width < min_face_size or height < min_face_size:
            continue

        entry = dict(detection)
        entry["area"] = width * height
        filtered.append(entry)

    filtered.sort(key=lambda item: item.get("confidence", 0.0), reverse=True)

    if top_k is None:
        return filtered

    if top_k <= 0:
        return []

    return filtered[:top_k]
