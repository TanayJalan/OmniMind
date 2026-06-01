# ─────────────────────────────────────────────────────────────────
#  detection.py — Detect objects in images using YOLOv8
#
#  HOW YOLO WORKS:
#    YOLO (You Only Look Once) is a CNN that divides an image into
#    a grid. Each cell predicts bounding boxes + class labels in
#    one single forward pass — that's why it's so fast.
#
#  MODEL CHOICE — yolov8n (nano):
#    Ultralytics offers 5 sizes: n, s, m, l, x
#    We use 'n' (nano) because it:
#      - Runs on Mac CPU in ~100ms per image
#      - Downloads only 6MB
#      - Detects 80 common object classes
#    For better accuracy, switch to "yolov8s.pt" or "yolov8m.pt"
#    (they're slower but more accurate — great to experiment with)
#
#  First run: downloads yolov8n.pt automatically (~6MB)
# ─────────────────────────────────────────────────────────────────

from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io

# ── Lazy-load the model ──────────────────────────────────────────
# We only load YOLO once (the first time detect_objects is called)
# and reuse it for every subsequent call. Loading a model takes
# ~1 second — we don't want to pay that cost on every image.
_model = None

def get_model():
    global _model
    if _model is None:
        print("[detection] Loading YOLOv8n model...")
        _model = YOLO("yolov8n.pt")   # downloads automatically on first run
        print("[detection] Model ready")
    return _model


# Colour palette for bounding boxes (one per class index)
COLOURS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
]


def detect_objects(image: Image.Image) -> tuple:
    """
    Detect objects in a PIL Image using YOLOv8.

    Returns:
      detections  → list of dicts: [{label, confidence, bbox}]
      annotated   → PIL Image with bounding boxes drawn on it
      summary     → human-readable string e.g. "2× person, 1× car"
    """
    model = get_model()

    # Run inference — YOLO handles resizing internally
    results = model(image, verbose=False)

    detections = []
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    # Count occurrences of each label for the summary
    label_counts = {}

    for r in results:
        for box in r.boxes:
            label = r.names[int(box.cls)]
            confidence = round(float(box.conf), 2)
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
            })

            label_counts[label] = label_counts.get(label, 0) + 1

            # Pick a colour based on class index
            colour = COLOURS[int(box.cls) % len(COLOURS)]

            # Draw bounding box rectangle
            draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)

            # Draw label background + text
            label_text = f"{label} {confidence:.0%}"
            text_bbox = draw.textbbox((x1, y1 - 20), label_text)
            draw.rectangle(text_bbox, fill=colour)
            draw.text((x1, y1 - 20), label_text, fill="white")

    # Build summary string: "2× person, 1× car, 3× chair"
    summary = ", ".join(f"{count}× {label}" for label, count in sorted(label_counts.items()))
    if not summary:
        summary = "No objects detected"

    print(f"[detection] Found: {summary}")
    return detections, annotated, summary


def detect_from_bytes(image_bytes: bytes) -> tuple:
    """Convenience wrapper — accepts raw bytes instead of PIL Image."""
    image = Image.open(io.BytesIO(image_bytes))
    return detect_objects(image)
