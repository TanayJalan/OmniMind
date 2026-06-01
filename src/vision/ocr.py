# ─────────────────────────────────────────────────────────────────
#  ocr.py — Extract text from images using Tesseract OCR
#
#  HOW OCR WORKS:
#    Tesseract analyses pixel patterns in an image to recognise
#    characters. It works best on clean, high-contrast images.
#    We pre-process the image (grayscale + contrast boost) to
#    improve accuracy before passing it to Tesseract.
#
#  INSTALL REQUIREMENT:
#    brew install tesseract   ← the actual OCR engine (binary)
#    pip install pytesseract  ← Python wrapper that calls it
# ─────────────────────────────────────────────────────────────────

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Pre-process image to improve OCR accuracy.

    Steps:
      1. Convert to grayscale — removes colour noise
      2. Sharpen — makes text edges crisper
      3. Boost contrast — makes text stand out from background
    """
    image = image.convert("L")                     # grayscale
    image = image.filter(ImageFilter.SHARPEN)      # sharpen edges
    image = ImageEnhance.Contrast(image).enhance(2.0)  # boost contrast
    return image


def extract_text(image: Image.Image) -> dict:
    """
    Extract all text from an image using Tesseract.

    pytesseract.image_to_data() returns a detailed breakdown
    including confidence scores for each word detected.

    Returns a dict with:
      "text"       → full extracted text as a single string
      "word_count" → number of words found
      "confidence" → average confidence score (0-100)
    """
    processed = preprocess_image(image)

    # Get detailed data including confidence per word
    data = pytesseract.image_to_data(
        processed,
        output_type=pytesseract.Output.DICT,
        config="--psm 3",   # psm 3 = fully automatic page segmentation
    )

    # Filter out low-confidence detections (noise)
    words = []
    confidences = []
    for i, word in enumerate(data["text"]):
        conf = int(data["conf"][i])
        if conf > 30 and word.strip():   # ignore if < 30% confident
            words.append(word)
            confidences.append(conf)

    full_text = " ".join(words)
    avg_conf = round(sum(confidences) / len(confidences), 1) if confidences else 0

    print(f"[ocr] Extracted {len(words)} words, avg confidence: {avg_conf}%")

    return {
        "text": full_text,
        "word_count": len(words),
        "confidence": avg_conf,
    }


def extract_text_from_bytes(image_bytes: bytes) -> dict:
    """Convenience wrapper — accepts raw bytes instead of PIL Image."""
    image = Image.open(io.BytesIO(image_bytes))
    return extract_text(image)
