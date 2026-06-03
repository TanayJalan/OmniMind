# ─────────────────────────────────────────────────────────────────
#  visual_qa.py — Answer questions about images using Gemini Flash
#
#  HOW VISION LLMs WORK:
#    A multimodal model encodes both the image (via a vision encoder
#    like ViT) and the text question, then generates an answer that
#    combines understanding of both inputs.
#
#  WHY GEMINI FLASH?
#    - Free tier: 1500 requests/day, 1M tokens/minute
#    - Excellent vision understanding
#    - No GPU needed — it runs in the cloud
#    - Easy API, same key as Google AI Studio
#
#  Get your free key at: https://aistudio.google.com/apikey
# ─────────────────────────────────────────────────────────────────

import os
import io
import base64
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

load_dotenv()

# Configure Gemini with API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# gemini-2.5-flash — fast, free, multimodal
MODEL_NAME = "gemini-2.5-flash"


def ask_about_image(image: Image.Image, question: str) -> dict:
    """
    Ask any question about a PIL Image using Gemini Flash.

    Gemini accepts images directly as PIL Image objects — no need
    to manually encode to base64 (the library handles that).

    Returns a dict with:
      "answer"     → Gemini's response string
      "model"      → model used
      "question"   → the original question (useful for display)
    """
    model = genai.GenerativeModel(MODEL_NAME)

    # Gemini's generate_content accepts a list of [image, text]
    # It processes both together and generates a grounded answer
    response = model.generate_content(
        [image, question],
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,       # low temperature = more factual
            max_output_tokens=1024,
        ),
    )

    answer = response.text
    print(f"[visual_qa] Question: '{question[:50]}...' → answered ({len(answer)} chars)")

    return {
        "answer": answer,
        "model": MODEL_NAME,
        "question": question,
    }


def describe_image(image: Image.Image) -> str:
    """
    Auto-describe an image without a specific question.
    Useful as a default when the user just uploads an image.
    """
    result = ask_about_image(
        image,
        "Describe this image in detail. List the main objects, "
        "scene, colours, and anything notable you observe.",
    )
    return result["answer"]


def ask_from_bytes(image_bytes: bytes, question: str) -> dict:
    """Convenience wrapper — accepts raw bytes instead of PIL Image."""
    image = Image.open(io.BytesIO(image_bytes))
    return ask_about_image(image, question)
