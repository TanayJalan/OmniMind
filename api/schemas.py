# ─────────────────────────────────────────────────────────────────
#  schemas.py — Pydantic models for FastAPI request & response
#
#  WHY PYDANTIC?
#    FastAPI uses Pydantic to automatically validate incoming JSON,
#    generate API docs, and serialise responses. If a request is
#    missing a required field or has the wrong type, FastAPI rejects
#    it with a clear 422 error before it even reaches your code.
# ─────────────────────────────────────────────────────────────────

from pydantic import BaseModel, Field
from typing import Optional


# ── Chat (RAG) ────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000,
                          example="What is gradient descent?")
    session_id: Optional[str] = Field(None,
                          example="user_123")   # for future multi-user support


class SourceChunk(BaseModel):
    source: str
    page: Optional[int] = None
    excerpt: str        # first 300 chars of the chunk


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    response_time_ms: float


# ── Vision ────────────────────────────────────────────────────────

class OCRResponse(BaseModel):
    text: str
    word_count: int
    confidence: float


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: list[int]     # [x1, y1, x2, y2]


class DetectionResponse(BaseModel):
    detections: list[Detection]
    summary: str
    object_count: int


class VisualQARequest(BaseModel):
    question: str = Field(..., example="What colour is the car?")


class VisualQAResponse(BaseModel):
    answer: str
    question: str
    model: str


# ── Health ────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict[str, bool]
