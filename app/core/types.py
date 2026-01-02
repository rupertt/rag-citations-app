from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(4, ge=1, le=20)
    debug: bool = False


class Citation(BaseModel):
    source: str
    chunk_id: str
    snippet: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float


class DebugInfo(BaseModel):
    retrieved: list[RetrievedChunk]


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    debug: DebugInfo | None = None


