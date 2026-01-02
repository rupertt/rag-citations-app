from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from app.core.logging import setup_logging
from app.core.types import AskRequest, AskResponse
from app.rag import answer_question

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="rag-citations-app")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        result = answer_question(req.question, top_k=req.top_k, debug=req.debug)
        return AskResponse(**result)
    except Exception as e:
        logger.exception("Error answering question")
        raise HTTPException(status_code=500, detail=str(e)) from e


