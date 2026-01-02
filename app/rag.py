from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.prompts import load_prompt

logger = logging.getLogger(__name__)

DOC_SOURCE = "doc.txt"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _data_dir() -> Path:
    return _project_root() / "data"


def _doc_path() -> Path:
    return _data_dir() / DOC_SOURCE


def _index_dir() -> Path:
    return _data_dir() / "index"


def load_doc() -> str:
    return _doc_path().read_text(encoding="utf-8")


def chunk_doc(text: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)

    docs: list[Document] = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"chunk-{i:02d}"
        docs.append(
            Document(
                page_content=chunk,
                metadata={"source": DOC_SOURCE, "chunk_id": chunk_id},
            )
        )
    return docs


def _get_embeddings() -> OpenAIEmbeddings:
    # langchain-openai reads OPENAI_API_KEY env var; we keep settings for clarity/validation.
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY is empty; OpenAI calls will fail until it is set.")
    return OpenAIEmbeddings(model=settings.embedding_model)


def _get_llm() -> ChatOpenAI:
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY is empty; OpenAI calls will fail until it is set.")
    return ChatOpenAI(model=settings.model, temperature=0)


def get_vectorstore() -> Chroma:
    _index_dir().mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name="doc",
        persist_directory=str(_index_dir()),
        embedding_function=_get_embeddings(),
    )


def index_if_needed() -> None:
    """
    Create or refresh the persistent Chroma index if empty.
    """
    vs = get_vectorstore()
    try:
        count = vs._collection.count()  # noqa: SLF001 (Chroma wrapper doesn't expose count)
    except Exception:  # pragma: no cover
        logger.exception("Failed to check Chroma collection count; re-indexing to be safe.")
        count = 0

    if count and count > 0:
        return

    text = load_doc()
    docs = chunk_doc(text)
    ids = [d.metadata["chunk_id"] for d in docs]

    if count and count > 0:
        try:
            vs._collection.delete(where={})  # delete all
        except Exception:
            logger.exception("Failed to clear existing Chroma collection; continuing.")

    vs.add_documents(docs, ids=ids)
    # Persistence happens automatically for persistent_directory, but calling persist is harmless
    try:
        vs.persist()
    except Exception:
        # Some versions deprecate persist() on wrapper; ignore if not available.
        pass


def get_retriever(top_k: int) -> Any:
    vs = get_vectorstore()
    return vs.as_retriever(search_kwargs={"k": top_k})


def retrieve_with_scores(question: str, top_k: int) -> list[tuple[Document, float]]:
    vs = get_vectorstore()
    # Chroma returns "distance" (lower is better) for many distance metrics.
    return vs.similarity_search_with_score(question, k=top_k)


def _build_context(retrieved: list[tuple[Document, float]]) -> str:
    parts: list[str] = []
    for doc, _score in retrieved:
        chunk_id = doc.metadata.get("chunk_id", "chunk-??")
        parts.append(f"[{DOC_SOURCE}#{chunk_id}]\n{doc.page_content}".strip())
    return "\n\n".join(parts).strip()


def _has_citation_token(text: str) -> bool:
    return f"[{DOC_SOURCE}#chunk-" in text


def answer_question(
    question: str,
    top_k: int = 4,
    debug: bool = False,
) -> dict[str, Any]:
    index_if_needed()

    retrieved = retrieve_with_scores(question, top_k=top_k)
    citations = []
    debug_retrieved = []

    for doc, score in retrieved:
        chunk_id = str(doc.metadata.get("chunk_id", "chunk-??"))
        snippet = doc.page_content.strip().replace("\n", " ")
        snippet = snippet[:240] + ("..." if len(snippet) > 240 else "")
        citations.append({"source": DOC_SOURCE, "chunk_id": chunk_id, "snippet": snippet})
        debug_retrieved.append({"chunk_id": chunk_id, "text": doc.page_content, "score": float(score)})

    if len(retrieved) == 0:
        out: dict[str, Any] = {
            "answer": "I can’t find that in the provided documentation.",
            "citations": citations,
        }
        if debug:
            out["debug"] = {"retrieved": debug_retrieved}
        return out

    system = load_prompt("system")
    answer_tmpl = load_prompt("answer")
    context = _build_context(retrieved)

    llm = _get_llm()
    messages = [
        SystemMessage(content=system),
        HumanMessage(
            content=answer_tmpl.format(
                question=question,
                context=context,
                source=DOC_SOURCE,
            )
        ),
    ]

    resp = llm.invoke(messages)
    answer = (resp.content or "").strip()

    if not _has_citation_token(answer):
        answer = "I can’t find that in the provided documentation."

    out2: dict[str, Any] = {"answer": answer, "citations": citations}
    if debug:
        out2["debug"] = {"retrieved": debug_retrieved}
    return out2


