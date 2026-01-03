from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

# ---- Chroma requires sqlite3 >= 3.35.0 on Linux/WSL.
# If the Python stdlib sqlite3 module is linked against an older libsqlite3,
# use the bundled pysqlite3-binary instead (no system upgrade required).
try:
    import sqlite3  # noqa: F401

    def _sqlite_version_tuple(v: str) -> tuple[int, int, int]:
        parts = (v.split(".") + ["0", "0", "0"])[:3]
        return int(parts[0]), int(parts[1]), int(parts[2])

    _min_sqlite = (3, 35, 0)
    _have_sqlite = _sqlite_version_tuple(sqlite3.sqlite_version)  # type: ignore[attr-defined]
    if _have_sqlite < _min_sqlite:
        import sys

        import pysqlite3  # type: ignore[import-not-found]

        sys.modules["sqlite3"] = pysqlite3
except Exception:
    # If any of this fails, we let the normal import error surface when Chroma loads.
    pass

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


def _raw_dir() -> Path:
    """
    Optional multi-document input directory.

    If it exists and contains supported files, we index those instead of the single hardcoded doc.txt.
    This preserves existing single-doc behavior by default (no ./data/raw directory required).
    """
    return _data_dir() / "raw"


def _supported_raw_files() -> list[Path]:
    """
    Return supported raw document files (txt/md) from ./data/raw.

    Notes:
    - Sorted for deterministic indexing behavior.
    - If ./data/raw does not exist or is empty, we fall back to the legacy single doc.txt.
    """
    raw = _raw_dir()
    if not raw.exists():
        return []
    files: list[Path] = []
    for p in raw.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() in {".txt", ".md"}:
            files.append(p)
    return sorted(files, key=lambda x: x.name.lower())


def _index_sources() -> list[Path]:
    """
    Determine which document sources should be indexed.

    Behavior:
    - If ./data/raw contains any supported docs, index those (multi-doc mode).
    - Otherwise, index the legacy ./data/doc.txt only (single-doc mode).
    """
    raw_files = _supported_raw_files()
    if raw_files:
        return raw_files
    return [_doc_path()]


def _index_dir() -> Path:
    return _data_dir() / "index"


def _fingerprint_path() -> Path:
    """
    Stores the fingerprint of the currently-indexed doc/settings so we can detect changes
    and rebuild the persistent Chroma index automatically.
    """
    return _index_dir() / "doc_fingerprint.json"


def load_doc() -> str:
    return _doc_path().read_text(encoding="utf-8")


def _compute_fingerprint() -> dict[str, Any]:
    """
    Fingerprint includes document contents + indexing-relevant settings.

    Notes:
    - We hash the raw bytes to avoid any encoding-normalization surprises.
    - If any of these fields change, we should rebuild the index.
    """
    sources = _index_sources()
    sources_fp: list[dict[str, str]] = []
    for p in sources:
        # Hash raw bytes to avoid encoding normalization differences.
        b = p.read_bytes()
        sources_fp.append({"source": p.name, "sha256": hashlib.sha256(b).hexdigest()})
    return {
        "sources": sources_fp,
        "chunk_size": int(settings.chunk_size),
        "chunk_overlap": int(settings.chunk_overlap),
        "embedding_model": str(settings.embedding_model),
    }


def _read_fingerprint() -> dict[str, Any] | None:
    path = _fingerprint_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to read index fingerprint; forcing re-index.")
        return None


def _write_fingerprint(fp: dict[str, Any]) -> None:
    _fingerprint_path().write_text(json.dumps(fp, indent=2), encoding="utf-8")


def _split_sections(text: str) -> list[tuple[str, str]]:
    """
    Split a document into (section_title, section_text) using a hierarchical approach.

    Supports:
    - Markdown ATX headings: "# Title", "## Title", ...
    - Setext headings:
        Title
        -----
      (or =====)

    If no headings are detected, returns a single ("", full_text) section.
    """
    lines = (text or "").splitlines()
    sections: list[tuple[str, list[str]]] = []
    cur_title = ""
    cur_lines: list[str] = []

    def _flush() -> None:
        nonlocal cur_title, cur_lines
        if cur_lines:
            sections.append((cur_title.strip(), cur_lines[:]))
        cur_lines = []

    i = 0
    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()

        # ATX heading: "# Title"
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                _flush()
                cur_title = title
                i += 1
                continue

        # Setext heading:
        # Title
        # ------
        if i + 1 < len(lines):
            next_ln = lines[i + 1].strip()
            if next_ln and set(next_ln) <= {"=", "-"} and len(next_ln) >= 3 and stripped:
                _flush()
                cur_title = stripped
                i += 2
                continue

        cur_lines.append(ln)
        i += 1

    _flush()

    # If we never created a titled section, fall back to one big section.
    if not any(t for t, _ in sections):
        return [("", text)]

    out: list[tuple[str, str]] = []
    for title, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        out.append((title, body))
    return out or [("", text)]


def chunk_doc(text: str, *, source: str) -> list[Document]:
    """
    Chunk a single document into stable chunk IDs and attach metadata.

    Requirements:
    - Chunk IDs are deterministic and stable: chunk-00, chunk-01, ...
    - Metadata includes:
      - source: filename
      - section: section header title (if available)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    docs: list[Document] = []
    chunk_index = 0

    for section_title, section_text in _split_sections(text):
        # Split within section for better locality; include the title as context if present.
        prefix = f"{section_title}\n\n" if section_title else ""
        chunks = splitter.split_text(prefix + section_text)
        for chunk in chunks:
            chunk_id = f"chunk-{chunk_index:02d}"
            chunk_index += 1
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source,
                        "chunk_id": chunk_id,
                        "section": section_title or "",
                    },
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
    Create or refresh the persistent Chroma index.

    Because we persist under ./data/index, we also persist a small fingerprint file so
    we can detect when doc.txt (or chunking/embedding settings) changed and rebuild the index.
    """
    vs = get_vectorstore()
    try:
        count = vs._collection.count()  # noqa: SLF001 (Chroma wrapper doesn't expose count)
    except Exception:  # pragma: no cover
        logger.exception("Failed to check Chroma collection count; re-indexing to be safe.")
        count = 0

    current_fp = _compute_fingerprint()
    stored_fp = _read_fingerprint()

    # If we have vectors and the fingerprint matches, do nothing.
    if count and count > 0 and stored_fp == current_fp:
        return

    # Build docs for all sources (single-doc legacy or multi-doc mode).
    docs: list[Document] = []
    ids: list[str] = []
    for src_path in _index_sources():
        text = src_path.read_text(encoding="utf-8")
        src_docs = chunk_doc(text, source=src_path.name)
        docs.extend(src_docs)
        # Chroma IDs must be unique across ALL docs; chunk_id alone collides in multi-doc mode.
        ids.extend([f"{src_path.name}::{d.metadata['chunk_id']}" for d in src_docs])

    # If vectors exist but doc/settings changed, clear them before re-adding.
    if count and count > 0:
        logger.info("Doc/settings changed (or fingerprint missing); rebuilding index.")
        try:
            # Delete all records in the collection.
            vs._collection.delete(where={})  # noqa: SLF001
        except Exception:
            logger.exception("Failed to clear existing Chroma collection; continuing (may duplicate).")

    vs.add_documents(docs, ids=ids)
    # Persistence happens automatically for persistent_directory, but calling persist is harmless
    try:
        vs.persist()
    except Exception:
        # Some versions deprecate persist() on wrapper; ignore if not available.
        pass

    # Save fingerprint so we can detect doc changes next time.
    _write_fingerprint(current_fp)


def get_retriever(top_k: int) -> Any:
    vs = get_vectorstore()
    # Prefer MMR (more diverse results) where supported by the vectorstore wrapper.
    # If unsupported, LangChain will fall back to similarity search.
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": top_k})


def _citation_key(doc: Document) -> str:
    """
    Canonical key for (source, chunk_id) pairs used in citations and deduplication.
    """
    src = str(doc.metadata.get("source", DOC_SOURCE))
    cid = str(doc.metadata.get("chunk_id", "chunk-??"))
    return f"{src}#{cid}"


def _select_diverse(
    candidates: list[tuple[Document, float]],
    *,
    top_k: int,
) -> list[tuple[Document, float]]:
    """
    Select up to top_k candidates with basic diversity constraints.

    Notes:
    - We deduplicate by citation key first.
    - We then avoid returning everything from the same section when possible.
    """
    # Deduplicate by citation key first (multi-query merges will produce duplicates).
    seen: set[str] = set()
    deduped: list[tuple[Document, float]] = []
    for d, s in candidates:
        key = _citation_key(d)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((d, s))

    if len(deduped) <= top_k:
        return deduped

    # Soft cap: at most 2 per (source, section) group when possible.
    out: list[tuple[Document, float]] = []
    per_group: dict[tuple[str, str], int] = {}
    for d, s in deduped:
        src = str(d.metadata.get("source", DOC_SOURCE))
        sec = str(d.metadata.get("section", ""))[:120]
        group = (src, sec)
        if per_group.get(group, 0) >= 2 and len(per_group) > 1:
            continue
        out.append((d, s))
        per_group[group] = per_group.get(group, 0) + 1
        if len(out) >= top_k:
            break

    # If we were too strict and didn't fill, backfill from remaining deduped.
    if len(out) < top_k:
        chosen = {_citation_key(d) for d, _ in out}
        for d, s in deduped:
            if _citation_key(d) in chosen:
                continue
            out.append((d, s))
            if len(out) >= top_k:
                break
    return out


def retrieve_with_scores(question: str, top_k: int) -> list[tuple[Document, float]]:
    """
    Retrieve documents with a score field.

    Behavior:
    - Prefer MMR (diverse) where available.
    - Fall back to similarity_search_with_score.
    """
    vs = get_vectorstore()
    fetch_k = max(int(top_k) * 4, 20)
    try:
        docs = vs.max_marginal_relevance_search(question, k=top_k, fetch_k=fetch_k)
        # MMR does not naturally expose a score; return a deterministic placeholder.
        return [(d, 0.0) for d in docs]
    except Exception:
        # Chroma returns "distance" (lower is better) for many distance metrics.
        candidates = vs.similarity_search_with_score(question, k=fetch_k)
        return _select_diverse(candidates, top_k=top_k)


def _build_context(retrieved: list[tuple[Document, float]]) -> str:
    parts: list[str] = []
    for doc, _score in retrieved:
        src = str(doc.metadata.get("source", DOC_SOURCE))
        chunk_id = str(doc.metadata.get("chunk_id", "chunk-??"))
        parts.append(f"[{src}#{chunk_id}]\n{doc.page_content}".strip())
    return "\n\n".join(parts).strip()


def _has_citation_token(text: str) -> bool:
    # Generalized: allow [<filename>#chunk-XX] for multi-doc mode.
    t = text or ""
    return ("[" in t) and ("#chunk-" in t)


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
        src = str(doc.metadata.get("source", DOC_SOURCE))
        snippet = doc.page_content.strip().replace("\n", " ")
        snippet = snippet[:240] + ("..." if len(snippet) > 240 else "")
        citations.append({"source": src, "chunk_id": chunk_id, "snippet": snippet})
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
            )
        ),
    ]

    resp = llm.invoke(messages)
    answer = (resp.content or "").strip()

    # Deterministic post-check: refuse unless there is at least one citation token AND
    # all cited (source, chunk_id) pairs were actually retrieved.
    if not _has_citation_token(answer):
        answer = "I can’t find that in the provided documentation."
    else:
        import re

        # Extract strict citations: [<filename>#chunk-XX]
        cited_pairs = re.findall(r"\[([^\]#]+)#(chunk-\d+)\]", answer)
        if not cited_pairs:
            answer = "I can’t find that in the provided documentation."
        else:
            allowed = {_citation_key(d) for d, _ in retrieved}
            for src, cid in cited_pairs:
                if f"{src}#{cid}" not in allowed:
                    answer = "I can’t find that in the provided documentation."
                    break

    out2: dict[str, Any] = {"answer": answer, "citations": citations}
    if debug:
        out2["debug"] = {"retrieved": debug_retrieved}
    return out2


