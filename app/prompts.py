from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=16)
def load_prompt(name: str) -> str:
    """
    Loads a prompt markdown file from ./prompts/{name}.md relative to this project.
    """
    base_dir = Path(__file__).resolve().parent.parent / "prompts"
    path = base_dir / f"{name}.md"
    return path.read_text(encoding="utf-8")


