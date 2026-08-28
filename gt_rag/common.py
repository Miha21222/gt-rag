"""Shared configuration and helpers for the GT DATABASE RAG pipeline."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CHROMA_DIR = ROOT / "data" / "chroma"
STATE_FILE = ROOT / "data" / "index_state.json"
MANIFEST_FILE = ROOT / "manifest.json"

COLLECTION_NAME = "gt_database"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Chunking parameters (characters, not tokens; multilingual MiniLM window is
# small so keep chunks compact).
MAX_CHUNK_CHARS = 1800
OVERLAP_CHARS = 200


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def load_manifest() -> dict:
    return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Split '---\\n...yaml...\\n---\\n<body>' into (meta, body)."""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not m:
        return {}, text
    meta = yaml.safe_load(m.group(1)) or {}
    return meta, text[m.end():]


def split_sections(body: str) -> list[tuple[str, str]]:
    """Split markdown body into (section_title, section_text) by ##/### headings."""
    lines = body.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current: list[str] = []
    for line in lines:
        h = re.match(r"^#{2,4}\s+(.*)", line)
        if h:
            if current and any(s.strip() for s in current):
                sections.append((current_title, current))
            current_title = h.group(1).strip().rstrip("#").strip()
            current = []
        else:
            current.append(line)
    if current and any(s.strip() for s in current):
        sections.append((current_title, current))
    return [(t, "\n".join(ls).strip()) for t, ls in sections if "\n".join(ls).strip()]


def window_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split oversized text into overlapping windows on paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]
    paras = re.split(r"\n\s*\n", text)
    windows: list[str] = []
    buf = ""
    for p in paras:
        if buf and len(buf) + len(p) + 2 > max_chars:
            windows.append(buf.strip())
            # carry tail for overlap
            buf = buf[-overlap:] + "\n\n" + p
        else:
            buf = (buf + "\n\n" + p) if buf else p
    if buf.strip():
        windows.append(buf.strip())
    return windows


def chunk_page(meta: dict, body: str) -> list[Chunk]:
    """Chunk one page into section-level chunks with metadata."""
    slug = meta.get("slug", "unknown")
    title = meta.get("title", slug)
    base_meta = {
        "title": title,
        "slug": slug,
        "type": str(meta.get("type", "")),
        "block": -1 if meta.get("block") is None else int(meta["block"]),
        "notion_url": meta.get("notion_url", ""),
    }
    sections = split_sections(body)
    if not sections:
        sections = [("", body.strip())] if body.strip() else []
    chunks: list[Chunk] = []
    idx = 0
    for sec_title, sec_text in sections:
        for win in window_text(sec_text):
            header = f"{title}" + (f" — {sec_title}" if sec_title else "")
            text = f"{header}\n\n{win}"
            md = dict(base_meta)
            md["section"] = sec_title
            md["chunk_index"] = idx
            chunks.append(Chunk(chunk_id=f"{slug}#{idx}", text=text, metadata=md))
            idx += 1
    return chunks
