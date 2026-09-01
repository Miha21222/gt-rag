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
STYLE_OVERRIDES_FILE = ROOT / "style-overrides.json"

COLLECTION_NAME = "gt_database"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_SCHEMA_VERSION = 2
VALID_TRADING_STYLES = {
    "Для свинга",
    "Для интрадей",
    "Для интрадей и свинга",
}

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


def load_style_overrides() -> dict[tuple[str, str], str]:
    """Return audited section-level trading-style overrides."""
    if not STYLE_OVERRIDES_FILE.exists():
        return {}
    data = json.loads(STYLE_OVERRIDES_FILE.read_text(encoding="utf-8"))
    result: dict[tuple[str, str], str] = {}
    for item in data.get("overrides", []):
        style = item.get("trading_style", data.get("label", ""))
        sections = item.get("sections")
        if style not in VALID_TRADING_STYLES:
            raise ValueError(f"invalid trading style override: {style!r}")
        if not item.get("slug") or not isinstance(sections, list) or not sections:
            raise ValueError("style override requires a slug and a non-empty sections list")
        for section in sections:
            if not isinstance(section, str) or not section:
                raise ValueError("style override sections must be non-empty strings")
            key = (item["slug"], section)
            previous = result.get(key)
            if previous and previous != style:
                raise ValueError(f"conflicting style overrides for {key}")
            result[key] = style
    return result


def index_hash(path: Path) -> str:
    """Hash source, index schema, and the style overrides affecting this page."""
    overrides = sorted(
        (section, style)
        for (slug, section), style in load_style_overrides().items()
        if slug == path.stem
    )
    digest = hashlib.sha256(path.read_bytes())
    digest.update(b"\0index-schema\0")
    digest.update(str(INDEX_SCHEMA_VERSION).encode("ascii"))
    digest.update(b"\0style-overrides\0")
    digest.update(json.dumps(overrides, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


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
    base_trading_style = str(meta.get("trading_style", "Для свинга"))
    if base_trading_style not in VALID_TRADING_STYLES:
        raise ValueError(f"invalid trading style in {slug}: {base_trading_style!r}")
    base_meta = {
        "title": title,
        "slug": slug,
        "type": str(meta.get("type", "")),
        "block": -1 if meta.get("block") is None else int(meta["block"]),
        "notion_url": meta.get("notion_url", ""),
        "trading_style": base_trading_style,
        "base_trading_style": base_trading_style,
    }
    for key in (
        "source_url",
        "guide_url",
        "course",
        "course_level",
        "source_kind",
        "lesson_label",
        "lesson_number",
        "market",
        "review_url",
        "source_video_url",
        "source_video_file",
        "source_video_size",
        "source_video_sha256",
        "source_bundle_file",
        "source_bundle_sha256",
        "duration_seconds",
        "transcript_segments",
        "whisper_model",
    ):
        value = meta.get(key)
        if value is not None and value != "":
            base_meta[key] = value
    sections = split_sections(body)
    if not sections:
        sections = [("", body.strip())] if body.strip() else []
    chunks: list[Chunk] = []
    style_overrides = load_style_overrides()
    overridden_sections = {
        section for (override_slug, section) in style_overrides
        if override_slug == slug
    }
    missing_sections = overridden_sections - {title for title, _ in sections}
    if missing_sections:
        missing = ", ".join(sorted(missing_sections))
        raise ValueError(f"style overrides reference missing sections in {slug}: {missing}")
    idx = 0
    for sec_title, sec_text in sections:
        for win in window_text(sec_text):
            header = f"{title}" + (f" — {sec_title}" if sec_title else "")
            text = f"{header}\n\n{win}"
            md = dict(base_meta)
            md["trading_style"] = style_overrides.get(
                (slug, sec_title), base_trading_style
            )
            md["section"] = sec_title
            md["chunk_index"] = idx
            chunks.append(Chunk(chunk_id=f"{slug}#{idx}", text=text, metadata=md))
            idx += 1
    return chunks
