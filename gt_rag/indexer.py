"""Reusable ingest core: index data/raw markdown into ChromaDB.

Print-free so it can be called from the MCP server (stdio transport: stdout
belongs to the protocol). CLI wrappers do their own printing from the report.
"""
from __future__ import annotations

from .common import (
    RAW_DIR,
    chunk_page,
    file_hash,
    load_state,
    parse_front_matter,
    save_state,
)
from .store import get_collection


def ingest(
    rebuild: bool = False,
    only_slugs: list[str] | None = None,
) -> dict:
    """(Re)index raw pages. Incremental by default (file-hash based).

    Args:
        rebuild: wipe the collection and re-embed everything.
        only_slugs: restrict to these slugs (still hash-checked unless rebuild).

    Returns a report dict:
        {"pages": {slug: chunk_count}, "orphans_removed": [...],
         "total_chunks_written": int, "collection_count": int}
    """
    files = sorted(RAW_DIR.glob("*.md"))
    state = {} if rebuild else load_state()
    col = get_collection()

    if rebuild:
        existing = col.get(include=[])["ids"]
        if existing:
            col.delete(ids=existing)

    if only_slugs is not None:
        wanted = set(only_slugs)
        files = [p for p in files if p.stem in wanted]
        # force re-index of explicitly requested slugs
        for s in wanted:
            state.pop(s, None)

    pages: dict[str, int] = {}
    total = 0
    for path in files:
        h = file_hash(path)
        slug = path.stem
        if state.get(slug) == h:
            continue
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        meta.setdefault("slug", slug)
        chunks = chunk_page(meta, body)
        col.delete(where={"slug": slug})
        if chunks:
            col.upsert(
                ids=[c.chunk_id for c in chunks],
                documents=[c.text for c in chunks],
                metadatas=[c.metadata for c in chunks],
            )
        state[slug] = h
        pages[slug] = len(chunks)
        total += len(chunks)

    # prune state entries and chunks for source files that no longer exist
    on_disk = {p.stem for p in sorted(RAW_DIR.glob("*.md"))}
    orphans = [s for s in state if s not in on_disk]
    for s in orphans:
        col.delete(where={"slug": s})
        state.pop(s)

    save_state(state)
    return {
        "pages": pages,
        "orphans_removed": orphans,
        "total_chunks_written": total,
        "collection_count": col.count(),
    }
