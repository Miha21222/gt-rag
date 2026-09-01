"""MCP server exposing the GT DATABASE (trading course) knowledge base.

Content tools:
    gt_search      — hybrid semantic + fuzzy search over all course content
    gt_get_page    — full markdown of one page by slug
    gt_list_pages  — inventory of all pages with indexed status
    gt_image_path  — resolve a markdown image link to an absolute file path

Maintenance tools:
    gt_health      — fast read-only diagnosis of the whole pipeline
    gt_doctor      — diagnose and apply safe fixes (incremental re-index)
    gt_reindex     — explicit re-index (incremental or full rebuild)
    gt_stats       — collection breakdown by type and block

Run (stdio):
    python server.py
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from gt_rag.common import (
    RAW_DIR,
    ROOT,
    index_hash,
    load_state,
    load_style_overrides,
    parse_front_matter,
)
from gt_rag.search import GTSearch

INSTRUCTIONS = """\
Knowledge base of a Russian-language Smart Money / ICT trading course
(8 modules, "Block 0-7"): lessons, conference/Q&A transcripts (timestamped),
homework, terminology. Content is Russian with English SMC terms (FVG, IDM,
Dealing Range...); queries work in both languages, typos tolerated.

Recommended flow:
1. gt_search(query) for anything conceptual; filter block=N,
   type=lesson|guide|conference|qa|homework|terminology|info|transcript, or
   trading_style="Для свинга"|"Для интрадей"|"Для интрадей и свинга" to
   narrow. A single-style filter automatically includes shared material.
   Transcripts answer "what was said/asked live"; lessons are the canon.
2. Follow up with gt_get_page(slug) when a hit needs full context.
3. Hits may contain image links `![img-NN](../images/<slug>/img-NN.png)` with
   a caption and a `[Описание графика: ...]` description. To view the actual
   chart, call gt_image_path(...) and read the returned absolute path with
   your file/vision tools.

If search errors or looks stale (content edited but not found): gt_health()
first; gt_doctor(fix=True) applies safe repairs; gt_reindex(rebuild=True)
is the last resort (full re-embed, minutes of work).
"""

mcp = MCPServer(
    "gt-database",
    description="RAG knowledge base of the GT trading course (Smart Money/ICT)",
    instructions=INSTRUCTIONS,
)

_searcher: GTSearch | None = None


def searcher() -> GTSearch:
    global _searcher
    if _searcher is None:
        _searcher = GTSearch()
    return _searcher


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)


def _resolve_under(root: Path, relative: str) -> Path | None:
    """Resolve a user-supplied relative path without escaping its root."""
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


@mcp.tool()
def gt_search(
    query: str,
    top_k: int = 5,
    block: int | None = None,
    type: str | None = None,
    trading_style: str | None = None,
) -> str:
    """Search the GT trading course knowledge base (Smart Money / ICT concepts:
    FVG, liquidity, order flow, dealing range, risk, prop trading...).
    Understands Russian and English queries, tolerates typos. Results include
    the chunk text (may contain image links + chart descriptions) and the
    source slug for gt_get_page.

    Args:
        query: what to look for (question, topic, or term).
        top_k: number of results (default 5).
        block: optional course module filter, 0-7.
        type: optional filter: lesson | guide | conference | qa | homework |
              terminology | info | transcript.
        trading_style: optional filter: Для свинга | Для интрадей |
                       Для интрадей и свинга. Swing/intraday filters also
                       include shared chunks.
    """
    hits = searcher().search(
        query,
        top_k=top_k,
        block=block,
        type_=type,
        trading_style=trading_style,
    )
    out = []
    for h in hits:
        m = h.metadata
        out.append({
            "score": h.score,
            "title": m.get("title"),
            "section": m.get("section") or None,
            "block": None if m.get("block", -1) == -1 else m.get("block"),
            "type": m.get("type"),
            "trading_style": m.get("trading_style"),
            "base_trading_style": m.get("base_trading_style", m.get("trading_style")),
            "course": m.get("course"),
            "course_level": m.get("course_level"),
            "market": m.get("market"),
            "slug": m.get("slug"),
            "notion_url": m.get("notion_url"),
            "text": h.text,
        })
    return _dump(out)


@mcp.tool()
def gt_get_page(slug: str) -> str:
    """Return the full markdown content of one course page by its slug
    (slugs come from gt_search results or gt_list_pages). Image links inside
    are relative to data/raw/; resolve them with gt_image_path."""
    path = _resolve_under(RAW_DIR, f"{slug}.md")
    if path is None:
        return "error: invalid slug path"
    if not path.exists():
        return f"error: no page with slug '{slug}'. Use gt_list_pages to see valid slugs."
    return path.read_text(encoding="utf-8")


@mcp.tool()
def gt_list_pages() -> str:
    """List all pages in the knowledge base: title, block, type, slug, source
    and audited trading styles, and whether the page is indexed for search."""
    state = load_state()
    style_overrides = load_style_overrides()
    rows = []
    for path in sorted(RAW_DIR.glob("*.md")):
        meta, _ = parse_front_matter(path.read_text(encoding="utf-8"))
        slug = meta.get("slug", path.stem)
        base_style = meta.get("trading_style", "Для свинга")
        styles = {base_style}
        styles.update(
            style for (override_slug, _), style in style_overrides.items()
            if override_slug == slug
        )
        rows.append({
            "slug": slug,
            "title": meta.get("title", path.stem),
            "block": meta.get("block"),
            "type": meta.get("type"),
            "trading_style": base_style,
            "trading_styles": sorted(styles),
            "course": meta.get("course", "GT DATABASE"),
            "course_level": meta.get("course_level"),
            "market": meta.get("market"),
            "indexed": state.get(path.stem) == index_hash(path),
        })
    return _dump(rows)


@mcp.tool()
def gt_image_path(link: str) -> str:
    """Resolve an image link from page/search content to an absolute file path
    you can open with file/vision tools.

    Args:
        link: as it appears in markdown, e.g. '../images/b3-dealing-range/img-01.png'
              or 'b3-dealing-range/img-01.png'.
    """
    images_dir = ROOT / "data" / "images"
    prefix = "../images/"
    relative = link[len(prefix):] if link.startswith(prefix) else link
    path = _resolve_under(images_dir, relative)
    if path is None:
        return _dump({"path": None, "exists": False, "error": "invalid image path"})
    return _dump({
        "path": str(path),
        "exists": path.exists(),
        **({} if path.exists() else
           {"hint": "file missing — gt_health() shows broken image links"}),
    })


@mcp.tool()
def gt_stats() -> str:
    """Collection statistics: total chunks and breakdown by content type and
    course block. Cheap sanity check that the index is populated."""
    from gt_rag.store import get_collection
    col = get_collection()
    metas = col.get(include=["metadatas"])["metadatas"]
    by_type = Counter(m.get("type", "?") for m in metas)
    by_block = Counter(m.get("block", -1) for m in metas)
    by_trading_style = Counter(m.get("trading_style", "?") for m in metas)
    by_course = Counter(m.get("course", "GT DATABASE") for m in metas)
    return _dump({
        "total_chunks": len(metas),
        "pages_on_disk": len(list(RAW_DIR.glob("*.md"))),
        "by_type": dict(by_type.most_common()),
        "by_block": {str(k): v for k, v in sorted(by_block.items())},
        "by_trading_style": dict(by_trading_style.most_common()),
        "by_course": dict(by_course.most_common()),
    })


@mcp.tool()
def gt_health() -> str:
    """Fast read-only health check of the whole pipeline: raw files,
    ChromaDB collection, index freshness, image archive, live search smoke
    test. Returns status ok | warn | error per check plus remediation hints.
    Call this first when anything misbehaves."""
    from gt_rag.doctor import health
    return _dump(health())


@mcp.tool()
def gt_doctor(fix: bool = False) -> str:
    """Diagnose and repair the knowledge base.

    fix=False: report problems and what would be done.
    fix=True: apply SAFE fixes — incremental re-index of stale/new pages,
    pruning of orphaned index entries. Never wipes the collection; for a
    corrupt store use gt_reindex(rebuild=True)."""
    from gt_rag.doctor import doctor
    return _dump(doctor(fix=fix))


@mcp.tool()
def gt_reindex(rebuild: bool = False) -> str:
    """Re-index content into ChromaDB.

    rebuild=False: incremental — only pages whose files changed (fast).
    rebuild=True: wipe the collection and re-embed everything (minutes;
    use only when the store is corrupt or chunking/embedding code changed)."""
    from gt_rag.indexer import ingest
    r = ingest(rebuild=rebuild)
    return _dump(r)


if __name__ == "__main__":
    mcp.run()
