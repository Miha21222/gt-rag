"""Health checks and self-repair for the knowledge base.

health()          — fast, read-only diagnosis; safe to call anytime.
doctor(fix=True)  — same checks, then applies the safe fixes (incremental
                    re-index of stale/new pages, pruning of orphaned chunks).
                    Never wipes the collection; that is gt_reindex(rebuild).

Print-free: called from the MCP server over stdio.
"""
from __future__ import annotations

import re
import time

from .common import (
    IMAGES_DIR,
    RAW_DIR,
    chunk_page,
    index_hash,
    load_state,
    load_style_overrides,
    parse_front_matter,
)

IMG_LINK_RE = re.compile(r"!\[[^\]]*\]\(\.\./images/([^)]+)\)")
REQUIRED_META = ("title", "slug", "type")


def _check_raw_files() -> dict:
    if not RAW_DIR.is_dir():
        return {"name": "raw_files", "status": "error",
                "detail": f"{RAW_DIR} does not exist — no source content. "
                          "Populate data/raw/*.md (see README data format)."}
    files = sorted(RAW_DIR.glob("*.md"))
    bad: list[str] = []
    for p in files:
        try:
            meta, body = parse_front_matter(p.read_text(encoding="utf-8"))
        except Exception as e:
            bad.append(f"{p.name}: unreadable frontmatter ({e})")
            continue
        missing = [k for k in REQUIRED_META if not meta.get(k)]
        if missing:
            bad.append(f"{p.name}: missing frontmatter keys {missing}")
            continue
        try:
            chunk_page(meta, body)
        except Exception as e:
            bad.append(f"{p.name}: invalid chunk metadata ({e})")
    try:
        override_slugs = {slug for slug, _ in load_style_overrides()}
        missing_slugs = sorted(override_slugs - {p.stem for p in files})
        bad.extend(f"style override references missing slug: {slug}" for slug in missing_slugs)
    except Exception as e:
        bad.append(f"style-overrides.json: {e}")
    status = "error" if not files else ("warn" if bad else "ok")
    return {"name": "raw_files", "status": status,
            "file_count": len(files), "problems": bad}


def _check_collection() -> dict:
    try:
        from .store import get_collection
        col = get_collection()
        count = col.count()
    except Exception as e:
        return {"name": "collection", "status": "error",
                "detail": f"ChromaDB failed to open: {e}. If the store is "
                          "corrupt, run gt_reindex(rebuild=True)."}
    status = "warn" if count == 0 else "ok"
    return {"name": "collection", "status": status, "chunk_count": count,
            **({"detail": "collection is empty — run gt_doctor(fix=True)"}
               if count == 0 else {})}


def _check_index_freshness() -> dict:
    try:
        state = load_state()
        files = {p.stem: p for p in sorted(RAW_DIR.glob("*.md"))} if RAW_DIR.is_dir() else {}
        stale = [s for s, p in files.items() if state.get(s) not in (None, index_hash(p))]
    except Exception as e:
        return {"name": "index_freshness", "status": "error",
                "detail": f"could not calculate index freshness: {e}"}
    unindexed = [s for s in files if s not in state]
    orphan_state = [s for s in state if s not in files]
    ok = not (stale or unindexed or orphan_state)
    return {"name": "index_freshness", "status": "ok" if ok else "warn",
            "stale_pages": stale, "unindexed_pages": unindexed,
            "orphaned_index_entries": orphan_state,
            **({} if ok else {"detail": "run gt_doctor(fix=True) to re-index"})}


def _check_images() -> dict:
    if not IMAGES_DIR.is_dir():
        return {"name": "images", "status": "ok", "detail":
                "no data/images dir (image archive not in use)"}
    disk = {str(p.relative_to(IMAGES_DIR)).replace("\\", "/")
            for p in IMAGES_DIR.rglob("*") if p.is_file()}
    linked: set[str] = set()
    for p in RAW_DIR.glob("*.md"):
        linked.update(IMG_LINK_RE.findall(p.read_text(encoding="utf-8")))
    broken = sorted(linked - disk)
    orphans = sorted(disk - linked)
    status = "warn" if broken else "ok"
    return {"name": "images", "status": status,
            "on_disk": len(disk), "linked": len(linked),
            "broken_links": broken[:20], "unlinked_files": orphans[:20],
            **({"detail": "broken links need a re-fetch of the source page "
                          "(image archive pass); search still works via "
                          "captions/descriptions"} if broken else {})}


def _check_search() -> dict:
    try:
        from .search import GTSearch
        t0 = time.time()
        hits = GTSearch().search("test", top_k=1)
        ms = int((time.time() - t0) * 1000)
    except Exception as e:
        return {"name": "search_smoke", "status": "error",
                "detail": f"search failed: {e}. Check that the embedding "
                          "model downloaded (first ingest fetches ~470 MB) "
                          "and the collection is intact."}
    if not hits:
        return {"name": "search_smoke", "status": "warn", "latency_ms": ms,
                "detail": "search returned nothing — collection likely empty"}
    return {"name": "search_smoke", "status": "ok", "latency_ms": ms}


def health() -> dict:
    """Fast read-only diagnosis of the whole pipeline."""
    checks = [_check_raw_files(), _check_collection(),
              _check_index_freshness(), _check_images(), _check_search()]
    worst = ("error" if any(c["status"] == "error" for c in checks)
             else "warn" if any(c["status"] == "warn" for c in checks)
             else "ok")
    return {"status": worst, "checks": checks}


def doctor(fix: bool = False) -> dict:
    """Diagnose and (optionally) apply safe fixes.

    Safe fixes: incremental re-index of stale/new pages, pruning of orphaned
    index entries and chunks. Never wipes the collection.
    """
    report = health()
    if not fix:
        report["fixes_applied"] = []
        report["hint"] = "call with fix=True to apply safe fixes"
        return report

    fixes: list[str] = []
    by_name = {c["name"]: c for c in report["checks"]}

    fresh = by_name.get("index_freshness", {})
    col = by_name.get("collection", {})
    needs_ingest = (
        fresh.get("stale_pages") or fresh.get("unindexed_pages")
        or fresh.get("orphaned_index_entries")
        or (col.get("chunk_count") == 0 and by_name["raw_files"].get("file_count"))
    )
    if needs_ingest and col.get("status") != "error":
        from .indexer import ingest
        r = ingest()
        fixes.append(
            f"re-indexed {len(r['pages'])} pages "
            f"({r['total_chunks_written']} chunks), removed "
            f"{len(r['orphans_removed'])} orphans; collection now "
            f"{r['collection_count']} chunks")

    report_after = health()
    report_after["fixes_applied"] = fixes
    if not fixes:
        report_after["hint"] = (
            "healthy — nothing to fix" if report_after["status"] == "ok"
            else "nothing auto-fixable. Collection errors: "
                 "gt_reindex(rebuild=True). Broken image links: re-run the "
                 "image archive pass.")
    return report_after
