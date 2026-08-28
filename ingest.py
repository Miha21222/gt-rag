"""Ingest crawled markdown pages into ChromaDB.

Incremental: pages whose file hash is unchanged since the last run are skipped.
Changed pages have their old chunks deleted and re-embedded.

Usage:
    python ingest.py                 # incremental
    python ingest.py --rebuild       # wipe and re-embed everything
    python ingest.py --only <slug>   # force re-index of one page
"""
from __future__ import annotations

import argparse

from gt_rag.indexer import ingest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="re-embed everything")
    ap.add_argument("--only", action="append", default=None,
                    help="force re-index of this slug (repeatable)")
    args = ap.parse_args()

    r = ingest(rebuild=args.rebuild, only_slugs=args.only)
    for slug, n in r["pages"].items():
        print(f"  {slug}: {n} chunks")
    for slug in r["orphans_removed"]:
        print(f"  {slug}: removed (source file gone)")
    print(f"done: {len(r['pages'])} pages (re)indexed, "
          f"{r['total_chunks_written']} chunks written, "
          f"{r['collection_count']} chunks total in collection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
