"""Quick CLI check of the hybrid search.

Usage:
    python query.py "что такое FVG"
    python query.py "premium discount" --top-k 3 --block 3
"""
from __future__ import annotations

import argparse

from gt_rag.search import GTSearch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--block", type=int, default=None)
    ap.add_argument("--type", dest="type_", default=None)
    args = ap.parse_args()

    for h in GTSearch().search(args.query, top_k=args.top_k, block=args.block, type_=args.type_):
        m = h.metadata
        sec = f" › {m['section']}" if m.get("section") else ""
        print(f"[{h.score:.3f}] ({m['type']}, block {m['block']}) {m['title']}{sec}")
        preview = h.text.replace("\n", " ")
        print(f"    {preview[:220]}...\n")


if __name__ == "__main__":
    main()
