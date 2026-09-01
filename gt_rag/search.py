"""Hybrid search over the GT DATABASE collection.

Combines ChromaDB semantic (vector) search with a rapidfuzz keyword layer so
that both meaning-based queries ("почему цена заполняет имбаланс") and
exact-ish / typo'd term lookups ("FVG", "premium discont") rank well.
"""
from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz

from .store import get_collection

VECTOR_WEIGHT = 0.65
FUZZY_WEIGHT = 0.35
# how many vector candidates to pull before re-ranking
CANDIDATE_MULTIPLIER = 4
SWING_STYLE = "Для свинга"
INTRADAY_STYLE = "Для интрадей"
COMBINED_STYLE = "Для интрадей и свинга"


@dataclass
class Hit:
    chunk_id: str
    score: float
    text: str
    metadata: dict


class GTSearch:
    def __init__(self) -> None:
        self.col = get_collection()

    def _where(
        self,
        block: int | None,
        type_: str | None,
        trading_style: str | None,
    ) -> dict | None:
        clauses = []
        if block is not None:
            clauses.append({"block": block})
        if type_:
            clauses.append({"type": type_})
        if trading_style:
            if trading_style in (SWING_STYLE, INTRADAY_STYLE):
                clauses.append({
                    "trading_style": {"$in": [trading_style, COMBINED_STYLE]}
                })
            else:
                clauses.append({"trading_style": trading_style})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def search(
        self,
        query: str,
        top_k: int = 5,
        block: int | None = None,
        type_: str | None = None,
        trading_style: str | None = None,
    ) -> list[Hit]:
        n = max(top_k * CANDIDATE_MULTIPLIER, 20)
        res = self.col.query(
            query_texts=[query],
            n_results=min(n, max(self.col.count(), 1)),
            where=self._where(block, type_, trading_style),
            include=["documents", "metadatas", "distances"],
        )
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        hits: list[Hit] = []
        q = query.lower()
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            # cosine distance -> similarity in [0..1]
            vec_score = max(0.0, 1.0 - dist)
            # fuzzy: best of match against title/section and against body
            label = f"{meta.get('title', '')} {meta.get('section', '')}".lower()
            fuzzy_label = fuzz.token_set_ratio(q, label) / 100.0
            fuzzy_body = fuzz.partial_ratio(q, doc.lower()) / 100.0
            fuzzy_score = max(fuzzy_label, fuzzy_body)
            score = VECTOR_WEIGHT * vec_score + FUZZY_WEIGHT * fuzzy_score
            hits.append(Hit(chunk_id=cid, score=round(score, 4), text=doc, metadata=meta))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
