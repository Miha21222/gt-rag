"""ChromaDB store with a local multilingual fastembed embedding function."""
from __future__ import annotations

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

from .common import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL


class MultilingualEmbedding(EmbeddingFunction):
    """fastembed ONNX embeddings; handles Russian + English trading terms."""

    def __init__(self) -> None:
        import warnings

        from fastembed import TextEmbedding

        with warnings.catch_warnings():
            # fastembed warns about a pooling change vs older versions; the
            # index and queries use the same model instance, so it's moot.
            warnings.simplefilter("ignore", UserWarning)
            self._model = TextEmbedding(model_name=EMBED_MODEL)

    def __call__(self, input: Documents) -> Embeddings:
        return [e.tolist() for e in self._model.embed(list(input))]

    # Chroma persists the EF name and validates it on reopen.
    @staticmethod
    def name() -> str:
        return "gt-multilingual-minilm"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "MultilingualEmbedding":
        return MultilingualEmbedding()


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=MultilingualEmbedding(),
        metadata={"hnsw:space": "cosine"},
    )
