"""Just-enough RAG: turn a text 'book' into a searchable vector store.

Embeddings are LOCAL and FREE (fastembed — a small ONNX model, downloaded once,
no API key), so this runs offline. Groq has no embeddings API, so the retrieval
layer lives entirely here; Groq is only the LLM (see agent.py).

LangChain's FAISS vector-store wrapper handles embedding conversion, indexing,
and similarity search while this module keeps the local model and chunking
choices explicit.
"""

from __future__ import annotations

import functools

from fastembed import TextEmbedding
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

MODEL_NAME = "BAAI/bge-small-en-v1.5"


@functools.lru_cache(maxsize=1)
def _embedder() -> TextEmbedding:
    """The local embedding model. Cached so we build it once per process."""
    return TextEmbedding(model_name=MODEL_NAME)


class _BookEmbeddings(Embeddings):
    """Adapt the existing local FastEmbed model to LangChain's interface."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in _embedder().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(_embedder().embed([text])).tolist()


class BookRetriever:
    """A top-k retriever backed by LangChain's FAISS vector-store wrapper."""

    def __init__(self, chunks: list[str], k: int = 4):
        self.chunks = chunks
        self.k = k
        self.store = FAISS.from_texts(
            texts=chunks,
            embedding=_BookEmbeddings(),
            normalize_L2=True,  # Unit vectors make L2 ranking equivalent to cosine similarity.
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )

    def similarity_search(self, query: str) -> list[Document]:
        """Return the k most similar chunks, including their cosine score."""
        results = self.store.similarity_search_with_score(
            query, k=min(self.k, len(self.chunks))
        )
        documents = []
        for document, distance in results:
            document.metadata["score"] = 1.0 - float(distance) / 2.0
            documents.append(document)
        return documents

    def invoke(self, query: str) -> list[Document]:
        """Compatibility method for existing retriever callers."""
        return self.similarity_search(query)


def _chunk(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    return splitter.split_text(text)


def build_retriever(source_path: str, k: int = 4) -> BookRetriever:
    """Read a text file, chunk it, embed the chunks, and return a top-k retriever."""
    with open(source_path, "r", encoding="utf-8") as f:
        return BookRetriever(_chunk(f.read()), k=k)


def retriever_from_text(text: str, k: int = 4) -> BookRetriever:
    """Same, but from an in-memory string (used by the upload flow in the UI)."""
    return BookRetriever(_chunk(text), k=k)
