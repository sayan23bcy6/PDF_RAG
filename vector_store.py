"""
Pure-Python vector store backed by numpy cosine similarity.

Replaces ChromaDB which crashes on Windows (exit code 1) due to its
C++/HNSW layer calling os._exit() when it encounters certain errors.
This implementation has zero C++ dependencies - just numpy and pickle.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Optional, Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.embeddings import Embeddings


class SimpleVectorStore:
    """
    Lightweight, pure-Python vector store.

    - Embeds documents using any LangChain-compatible Embeddings object.
    - Persists documents + vectors to a single pickle file.
    - Retrieval: cosine similarity via numpy (no C extensions required).
    """

    def __init__(self, embedding_function: Embeddings, persist_path: str):
        self.embedding_function = embedding_function
        self.persist_path = persist_path
        self.documents: List[Document] = []
        self._embeddings: List[List[float]] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self):
        try:
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data.get("documents", [])
                self._embeddings = data.get("embeddings", [])
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            pass  # fresh store

    def _save(self):
        Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = self.persist_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(
                {"documents": self.documents, "embeddings": self._embeddings},
                f,
            )
        if os.path.exists(self.persist_path):
            os.remove(self.persist_path)
        os.rename(tmp, self.persist_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def add_documents(self, documents: List[Document]) -> None:
        texts = [doc.page_content for doc in documents]
        new_vecs = self.embedding_function.embed_documents(texts)
        self.documents.extend(documents)
        self._embeddings.extend(new_vecs)
        self._save()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if not self._embeddings:
            return []

        q_vec = np.array(self.embedding_function.embed_query(query), dtype=np.float32)
        mat = np.array(self._embeddings, dtype=np.float32)

        # Cosine similarity: (mat @ q_vec) / (||mat|| * ||q_vec||)
        q_norm = np.linalg.norm(q_vec) + 1e-10
        row_norms = np.linalg.norm(mat, axis=1) + 1e-10
        scores = mat @ q_vec / (row_norms * q_norm)

        top_k = min(k, len(scores))
        idx = np.argpartition(scores, -top_k)[-top_k:]
        idx = idx[np.argsort(scores[idx])[::-1]]

        return [self.documents[i] for i in idx]

    def __len__(self) -> int:
        return len(self.documents)

    # ------------------------------------------------------------------
    # LangChain retriever
    # ------------------------------------------------------------------
    def as_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[dict] = None,
    ) -> "SimpleRetriever":
        k = (search_kwargs or {}).get("k", 5)
        return SimpleRetriever(store=self, k=k)


class SimpleRetriever(BaseRetriever):
    """LangChain BaseRetriever wrapping SimpleVectorStore."""

    store: Any  # SimpleVectorStore
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        return self.store.similarity_search(query, k=self.k)
