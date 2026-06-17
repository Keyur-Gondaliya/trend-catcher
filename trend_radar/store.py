"""
A tiny, transparent vector store.

In production this is Pinecone / Weaviate / pgvector. For the prototype a numpy
cosine store keeps the similarity math visible -- which is better for a demo,
because you can actually show how "related" is computed rather than hiding it
behind an API. The interface (add / search) is the same shape a real vector DB
exposes, so swapping it out later is contained to this file.
"""
import numpy as np


def l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class VectorStore:
    def __init__(self):
        self.vectors = None       # (N, D) normalized
        self.ids = []             # parallel list of tweet ids

    def add(self, ids: list[int], vectors: np.ndarray):
        v = l2_normalize(vectors.astype(np.float32))
        self.vectors = v if self.vectors is None else np.vstack([self.vectors, v])
        self.ids.extend(ids)

    def search(self, query: np.ndarray, k: int = 10):
        """Return [(id, cosine_similarity), ...] for the top-k matches."""
        q = l2_normalize(query.reshape(1, -1).astype(np.float32))
        sims = (self.vectors @ q.T).ravel()
        top = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in top]

    def similarity_matrix(self) -> np.ndarray:
        """Full pairwise cosine matrix -- used by the clustering step."""
        return self.vectors @ self.vectors.T
