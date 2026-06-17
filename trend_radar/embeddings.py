"""
Embedding layer with a pluggable backend.

Why pluggable: the demo needs to run with zero keys and zero downloads, but the
"real" version should use proper semantic embeddings. Same interface, swap the
backend via EMBEDDING_BACKEND. This is a decision worth defending on camera:
the rest of the system only depends on "give me vectors", not on any vendor.
"""
import numpy as np
from trend_radar import config


class Embedder:
    def fit_transform(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def transform(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class TfidfEmbedder(Embedder):
    """Offline, no downloads. Weaker semantics but fully deterministic.
    Good enough to demonstrate the pipeline end-to-end."""
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                   min_df=1, sublinear_tf=True)

    def fit_transform(self, texts):
        return self._dense(self.vec.fit_transform(texts))

    def transform(self, texts):
        return self._dense(self.vec.transform(texts))

    @staticmethod
    def _dense(m):
        return np.asarray(m.todense(), dtype=np.float32)


class SentenceTransformerEmbedder(Embedder):
    """Proper semantic embeddings. Downloads a small model once."""
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(config.ST_MODEL_NAME)

    def fit_transform(self, texts):
        return self.transform(texts)

    def transform(self, texts):
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True),
            dtype=np.float32,
        )


class OpenAIEmbedder(Embedder):
    """Best semantics. Needs OPENAI_API_KEY."""
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()

    def fit_transform(self, texts):
        return self.transform(texts)

    def transform(self, texts):
        resp = self.client.embeddings.create(
            model=config.OPENAI_EMBED_MODEL, input=texts)
        return np.asarray([d.embedding for d in resp.data], dtype=np.float32)


def get_embedder() -> Embedder:
    backend = config.EMBEDDING_BACKEND
    if backend == "tfidf":
        return TfidfEmbedder()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder()
    if backend == "openai":
        return OpenAIEmbedder()
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend}")
