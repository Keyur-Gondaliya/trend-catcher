"""
Personalization via an embedding centroid.

The preference store holds a single vector: the running centroid of everything
you've marked "relevant", minus a pull from what you've marked "not for me".
Ranking a trend = cosine similarity between its centroid and yours.

Why a centroid rather than topic tags: it generalizes. If you upvote a few
posts about agents and context engineering, a brand-new cluster about "LLM
memory" scores high *without* you ever having tagged it, because it sits near
your centroid in embedding space. Tags can't do that.

The honest failure mode -- and you should say this on camera -- is COLD START:
with no feedback yet, the centroid is empty, so the system can only rank by raw
trend strength. Personalization is earned over a few digests, not immediate.
"""
import json
import os
import numpy as np

from trend_radar import config


class PreferenceStore:
    def __init__(self, path=config.PREFERENCE_PATH):
        self.path = path
        self.centroid = None     # np.ndarray or None
        self.count = 0           # how many relevant signals folded in
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                d = json.load(f)
            self.centroid = (np.array(d["centroid"], dtype=np.float32)
                             if d.get("centroid") else None)
            self.count = d.get("count", 0)

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "centroid": None if self.centroid is None
                            else self.centroid.tolist(),
                "count": self.count,
            }, f)

    @property
    def is_cold(self) -> bool:
        return self.centroid is None

    def relevance(self, vec: np.ndarray) -> float:
        """Cosine similarity of a trend centroid to the preference centroid,
        mapped to 0..1. Neutral 0.5 during cold start."""
        if self.is_cold:
            return 0.5
        a = vec / (np.linalg.norm(vec) or 1.0)
        b = self.centroid / (np.linalg.norm(self.centroid) or 1.0)
        return float((np.dot(a, b) + 1) / 2)

    def update(self, vec: np.ndarray, relevant: bool):
        """Fold a feedback signal into the centroid.

        Relevant -> exponential moving average pulls the centroid toward it.
        Not relevant -> nudge the centroid away. EMA keeps recent taste weighted
        a bit more, so your interests can drift over time (which you said they
        would)."""
        vec = vec.astype(np.float32)
        if self.centroid is None:
            self.centroid = vec.copy() if relevant else -0.2 * vec
            self.count = 1 if relevant else 0
            return
        # align dimensions defensively (embedding backend changes break this)
        if vec.shape != self.centroid.shape:
            return
        alpha = 0.3
        if relevant:
            self.centroid = (1 - alpha) * self.centroid + alpha * vec
            self.count += 1
        else:
            self.centroid = self.centroid - 0.15 * vec
