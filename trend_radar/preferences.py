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
        self.prior = False       # True while the centroid is only the seeded
                                 # default-interest prior (no feedback yet)
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                d = json.load(f)
            self.centroid = (np.array(d["centroid"], dtype=np.float32)
                             if d.get("centroid") else None)
            self.count = d.get("count", 0)
            self.prior = d.get("prior", False)

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "centroid": None if self.centroid is None
                            else self.centroid.tolist(),
                "count": self.count,
                "prior": self.prior,
            }, f)

    @property
    def is_cold(self) -> bool:
        """No earned feedback yet. A seeded default-interest prior still counts
        as cold -- it's a sensible starting point, not learned taste."""
        return self.count == 0

    def seed_prior(self, vec: np.ndarray):
        """Initialize the centroid from configured default interests.

        Only applies while the store is untouched by feedback (a fresh store,
        or one still holding a previously-seeded prior). The prior is re-derived
        each run by the caller from the *current* embedding space, so it always
        matches the trend centroids it's compared against. The first feedback
        signal then folds into this prior and gradually overrides it."""
        if vec is None or not (self.prior or self.centroid is None):
            return
        self.centroid = np.asarray(vec, dtype=np.float32)
        self.prior = True

    def relevance(self, vec: np.ndarray) -> float:
        """Cosine similarity of a trend centroid to the preference centroid,
        mapped to 0..1. Neutral 0.5 only when there's no centroid at all
        (no feedback and no configured prior)."""
        if self.centroid is None:
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
            self.prior = False
            return
        # align dimensions defensively (embedding backend changes break this)
        if vec.shape != self.centroid.shape:
            return
        # real feedback now folds in -- this is no longer just the seeded prior
        self.prior = False
        alpha = 0.3
        if relevant:
            self.centroid = (1 - alpha) * self.centroid + alpha * vec
            self.count += 1
        else:
            self.centroid = self.centroid - 0.15 * vec
