"""
Personalization via a small set of embedding centroids.

Taste is rarely one thing. A single averaged vector blurs distinct interests --
if you care about both infra *and* ML, their average sits in a no-man's-land
that matches neither well. So the preference store keeps up to N centroids, one
per interest "mode", and scores a trend against its *best-matching* mode.

Learning is online clustering: a "relevant" signal merges into the nearest mode
if it's close enough (cosine >= a threshold), otherwise it starts a new mode
(until the cap). "Not relevant" nudges the nearest mode away. New, never-seen
topics still rank correctly if they sit near any mode in embedding space -- tags
can't generalize like that.

Cold start: with no feedback, the modes are seeded from a configured interest
prompt (see config.DEFAULT_INTEREST_PROMPT), so the first digest already leans
your way. Feedback then folds in and gradually overrides the prior.
"""
import json
import os
import numpy as np

from trend_radar import config


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) or 1.0)


class PreferenceStore:
    def __init__(self, path=config.PREFERENCE_PATH,
                 max_centroids: int = None, merge_threshold: float = None):
        self.path = path
        self.max_centroids = (config.MAX_PREFERENCE_CENTROIDS
                              if max_centroids is None else max_centroids)
        self.merge_threshold = (config.PREFERENCE_MERGE_THRESHOLD
                                if merge_threshold is None else merge_threshold)
        self.centroids = []      # list[np.ndarray] -- one per interest mode
        self.counts = []         # parallel positive-signal counts
        self.prior = False       # True while modes are only the seeded prior
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            d = json.load(f)
        if d.get("centroids") is not None:
            self.centroids = [np.array(c, dtype=np.float32)
                              for c in d["centroids"]]
            self.counts = list(d.get("counts", [0] * len(self.centroids)))
        else:  # back-compat with the old single-centroid format
            c = d.get("centroid")
            self.centroids = [np.array(c, dtype=np.float32)] if c else []
            self.counts = [d.get("count", 0)] if c else []
        self.prior = d.get("prior", False)

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "centroids": [c.tolist() for c in self.centroids],
                "counts": self.counts,
                "prior": self.prior,
            }, f)

    @property
    def count(self) -> int:
        """Total relevant signals folded across all modes."""
        return int(sum(self.counts))

    @property
    def is_cold(self) -> bool:
        """No earned feedback yet. A seeded interest prior still counts as cold
        -- it's a sensible starting point, not learned taste."""
        return self.count == 0

    def seed_prior(self, vec: np.ndarray):
        """Initialize a single mode from the configured interest prompt.

        Only applies while the store is untouched by feedback (fresh, or still
        holding a previously-seeded prior). The prior is re-derived each run by
        the caller from the *current* embedding space, so it always matches the
        trend centroids it's compared against. Feedback then folds in and
        gradually overrides it."""
        if vec is None or (self.centroids and not self.prior):
            return
        self.centroids = [np.asarray(vec, dtype=np.float32)]
        self.counts = [0]
        self.prior = True

    def _best_match(self, vec: np.ndarray):
        """(index, cosine) of the closest same-dimension mode, or (None, -1)."""
        a = _unit(vec)
        best_i, best_s = None, -1.0
        for i, c in enumerate(self.centroids):
            if c.shape != vec.shape:
                continue
            s = float(np.dot(a, _unit(c)))
            if s > best_s:
                best_i, best_s = i, s
        return best_i, best_s

    def relevance(self, vec: np.ndarray) -> float:
        """Best-match cosine similarity of a trend centroid to any preference
        mode, mapped to 0..1. Neutral 0.5 when there are no usable modes (no
        feedback and no configured prior)."""
        if not self.centroids:
            return 0.5
        _, best = self._best_match(vec)
        if best < -1.0 + 1e-9:  # no same-dim mode (e.g. backend changed)
            return 0.5
        return float((best + 1) / 2)

    def update(self, vec: np.ndarray, relevant: bool):
        """Fold a feedback signal into the modes (online clustering).

        Relevant -> merge into the nearest mode if close enough (EMA), else open
        a new mode until the cap, then fold into the nearest. Not relevant ->
        nudge the nearest mode away. EMA weights recent taste a little more, so
        interests can drift over time."""
        vec = np.asarray(vec, dtype=np.float32)
        self.prior = False  # any real feedback supersedes the seeded prior
        alpha = 0.3
        i, sim = self._best_match(vec)

        if not relevant:
            if i is not None:
                self.centroids[i] = self.centroids[i] - 0.15 * vec
            elif len(self.centroids) < self.max_centroids:
                # nothing to push away from yet; remember it as a repulsor
                self.centroids.append(-0.2 * vec)
                self.counts.append(0)
            return

        if i is not None and sim >= self.merge_threshold:
            self.centroids[i] = (1 - alpha) * self.centroids[i] + alpha * vec
            self.counts[i] += 1
        elif len(self.centroids) < self.max_centroids:
            self.centroids.append(vec.copy())
            self.counts.append(1)
        elif i is not None:  # at capacity: reinforce the nearest mode
            self.centroids[i] = (1 - alpha) * self.centroids[i] + alpha * vec
            self.counts[i] += 1
        else:  # at capacity, no usable mode: replace the weakest
            j = int(np.argmin(self.counts))
            self.centroids[j] = vec.copy()
            self.counts[j] = 1
