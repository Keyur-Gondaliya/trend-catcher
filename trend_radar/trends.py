"""
Trend engine -- the heart of the system.

Two ideas combine here:

1. CLUSTERING. Tweets that are semantically close get grouped. A "trend" is a
   cluster that several *different people* are talking about. We use
   agglomerative clustering on cosine distance with a distance threshold, so we
   never have to guess the number of clusters up front.

2. VELOCITY. A real trend isn't just popular, it's *accelerating*. The dataset
   is static (a frozen dump), so there's no live feed to watch. Instead we use
   each tweet's own timestamp: slice the time range into buckets and measure how
   a cluster's engagement grows across buckets. A rising slope = heating up.
   This is the answer to the obvious reviewer question, "how do you detect a
   trend from static data?"
"""
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from trend_radar import config
from trend_radar.ingest import Tweet


@dataclass
class Trend:
    cluster_id: int
    tweet_ids: list[int]
    texts: list[str]
    authors: set
    centroid: np.ndarray
    total_engagement: int
    velocity: float          # engagement growth across time buckets
    strength: float          # combined, normalized 0..1
    top_tweet: str


def _cluster(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) == 1:
        return np.array([0])
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=config.CLUSTER_DISTANCE_THRESHOLD,
        metric="cosine",
        linkage="average",
    )
    return model.fit_predict(vectors)


def _velocity(engagements: list[int], times: list[float], t_min, t_max) -> float:
    """Slope of engagement-per-bucket over time, normalized.

    Positive = the conversation is growing toward the recent end of the window.
    A flat or front-loaded cluster scores near zero or negative.
    """
    if t_max == t_min:
        return 0.0
    buckets = np.zeros(config.TIME_BUCKETS)
    span = t_max - t_min
    for eng, t in zip(engagements, times):
        idx = min(int((t - t_min) / span * config.TIME_BUCKETS),
                  config.TIME_BUCKETS - 1)
        buckets[idx] += eng
    x = np.arange(config.TIME_BUCKETS)
    # least-squares slope; normalize by mean engagement so big topics don't
    # automatically dominate purely by size
    slope = np.polyfit(x, buckets, 1)[0]
    denom = buckets.mean() if buckets.mean() > 0 else 1.0
    return float(slope / denom)


def detect_trends(tweets: list[Tweet], vectors: np.ndarray) -> list[Trend]:
    labels = _cluster(vectors)
    t_all = [t.created_at.timestamp() for t in tweets]
    t_min, t_max = min(t_all), max(t_all)

    trends = []
    for cid in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == cid]
        authors = {tweets[i].author for i in idx}

        # A trend needs several posts AND several distinct voices. One person
        # tweeting 5 times is not a trend.
        if len(idx) < config.MIN_CLUSTER_SIZE or len(authors) < 2:
            continue

        engs = [tweets[i].engagement for i in idx]
        times = [t_all[i] for i in idx]
        centroid = vectors[idx].mean(axis=0)
        top_i = max(idx, key=lambda i: tweets[i].engagement)

        trends.append(Trend(
            cluster_id=int(cid),
            tweet_ids=[tweets[i].id for i in idx],
            texts=[tweets[i].text for i in idx],
            authors=authors,
            centroid=centroid,
            total_engagement=sum(engs),
            velocity=_velocity(engs, times, t_min, t_max),
            strength=0.0,  # filled in below
            top_tweet=tweets[top_i].text,
        ))

    # Normalize engagement + velocity to 0..1, then combine into strength.
    if trends:
        max_eng = max(t.total_engagement for t in trends) or 1
        vels = [t.velocity for t in trends]
        vmin, vmax = min(vels), max(vels)
        vrange = (vmax - vmin) or 1.0
        for t in trends:
            eng_n = t.total_engagement / max_eng
            vel_n = (t.velocity - vmin) / vrange
            # weight velocity a touch higher: we care about *emerging*, not
            # just *big*. Established-but-flat topics are things you already know.
            t.strength = 0.45 * eng_n + 0.55 * vel_n

    return sorted(trends, key=lambda t: t.strength, reverse=True)
