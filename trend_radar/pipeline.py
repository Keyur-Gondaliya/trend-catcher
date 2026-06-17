"""
Pipeline: wires the whole diagram together for one scheduled run.

    dataset -> ingest -> embed -> store -> detect trends
            -> rank against preferences -> digest -> deliver

It also persists each digest item's centroid so that when feedback comes back
("item 2 was relevant"), we know exactly which vector to fold into the
preference store. That persisted mapping is the bridge between the outgoing
digest and the incoming feedback -- the loop in the diagram.
"""
import json
import os
import numpy as np

from trend_radar import config
from trend_radar.ingest import load_tweets
from trend_radar.embeddings import get_embedder
from trend_radar.store import VectorStore
from trend_radar.trends import detect_trends
from trend_radar.preferences import PreferenceStore
from trend_radar.digest import build_digest, render_text
from trend_radar.notify import deliver


def run_digest():
    tweets = load_tweets(config.DATA_PATH)
    embedder = get_embedder()
    vectors = embedder.fit_transform([t.text for t in tweets])

    store = VectorStore()
    store.add([t.id for t in tweets], vectors)  # parity with a real vector DB

    trends = detect_trends(tweets, vectors)

    prefs = PreferenceStore()
    # Cold start: seed the preference centroid from configured default interests
    # so the first digest already leans toward what I care about. Embedded with
    # the same (now-fitted) embedder, so it shares the trend centroids' space.
    if prefs.is_cold and config.DEFAULT_INTERESTS:
        prior = embedder.transform(config.DEFAULT_INTERESTS).mean(axis=0)
        prefs.seed_prior(prior)
        prefs.save()  # persist so feedback on this digest builds on the prior

    items = build_digest(trends, prefs)

    # Persist item_id -> centroid so feedback can update the right vector.
    id_to_centroid = {}
    for it in items:
        c = next(t.centroid for t in trends if t.cluster_id == it.cluster_id)
        id_to_centroid[it.item_id] = c.tolist()
    os.makedirs(os.path.dirname(config.DIGEST_PATH) or ".", exist_ok=True)
    with open(config.DIGEST_PATH, "w") as f:
        json.dump({"centroids": id_to_centroid}, f)

    text = render_text(items, cold=prefs.is_cold, using_prior=prefs.prior)
    deliver(text)
    return items


def apply_feedback(feedback: dict[int, bool]):
    """feedback: {item_id: True(relevant)/False(not)}.
    Reads the last digest's stored centroids and updates the preference store."""
    with open(config.DIGEST_PATH) as f:
        centroids = json.load(f)["centroids"]
    prefs = PreferenceStore()
    for item_id, relevant in feedback.items():
        c = centroids.get(str(item_id))
        if c is None:
            continue
        prefs.update(np.array(c, dtype=np.float32), relevant)
    prefs.save()
    print(f"Folded {len(feedback)} signals. "
          f"Preference store now has {prefs.count} relevant examples "
          f"(cold={prefs.is_cold}).")
