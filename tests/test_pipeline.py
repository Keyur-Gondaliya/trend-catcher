"""Smoke + behaviour tests for the trend pipeline.

These run fully offline (tfidf embeddings, template summaries, no API keys) and
exercise the two things that matter most: that real trends surface, and that
feedback actually moves the ranking.
"""
import numpy as np

from trend_radar.embeddings import get_embedder
from trend_radar.trends import detect_trends
from trend_radar.preferences import PreferenceStore
from trend_radar.digest import build_digest
from trend_radar.sample_data import main as generate_sample


def _load_sample(tmp_path):
    csv_path = tmp_path / "tweets.csv"
    generate_sample(path=str(csv_path))
    from trend_radar.ingest import load_tweets
    return load_tweets(str(csv_path))


def test_detect_trends_finds_clusters(tmp_path):
    tweets = _load_sample(tmp_path)
    vectors = get_embedder().fit_transform([t.text for t in tweets])
    trends = detect_trends(tweets, vectors)

    assert trends, "expected at least one trend from the sample data"
    for t in trends:
        # every trend honours the >=3 tweets / >=2 distinct voices rule
        assert len(t.tweet_ids) >= 3
        assert len(t.authors) >= 2
        assert 0.0 <= t.strength <= 1.0


def test_feedback_moves_ranking(tmp_path):
    tweets = _load_sample(tmp_path)
    vectors = get_embedder().fit_transform([t.text for t in tweets])
    trends = detect_trends(tweets, vectors)

    # Isolate preference state to a temp file.
    pref_path = tmp_path / "preferences.json"
    prefs = PreferenceStore(path=str(pref_path))
    assert prefs.is_cold

    # Cold-start ranking: relevance is neutral for everyone.
    cold = build_digest(trends, prefs)
    assert all(it.relevance == 0.5 for it in cold)

    # Mark the lowest-ranked cold item as the one we love, then re-rank.
    target = cold[-1]
    target_centroid = next(t.centroid for t in trends if t.cluster_id == target.cluster_id)
    prefs.update(np.asarray(target_centroid, dtype=np.float32), relevant=True)

    warm = build_digest(trends, prefs)
    moved = next(it for it in warm if it.cluster_id == target.cluster_id)
    assert moved.relevance > 0.5
    # The thing we upvoted should now outrank where it started.
    assert warm[0].cluster_id == target.cluster_id or moved.score > target.score


def test_default_interest_prior_personalizes_cold_start(tmp_path):
    tweets = _load_sample(tmp_path)
    embedder = get_embedder()
    vectors = embedder.fit_transform([t.text for t in tweets])
    trends = detect_trends(tweets, vectors)

    prefs = PreferenceStore(path=str(tmp_path / "preferences.json"))
    prior = embedder.transform(
        ["AI code review agents", "context engineering for LLMs"]
    ).mean(axis=0)
    prefs.seed_prior(prior)

    # A prior is a starting point, not earned taste: still "cold", but flagged.
    assert prefs.is_cold
    assert prefs.prior

    items = build_digest(trends, prefs)
    # The prior must actually move relevance off the neutral 0.5 baseline.
    assert any(it.relevance != 0.5 for it in items)
    assert items[0].relevance > 0.5

    # Real feedback clears the prior flag and is no longer "cold".
    prefs.update(np.asarray(prior, dtype=np.float32), relevant=True)
    assert not prefs.prior
    assert not prefs.is_cold


def test_whatsapp_webhook_parses_and_applies(monkeypatch):
    """The inbound webhook must route a WhatsApp reply through the same
    feedback path as the CLI. Skips if the whatsapp extra (Flask) isn't here."""
    import pytest
    pytest.importorskip("flask")

    import trend_radar.webhook as webhook

    applied = {}
    monkeypatch.setattr(webhook, "apply_feedback", lambda fb: applied.update(fb))

    client = webhook._build_app().test_client()

    # A well-formed reply is parsed and applied.
    resp = client.post("/whatsapp", data={"Body": "2 yes, 4 no"})
    assert resp.status_code == 200
    assert applied == {2: True, 4: False}
    assert b"folded 2 signal" in resp.data.lower()

    # Gibberish is rejected without touching the preference store.
    applied.clear()
    resp = client.post("/whatsapp", data={"Body": "hi there"})
    assert applied == {}
    assert b"didn't catch" in resp.data.lower()
