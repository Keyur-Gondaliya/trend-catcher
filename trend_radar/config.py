"""
Central configuration for Trend Radar.

Everything that might change between "runs on my laptop with no keys" and
"runs in production with real APIs" lives here. The providers are pluggable
on purpose -- it's one of the design decisions worth explaining on camera.
"""
import os

# ---- Embedding backend ----------------------------------------------------
# "tfidf"                 -> offline, zero downloads, instant. Good for a demo.
# "sentence-transformers" -> better semantics, downloads a model (~80MB).
# "openai"                -> best semantics, needs OPENAI_API_KEY.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "tfidf")
ST_MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_EMBED_MODEL = "text-embedding-3-small"

# ---- Digest summarizer backend -------------------------------------------
# "template" -> no LLM, builds a headline from the top tweet + key terms.
# "anthropic" -> needs ANTHROPIC_API_KEY.
# "openai"    -> needs OPENAI_API_KEY.
SUMMARIZER_BACKEND = os.getenv("SUMMARIZER_BACKEND", "template")

# ---- Trend engine knobs ---------------------------------------------------
# Two tweets join the same cluster if their cosine distance is below this.
# Lower = tighter, more clusters. Tune per embedding backend.
CLUSTER_DISTANCE_THRESHOLD = float(os.getenv("CLUSTER_DISTANCE_THRESHOLD", "0.45"))
MIN_CLUSTER_SIZE = int(os.getenv("MIN_CLUSTER_SIZE", "3"))

# Number of time buckets we slice the dataset into to estimate "velocity".
# Static dataset has no live feed, so velocity is derived from each tweet's
# own timestamp -- see trends.py.
TIME_BUCKETS = int(os.getenv("TIME_BUCKETS", "6"))

# Final score = w_trend * trend_strength + w_pref * personal_relevance
WEIGHT_TREND = float(os.getenv("WEIGHT_TREND", "0.6"))
WEIGHT_PREF = float(os.getenv("WEIGHT_PREF", "0.4"))

TOP_N_TRENDS = int(os.getenv("TOP_N_TRENDS", "5"))

# ---- Default interests (cold-start prior) ---------------------------------
# Before you've given ANY feedback, the preference centroid is seeded from
# these phrases, so the very first digest already leans toward what you care
# about instead of ranking purely by trend strength. Feedback then folds into
# this prior and gradually overrides it -- the prior is just the starting point.
#
# Semicolon-separated phrases. Short topic phrases embed better than a long
# instruction sentence. Set DEFAULT_INTERESTS="" to disable (pure cold start).
DEFAULT_INTERESTS = [
    s.strip() for s in os.getenv(
        "DEFAULT_INTERESTS",
        "AI agents; LLM applications; context engineering for LLMs; "
        "AI code review agents; retrieval augmented generation; "
        "prompt engineering; vector databases"
    ).split(";") if s.strip()
]

# ---- Delivery -------------------------------------------------------------
# "console" -> prints the digest (default, no creds).
# "whatsapp" -> sends via Twilio, needs TWILIO_* env vars + a number.
DELIVERY_BACKEND = os.getenv("DELIVERY_BACKEND", "console")

# ---- Paths ----------------------------------------------------------------
DATA_PATH = os.getenv("DATA_PATH", "data/tweets.csv")
PREFERENCE_PATH = os.getenv("PREFERENCE_PATH", "data/preferences.json")
DIGEST_PATH = os.getenv("DIGEST_PATH", "data/last_digest.json")
