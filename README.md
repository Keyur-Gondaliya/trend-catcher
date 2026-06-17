# Trend Radar

A personal tech-trend radar. It reads a stream of tech tweets, finds the topics
that are actually *gaining traction* (not just loud), filters them against what
**I** care about, and sends me a twice-weekly digest on WhatsApp. I reply with a
quick "this was useful / this wasn't" and it gets better at picking for me.

> **The problem (mine, real):** Keeping up with tech moves through Twitter/X, but
> I can't follow everyone, and the signal-to-noise is terrible. I lose ~20 min a
> day skimming to figure out what's genuinely emerging vs. recycled noise. I want
> a system that does that triage for me and learns my taste over time.

This is the AI-Engineer-track build for the *Build At Damco* challenge.

---

## Why this is a system, not a script

It has distinct layers, each swappable, with a closed feedback loop:

```
X dataset ──▶ ingest ──▶ embed ──▶ vector store
                                       │
                          ┌────────────┘
                          ▼
              cluster (semantic similarity)
                          ▼
              velocity (engagement over time)  ──▶  trend strength
                          ▼
              rank against preference centroid  ◀── preference store
                          ▼
                 digest (per-item)
                          ▼
                 WhatsApp delivery
                          ▼
               feedback reply  ───────────────▶  update preference centroid
```

A trend is a **cluster of semantically similar tweets, from several different
people, whose engagement is accelerating.** All three conditions matter: the
"several people" rule stops one loud account from faking a trend, and the
acceleration rule is what separates *emerging* from *already-big-and-flat*.

---

## Key design decisions (the things to ask me about)

**Embedding centroid for preferences, not topic tags.** The preference store is
a single vector — the running average of what I marked relevant. New, never-seen
topics get ranked correctly if they sit near my centroid in embedding space.
Tags can't generalize like that. (`trend_radar/preferences.py`)

**Velocity from timestamps, on a static dataset.** I don't have X API access, so
there's no live feed to watch grow. Instead each tweet carries its own
timestamp; I bucket the time window and measure the slope of a cluster's
engagement across buckets. Rising slope = heating up. This is how you detect a
*trend* from a frozen dump. (`trend_radar/trends.py`)

**Per-item feedback, not per-digest.** Each digest item has a stable id; I reply
`2 yes, 4 no`. Each reply maps to exactly one cluster centroid, so every digest
is several labeled training signals instead of one vague thumbs-up. (`trend_radar/digest.py`, `trend_radar/pipeline.py`)

**Everything is pluggable.** Embeddings (`tfidf` offline / `sentence-transformers`
/ `openai`), summaries (`template` / `anthropic` / `openai`), delivery
(`console` / `whatsapp`). The pipeline depends on interfaces, not vendors, so it
runs with zero keys for review and upgrades by flipping an env var.

---

## Run it (no API keys needed)

```bash
pip install -r requirements.txt
python run.py seed       # generate a sample X dataset
python run.py digest     # first run: cold start, ranked by trend strength
python run.py feedback "2 yes, 3 yes, 1 no, 5 no"
python run.py digest     # now personalized — watch the ranking move
```

Upgrade quality when you have keys:

```bash
EMBEDDING_BACKEND=sentence-transformers SUMMARIZER_BACKEND=anthropic python run.py digest
```

Send to real WhatsApp (Twilio):

```bash
DELIVERY_BACKEND=whatsapp TWILIO_SID=... TWILIO_TOKEN=... \
TWILIO_WHATSAPP_FROM=+14155238886 MY_WHATSAPP_NUMBER=+91... python run.py digest
```

The twice-weekly schedule (Wed + Sun) is just cron calling `python run.py digest`:
```
0 9 * * 3,0  cd /path/to/trend-radar && python run.py digest
```

---

## Run it with Docker

The app is a scheduled task runner, so the container's entrypoint is `python
run.py` and you pass the subcommand. `./data` is mounted as a volume, so the
dataset and learned preferences persist across runs.

```bash
docker compose build
docker compose run --rm trend-radar seed                       # generate dataset
docker compose run --rm trend-radar digest                     # cold start
docker compose run --rm trend-radar feedback "2 yes, 3 no"     # tune
docker compose run --rm trend-radar digest                     # personalized
```

Switch backends / add secrets by copying `.env.example` to `.env` (read
automatically by compose). For the cron job, point it at the container:
```
0 9 * * 3,0  cd /path/to/trend-radar && docker compose run --rm trend-radar digest
```
Note: the image ships the offline core deps only. For `sentence-transformers`,
`anthropic`/`openai`, or `twilio` backends, add them to the Dockerfile's install
step (they map to the optional extras in `pyproject.toml`).

---

## What's broken / what I'd do with more time (honest)

- **Cold start.** Mitigated: `DEFAULT_INTEREST_PROMPT` in `config.py` is a plain
  sentence describing what I follow ("…AI agents, context engineering, AI code
  review…"). It's embedded and seeds the preference centroid before any feedback,
  so the *first* digest already leans my way. Feedback then folds in and overrides
  the prior over a few cycles. Set it empty for pure trend-strength ranking.
- **TF-IDF clusters by words, not meaning.** The offline default groups "stop
  prompt hacking" and "context engineering" as near-but-separate clusters that a
  real embedding model would merge. Switching `EMBEDDING_BACKEND` fixes this;
  the demo just shows the weak case honestly.
- **False-positive trends.** A burst of similar-looking but unrelated tweets can
  cluster together and look like a trend. The "≥2 distinct authors" rule helps
  but isn't bulletproof; a real fix is engagement-source diversity scoring.
- **False negatives on early signals.** A genuinely new trend with only 1–2 early
  tweets is below `MIN_CLUSTER_SIZE` and gets dropped. There's a real tension
  between noise suppression and catching things early.
- **The centroid can blur** if my interests are multi-modal (e.g. infra *and*
  ML). A single centroid averages them; clustering my *preferences* into a few
  centroids would represent that better.
- **WhatsApp feedback webhook** is stubbed as a CLI command here; production needs
  the Twilio inbound webhook wired to `apply_feedback`.

---

## Layout

```
run.py                       CLI: seed / digest / feedback / reset
pyproject.toml               packaging + pinned/optional dependencies
requirements.txt             core deps (no API keys needed)
trend_radar/                 the package
├── config.py                all knobs + backend selection
├── ingest.py                load + normalize tweets  (only file that changes for live X)
├── embeddings.py            pluggable embedding backends
├── store.py                 transparent numpy vector store
├── trends.py                clustering + engagement velocity  ← the core
├── preferences.py           embedding-centroid personalization + feedback
├── digest.py                ranking + per-item digest + pluggable summaries
├── notify.py                console / WhatsApp delivery
├── pipeline.py              wires the diagram together; closes the feedback loop
└── sample_data.py           synthetic X dump (real-schema stand-in)
data/                        generated artifacts (tweets.csv, preferences.json) — gitignored
docs/architecture.md         the system diagram
tests/test_pipeline.py       offline smoke + feedback-loop tests
```
