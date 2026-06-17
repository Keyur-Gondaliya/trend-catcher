# Trend Radar

[![CI](https://github.com/Keyur-Gondaliya/trend-catcher/actions/workflows/ci.yml/badge.svg)](https://github.com/Keyur-Gondaliya/trend-catcher/actions/workflows/ci.yml)

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

**Embedding centroids for preferences, not topic tags.** The preference store is
a small set of vectors — interest *modes* learned from what I mark relevant via
online clustering. A trend is scored against its *best-matching* mode, so caring
about both infra *and* ML doesn't average into a mush that matches neither.
New, never-seen topics still rank correctly if they sit near any mode in
embedding space; tags can't generalize like that.
(`trend_radar/preferences.py`)

**Configured cold-start prior, then learned.** Before any feedback, the
preference centroid is seeded from a plain-language interest prompt in
`config.py` (`DEFAULT_INTEREST_PROMPT`), embedded into the same space as the
tweets — so the *very first* digest already leans toward what I care about
instead of ranking by raw popularity. Feedback then folds in and overrides the
prior over a few cycles. (`trend_radar/preferences.py`, `trend_radar/pipeline.py`)

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
python run.py digest     # first run: seeded by your interest prompt (see config)
python run.py feedback "2 yes, 3 yes, 1 no, 5 no"
python run.py digest     # now personalized — watch the ranking move
```

Or run the whole story in one command:

```bash
./scripts/demo.sh            # seed → cold digest → feedback → personalized re-rank
PAUSE=1 ./scripts/demo.sh    # pause before each step (handy when recording a demo)
```

Run the tests (installs the package + pytest):

```bash
pip install -e ".[dev]"
pytest
```

Upgrade quality with the optional backends (each maps to an extra in
`pyproject.toml`):

```bash
pip install ".[embeddings]"   # better semantics — sentence-transformers
EMBEDDING_BACKEND=sentence-transformers SUMMARIZER_BACKEND=anthropic python run.py digest
```

Send to real WhatsApp (Twilio) and receive feedback replies:

```bash
pip install ".[whatsapp]"     # twilio + flask + gunicorn
DELIVERY_BACKEND=whatsapp TWILIO_SID=... TWILIO_TOKEN=... \
TWILIO_WHATSAPP_FROM=+14155238886 MY_WHATSAPP_NUMBER=+91... python run.py digest
python run.py webhook         # serve the inbound-feedback endpoint on :5000
```

The twice-weekly schedule (Wed + Sun) is just cron calling `python run.py digest`:
```
0 9 * * 3,0  cd /path/to/trend-radar && python run.py digest
```

---

## What it looks like

First run — a **cold start that's already seeded by the interest prompt** (note
the spread in `fit`, not a flat 0.5):

```
📡 Trend Radar — your twice-weekly tech digest
   (starting from your configured interests — reply to refine)

1. Stop Prompt          · 3 voices    trend 0.942 · fit 0.616 · score 0.812
2. Harness Engineering  · 5 voices    trend 0.972 · fit 0.537 · score 0.798
3. Engineering Context  · 3 voices    trend 0.752 · fit 0.752 · score 0.752
4. Shipping Code        · 4 voices    trend 0.721 · fit 0.654 · score 0.694
5. Rebuilt Harness      · 4 voices    trend 0.559 · fit 0.527 · score 0.546
```

Reply `1 yes, 3 yes, 2 no`, run again — **same data, the ranking moves toward
your taste**:

```
1. Stop Prompt          · 3 voices    trend 0.942 · fit 1.00 · score 0.965    ▲ liked → fit maxes out
2. Engineering Context  · 3 voices    trend 0.752 · fit 0.858 · score 0.794    ▲ #3 → #2
3. Harness Engineering  · 5 voices    trend 0.972 · fit 0.525 · score 0.793    ▼ #2 → #3 (marked "no")
4. Shipping Code        · 4 voices    trend 0.721 · fit 0.621 · score 0.681
5. Rebuilt Harness      · 4 voices    trend 0.559 · fit 0.518 · score 0.543
```

`trend` is strength (engagement × velocity), `fit` is personal relevance
(best match across your interest modes), and `score` blends them 0.6 / 0.4.
Each item also carries a one-line summary (elided above). Reproduce it any time
with `./scripts/demo.sh`.

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

The default image is the lean offline build (tfidf + template). Heavier
backends ship as compose **profiles** that build with the matching optional
extras from `pyproject.toml`:

```bash
# Semantic embeddings (sentence-transformers) — model cached in a named volume
docker compose --profile semantic run --rm trend-radar-semantic digest

# WhatsApp inbound-feedback webhook (long-running, listens on :5000)
docker compose --profile webhook up webhook
```

---

## Limitations & trade-offs (honest)

- **The offline default trades quality for zero-setup.** With no keys it uses
  TF-IDF, which clusters by words, not meaning. The `sentence-transformers`
  backend is the real fix (verified: "stop prompt hacking" + "context
  engineering" merge into one 8-voice cluster, AI code review goes 4 → 12 voices)
  via `EMBEDDING_BACKEND=sentence-transformers` or the `semantic` Docker profile.
  TF-IDF stays the default only to keep review frictionless.
- **False-positive trends.** A burst of similar-looking but unrelated tweets can
  cluster together and look like a trend. The "≥2 distinct authors" rule helps
  but isn't bulletproof; a real fix is engagement-source diversity scoring.
- **False negatives on early signals.** A genuinely new trend with only 1–2 early
  tweets is below `MIN_CLUSTER_SIZE` and gets dropped. There's a real tension
  between noise suppression and catching things early.
- **WhatsApp feedback needs your own setup to go live.** The endpoint exists
  (`trend_radar/webhook.py`), but receiving real replies needs your Twilio creds
  and a public URL (e.g. ngrok) pointed at `/whatsapp`.

---

## Layout

```
run.py                       CLI: seed / digest / feedback / reset / webhook
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
├── webhook.py               Twilio inbound-feedback webhook (closes the loop live)
└── sample_data.py           synthetic X dump (real-schema stand-in)
scripts/demo.sh              one-command end-to-end walkthrough (offline)
data/                        generated artifacts (tweets.csv, preferences.json) — gitignored
docs/architecture.md         the system diagram
tests/test_pipeline.py       offline smoke + feedback-loop + multi-centroid + webhook tests
Dockerfile / docker-compose.yml   lean default + semantic / webhook profiles
.github/workflows/ci.yml     runs the tests on push (py 3.9 + 3.12)
```

---

## License

MIT — see [LICENSE](LICENSE).
