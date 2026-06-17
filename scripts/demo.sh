#!/usr/bin/env bash
#
# Trend Radar — end-to-end demo (offline, no API keys).
#
# Tells the whole story in one command: a cold start that's already seeded by
# your interest prompt, per-item feedback, and a personalized re-rank you can
# watch move. Then it shows the interest "modes" the system actually learned.
#
# Usage:
#   ./scripts/demo.sh           # run straight through
#   PAUSE=1 ./scripts/demo.sh   # pause before each step (handy when recording)
#   EMBEDDING_BACKEND=sentence-transformers ./scripts/demo.sh   # semantic
#
# State is isolated to data/demo_* (gitignored), so your real preferences and
# dataset are never touched.
set -euo pipefail

cd "$(dirname "$0")/.."

# Keep the demo's state separate from any real run.
export DATA_PATH="data/demo_tweets.csv"
export PREFERENCE_PATH="data/demo_preferences.json"
export DIGEST_PATH="data/demo_last_digest.json"

step() {
  echo
  echo "════════════════════════════════════════════════════════════"
  echo "▶ $*"
  echo "════════════════════════════════════════════════════════════"
  if [ "${PAUSE:-0}" = "1" ]; then read -rp "   (press enter to run)…" _; fi
}

step "0. Clean slate (isolated demo state)"
python3 run.py reset

step "1. Seed a sample X dataset — stands in for a real export"
python3 run.py seed

step "2. First digest — cold start, but already leaning toward your configured"
echo "   interests (DEFAULT_INTEREST_PROMPT), not just raw popularity."
python3 run.py digest

step "3. Reply with per-item feedback — this is the WhatsApp reply, as a CLI cmd"
python3 run.py feedback "1 yes, 3 yes, 2 no"

step "4. Digest again — same data, but watch the ranking move toward your taste"
python3 run.py digest

step "5. Under the hood — the interest 'modes' it actually learned"
python3 - <<'PY'
from trend_radar.preferences import PreferenceStore
p = PreferenceStore()
print(f"   interest modes: {len(p.centroids)} | per-mode signal counts: {p.counts} "
      f"| cold: {p.is_cold} | prior: {p.prior}")
PY

echo
echo "Done. Tip: EMBEDDING_BACKEND=sentence-transformers ./scripts/demo.sh"
echo "for real semantic clustering instead of the offline tfidf default."
