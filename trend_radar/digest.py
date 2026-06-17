"""
Digest generation: turn ranked trends into a readable, per-item message.

Per-item is deliberate. Each trend gets its own block with a stable item id, so
the feedback reply ("2 not for me") maps to exactly one cluster centroid. That
makes every digest a labeled training signal instead of one vague thumbs-up.

The summarizer is pluggable like the embedder: "template" builds a headline
from the top tweet + distinctive terms (no LLM, runs anywhere), while
"anthropic"/"openai" write a tighter natural-language summary when keys exist.
"""
from dataclasses import dataclass
import re
from collections import Counter

from trend_radar import config
from trend_radar.trends import Trend
from trend_radar.preferences import PreferenceStore


@dataclass
class DigestItem:
    item_id: int
    label: str
    summary: str
    strength: float
    relevance: float
    score: float
    voices: int
    cluster_id: int


_STOP = set("the a an and or for to of in on with is are this that it your you "
            "we our why how what just here s re".split())


def _keyterms(texts, k=3):
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", " ".join(texts).lower())
    words = [w for w in words if w not in _STOP and len(w) > 2]
    return [w for w, _ in Counter(words).most_common(k)]


def _template_summary(trend: Trend) -> tuple[str, str]:
    terms = _keyterms(trend.texts)
    label = " ".join(terms[:2]).title() if terms else "Emerging topic"
    summary = (f"{len(trend.authors)} people are talking about this. "
               f"Top post: \"{trend.top_tweet}\". "
               f"Key terms: {', '.join(terms)}.")
    return label, summary


def _llm_summary(trend: Trend):
    sample = "\n".join(f"- {t}" for t in trend.texts[:8])
    prompt = (f"These tweets all cluster into one emerging tech trend.\n{sample}\n\n"
              "Reply with: a 3-5 word headline on line 1, then one sentence "
              "explaining why it's gaining traction.")
    try:
        if config.SUMMARIZER_BACKEND == "anthropic":
            from anthropic import Anthropic
            msg = Anthropic().messages.create(
                model="claude-sonnet-4-6", max_tokens=120,
                messages=[{"role": "user", "content": prompt}])
            text = msg.content[0].text.strip()
        else:  # openai
            from openai import OpenAI
            r = OpenAI().chat.completions.create(
                model="gpt-4o-mini", max_tokens=120,
                messages=[{"role": "user", "content": prompt}])
            text = r.choices[0].message.content.strip()
        head, *rest = text.split("\n", 1)
        return head.strip("# "), (rest[0].strip() if rest else "")
    except Exception:
        return _template_summary(trend)  # graceful fallback


def build_digest(trends: list[Trend], prefs: PreferenceStore) -> list[DigestItem]:
    scored = []
    for t in trends:
        rel = prefs.relevance(t.centroid)
        score = config.WEIGHT_TREND * t.strength + config.WEIGHT_PREF * rel
        scored.append((t, rel, score))
    scored.sort(key=lambda x: x[2], reverse=True)

    items = []
    for i, (t, rel, score) in enumerate(scored[:config.TOP_N_TRENDS], start=1):
        if config.SUMMARIZER_BACKEND == "template":
            label, summary = _template_summary(t)
        else:
            label, summary = _llm_summary(t)
        items.append(DigestItem(
            item_id=i, label=label, summary=summary,
            strength=round(t.strength, 3), relevance=round(rel, 3),
            score=round(score, 3), voices=len(t.authors), cluster_id=t.cluster_id))
    return items


def render_text(items: list[DigestItem], cold: bool) -> str:
    lines = ["📡 *Trend Radar* — your twice-weekly tech digest", ""]
    if cold:
        lines.append("_(cold start: still learning your taste — "
                     "ranked by trend strength for now)_\n")
    for it in items:
        lines.append(f"*{it.item_id}. {it.label}*  ·  {it.voices} voices")
        lines.append(it.summary)
        lines.append(f"   trend {it.strength} · fit {it.relevance} · "
                     f"score {it.score}")
        lines.append("")
    lines.append("Reply e.g. `1 yes, 3 no` to tune what you see next time.")
    return "\n".join(lines)
