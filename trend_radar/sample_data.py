"""
Generates a synthetic X (Twitter) dataset that stands in for a real dump
until X API access is available. The schema matches what a real export looks
like, so swapping in real data later is just pointing DATA_PATH at it.

Crucially: tweets are spread across a time range and some topics *accelerate*
(more, higher-engagement tweets toward the end) while others stay flat. That's
what lets the trend engine show velocity on otherwise-static data.
"""
import csv
import random
from datetime import datetime, timedelta

random.seed(42)

# (topic, sample phrasings, whether it's "heating up" over the window)
TOPICS = [
    ("harness engineering",
     ["Harness engineering is changing how we ship",
      "Just rebuilt our CI with harness engineering principles",
      "Why harness engineering beats hand-rolled pipelines",
      "Our team moved to harness engineering and deploys dropped to minutes",
      "harness engineering thread: what it is and why it matters"], True),
    ("context engineering for LLMs",
     ["Context engineering is the real skill for LLM apps",
      "Stop prompt hacking, start context engineering",
      "Context engineering > prompt engineering, here's why",
      "How we cut hallucinations with better context engineering",
      "Context engineering patterns for production agents"], True),
    ("vector databases",
     ["Comparing vector databases for RAG at scale",
      "Vector databases are eating search",
      "When do you actually need a vector database?",
      "Benchmarking pgvector vs dedicated vector databases"], False),
    ("rust for backend",
     ["Rewrote our service in Rust, here's what happened",
      "Rust for backend is finally practical",
      "Why we picked Rust over Go for the new service"], False),
    ("ai code review agents",
     ["AI code review agents caught a bug we missed for months",
      "Shipping an AI code review agent this week",
      "AI code review agents are getting scary good",
      "Our AI code review agent now gates every PR",
      "Thread: building an AI code review agent from scratch"], True),
    ("kubernetes cost",
     ["Cutting our kubernetes bill in half",
      "Kubernetes cost is out of control, here's our fix"], False),
    ("local first apps",
     ["Local-first apps are the future of UX",
      "Building a local-first sync engine"], False),
]

def main(path="data/tweets.csv", n_noise=40):
    start = datetime(2025, 5, 1)
    window_days = 30
    rows = []
    tid = 1000

    for topic, phrasings, heating in TOPICS:
        # Heating topics get more tweets, weighted toward the end of the window
        count = random.randint(8, 14) if heating else random.randint(4, 7)
        for _ in range(count):
            if heating:
                # bias the day toward the end (recent) of the window
                day = int(window_days * (random.random() ** 0.5))
                eng = random.randint(40, 400) + day * 12  # engagement grows late
            else:
                day = random.randint(0, window_days)
                eng = random.randint(10, 120)
            ts = start + timedelta(days=day, hours=random.randint(0, 23))
            text = random.choice(phrasings)
            rows.append({
                "id": tid,
                "text": text,
                "likes": int(eng * 0.7),
                "retweets": int(eng * 0.3),
                "created_at": ts.isoformat(),
                "author": f"user_{random.randint(1, 60)}",
            })
            tid += 1

    # Unrelated noise so clustering has to actually discriminate
    noise = ["coffee thread", "weekend hike pics", "hot take on tabs vs spaces",
             "my cat just deployed to prod", "is it friday yet"]
    for _ in range(n_noise):
        day = random.randint(0, window_days)
        ts = start + timedelta(days=day, hours=random.randint(0, 23))
        rows.append({
            "id": tid, "text": random.choice(noise),
            "likes": random.randint(0, 30), "retweets": random.randint(0, 8),
            "created_at": ts.isoformat(), "author": f"user_{random.randint(1, 60)}",
        })
        tid += 1

    random.shuffle(rows)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "likes", "retweets",
                                          "created_at", "author"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} tweets to {path}")

if __name__ == "__main__":
    main()
