"""
Ingestion layer: load + clean tweets from the static dataset.

This is deliberately the thinnest part of the system. In the diagram it's the
box between the X dataset and the embedder. When real X API access lands, this
is the *only* file that changes -- everything downstream consumes the same
normalized Tweet objects.
"""
from dataclasses import dataclass
from datetime import datetime
import pandas as pd


@dataclass
class Tweet:
    id: int
    text: str
    engagement: int          # likes + retweets, our traction proxy
    created_at: datetime
    author: str


def load_tweets(path: str) -> list[Tweet]:
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text", "author"])

    tweets = []
    for _, r in df.iterrows():
        tweets.append(Tweet(
            id=int(r["id"]),
            text=r["text"],
            engagement=int(r.get("likes", 0)) + int(r.get("retweets", 0)),
            created_at=datetime.fromisoformat(str(r["created_at"])),
            author=str(r["author"]),
        ))
    return tweets
