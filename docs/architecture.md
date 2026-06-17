flowchart TD
    DS[(X Dataset<br/>tweets.csv)] -->|ingest.py| ING[Load & normalize tweets]
    ING -->|embeddings.py| EMB[Embed text<br/>tfidf / ST / openai]
    EMB --> VDB[(Vector Store<br/>store.py)]

    CRON{{Cron trigger<br/>Wed + Sun}} -->|run.py digest| ENG

    VDB --> ENG[Trend Engine<br/>trends.py]
    subgraph ENG_INNER[Inside the engine]
        CL[Cluster by similarity<br/>>=3 tweets, >=2 authors] --> VEL[Engagement velocity<br/>timestamp buckets]
        VEL --> STR[Trend strength]
    end
    ENG --> CL

    STR --> RANK{Rank<br/>0.6 trend + 0.4 fit}
    PREF[(Preference centroid<br/>preferences.py)] --> RANK

    RANK -->|digest.py| DIG[Per-item digest<br/>label + summary]
    DIG -->|notify.py| WA[/WhatsApp message<br/>to my number/]
    WA --> ME([I read it])
    ME -->|reply: 2 yes, 4 no| FB[apply_feedback<br/>pipeline.py]
    FB -->|update centroid| PREF

    classDef store fill:#e8f0fe,stroke:#4285f4;
    classDef engine fill:#fff4e5,stroke:#fb8c00;
    class DS,VDB,PREF store;
    class CL,VEL,STR,RANK engine;