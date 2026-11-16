"""Complaint themes in negative reviews: BERTopic (sentence embeddings + UMAP + HDBSCAN) when installed, otherwise
NMF on TF-IDF. Reports the method used, so results are never presented as BERTopic when they are not."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from . import config as C


def themes(neg: pd.DataFrame, n_sample: int = C.TOPIC_SAMPLE, prefer: str = "bertopic", k_nmf: int = 12) -> Dict:
    d = neg.sample(min(n_sample, len(neg)), random_state=C.SEED).reset_index(drop=True)
    docs = d.review.tolist()
    if prefer == "bertopic":
        try:
            return _bertopic(d, docs)
        except ImportError:
            pass
    return _nmf(d, docs, k_nmf)


def _table(d, topic_ids, labels, sizes_from_ids=True):
    d = d.assign(topic=topic_ids)
    d = d[d.topic >= 0]
    out = []
    for t, g in d.groupby("topic"):
        out.append({"topic": int(t), "keywords": labels[t], "size": int(len(g)),
                    "share": round(len(g) / len(d), 4),
                    "top_conditions": g.condition.value_counts().head(3).index.tolist(),
                    "example": g.review.iloc[0][:220]})
    return sorted(out, key=lambda r: -r["size"])


def _bertopic(d, docs):
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    emb = SentenceTransformer("all-MiniLM-L6-v2")
    tm = BERTopic(embedding_model=emb, min_topic_size=60, vectorizer_model=CountVectorizer(stop_words="english",
                  ngram_range=(1, 2), min_df=5), calculate_probabilities=False, verbose=False)
    ids, _ = tm.fit_transform(docs)
    info = tm.get_topic_info()
    labels = {int(r.Topic): [w for w, _ in tm.get_topic(int(r.Topic))[:6]] for r in info.itertuples() if r.Topic >= 0}
    rows = _table(d, ids, labels)
    return {"method": "BERTopic (all-MiniLM-L6-v2, UMAP, HDBSCAN)", "documents": len(docs),
            "outlier_share": round(float((pd.Series(ids) == -1).mean()), 4), "topics": rows[:12]}


def _nmf(d, docs, k):
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(stop_words="english", min_df=5, max_df=0.5, ngram_range=(1, 2))
    X = v.fit_transform(docs)
    nmf = NMF(n_components=k, random_state=C.SEED, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X)
    vocab = v.get_feature_names_out()
    labels = {t: [vocab[i] for i in nmf.components_[t].argsort()[::-1][:6]] for t in range(k)}
    return {"method": "NMF on TF-IDF (BERTopic not installed)", "documents": len(docs), "outlier_share": 0.0,
            "topics": _table(d, W.argmax(1), labels)}
