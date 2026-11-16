"""Run everything that runs locally (free): data, models, entities, topics, memo -> results/."""
from __future__ import annotations

import json
import sys
import time

from . import config as C
from . import data, memo, models, ner, topics


def main(skip_keras: bool = False, topic_method: str = "bertopic"):
    C.RESULTS.mkdir(exist_ok=True)
    t0 = time.time()
    data.download()
    df = data.load()
    s = data.build_splits(df)
    print(json.dumps(s["stats"], indent=1)[:1500])
    r = {"data": s["stats"], "models": {}}

    lr, r["models"]["tfidf_lr"] = models.fit_tfidf(s)
    print("tfidf", r["models"]["tfidf_lr"]["test"], r["models"]["tfidf_lr"]["unseen"])
    import joblib
    joblib.dump(lr, C.RESULTS / "tfidf_lr.joblib")
    sample, r["disagreement_share"] = models.disagreements(lr, s["test"])
    sample[["drug", "condition", "rating", "p_positive", "review"]].assign(rating_or_model_wrong="").to_csv(
        C.RESULTS / "disagreement_sample.csv", index=False)
    if not skip_keras:
        try:
            _, r["models"]["keras"] = models.fit_keras(s)
            print("keras", r["models"]["keras"]["test"], r["models"]["keras"]["unseen"])
        except ImportError as e:
            r["models"]["keras"] = None
            print("TensorFlow not installed, skipping Keras:", e)

    sub = s["test"].sample(min(C.NER_SAMPLE, len(s["test"])), random_state=C.SEED).reset_index(drop=True)
    nlp = ner.build_nlp(df.drug.unique())
    ents = ner.extract(nlp, sub.review.tolist())
    r["ner"] = ner.summary(nlp, ents)
    r["side_effects"] = ner.side_effect_risk(sub, ents).to_dict("records")
    ner.entity_review_sample(sub, ents).to_csv(C.RESULTS / "entity_sample.csv", index=False)

    neg = s["train"][s["train"].label == 0]
    r["topics"] = topics.themes(neg, prefer=topic_method)
    r["minutes"] = round((time.time() - t0) / 60, 1)
    (C.RESULTS / "results.json").write_text(json.dumps(r, indent=2, default=str))
    (C.RESULTS / "memo.md").write_text(memo.render(r))
    print(memo.render(r))
    return r


if __name__ == "__main__":
    main(skip_keras="--no-keras" in sys.argv, topic_method="nmf" if "--nmf" in sys.argv else "bertopic")
