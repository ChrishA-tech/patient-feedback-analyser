"""SageMaker scikit-learn entry point: training (python train.py) and inference (model_fn ... output_fn).

Self-contained on purpose: the SageMaker container does not have the pfa package."""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd


def build():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300000, sublinear_tf=True,
                                               strip_accents="unicode")),
                     ("lr", LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"))])


def best_threshold(y, p):
    from sklearn.metrics import f1_score
    grid = np.linspace(0.2, 0.8, 61)
    return float(max(grid, key=lambda t: f1_score(y, (p >= t).astype(int), average="macro")))


def train(args):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    tr = pd.read_csv(os.path.join(args.train, "train.csv"))
    va = pd.read_csv(os.path.join(args.train, "valid.csv"))
    m = build().fit(tr.review.astype(str), tr.label)
    p = m.predict_proba(va.review.astype(str))[:, 1]
    thr = best_threshold(va.label.to_numpy(), p)
    pred = (p >= thr).astype(int)
    metrics = {"valid_macro_f1": float(f1_score(va.label, pred, average="macro")),
               "valid_accuracy": float(accuracy_score(va.label, pred)), "valid_auc": float(roc_auc_score(va.label, p)),
               "threshold": thr, "train_rows": int(len(tr)), "valid_rows": int(len(va))}
    for k, v in metrics.items():
        print(f"{k}={v}")                     # parsed by the training job's metric definitions
    joblib.dump(m, os.path.join(args.model_dir, "model.joblib"))
    json.dump({"threshold": thr}, open(os.path.join(args.model_dir, "threshold.json"), "w"))
    os.makedirs(args.output_data_dir, exist_ok=True)
    # Baseline for Model Monitor (model quality): the model's own validation predictions with true labels.
    pd.DataFrame({"prediction": pred, "probability": p.round(6), "label": va.label}).to_csv(
        os.path.join(args.output_data_dir, "baseline.csv"), index=False)
    json.dump(metrics, open(os.path.join(args.output_data_dir, "metrics.json"), "w"), indent=2)


# ------------------------------------------------------------------ inference (SageMaker hosting)
def model_fn(model_dir):
    thr = json.load(open(os.path.join(model_dir, "threshold.json")))["threshold"]
    return {"model": joblib.load(os.path.join(model_dir, "model.joblib")), "threshold": thr}


def input_fn(body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"unsupported content type {content_type}")
    d = json.loads(body)
    return d["reviews"] if "reviews" in d else [d["review"]]


def predict_fn(reviews, m):
    p = m["model"].predict_proba([str(r) for r in reviews])[:, 1]
    return [{"predicted_label": int(x >= m["threshold"]), "probability": round(float(x), 6)} for x in p]


def output_fn(preds, accept="application/json"):
    # One review per request -> one flat record, which is what Model Monitor's model-quality merge expects.
    return json.dumps(preds[0] if len(preds) == 1 else {"predictions": preds}), "application/json"


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "model"))
    a.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "data"))
    a.add_argument("--output-data-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "output"))
    train(a.parse_args())
