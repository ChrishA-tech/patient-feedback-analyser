"""Two sentiment models: TF-IDF + logistic regression (deployed on SageMaker), and a TensorFlow/Keras CNN."""
from __future__ import annotations

import time
from typing import Dict

import numpy as np

from . import config as C
from .metrics import best_threshold, evaluate


def tfidf_lr():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True,
                                               strip_accents="unicode")),
                     ("lr", LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"))])


def fit_tfidf(s: Dict):
    t = time.time()
    m = tfidf_lr().fit(s["train"].review, s["train"].label)
    pv = m.predict_proba(s["valid"].review)[:, 1]
    thr = best_threshold(s["valid"].label, pv)
    res = {"model": "TF-IDF (1-2 grams) + logistic regression", "threshold": round(thr, 3),
           "train_seconds": round(time.time() - t, 1)}
    for k in ("valid", "test", "unseen"):
        res[k] = evaluate(s[k].label, m.predict_proba(s[k].review)[:, 1], thr)
    t = time.time()
    m.predict_proba(s["test"].review.head(1000))
    res["ms_per_review_cpu"] = round((time.time() - t), 3)
    return m, res


def fit_keras(s: Dict, epochs: int = 4, max_tokens: int = 30000, seq_len: int = 256):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(C.SEED)
    vec = tf.keras.layers.TextVectorization(max_tokens=max_tokens, output_sequence_length=seq_len)
    vec.adapt(tf.data.Dataset.from_tensor_slices(s["train"].review.to_numpy()).batch(512))
    inp = tf.keras.Input(shape=(), dtype=tf.string)
    x = vec(inp)
    x = tf.keras.layers.Embedding(max_tokens, 128)(x)
    x = tf.keras.layers.Conv1D(128, 5, activation="relu")(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])
    pos = s["train"].label.mean()
    t = time.time()
    def ds(df, shuffle=False):
        d = tf.data.Dataset.from_tensor_slices((tf.constant(df.review.astype(str).tolist()),
                                                tf.constant(df.label.to_numpy(), dtype=tf.float32)))
        return (d.shuffle(len(df), seed=C.SEED) if shuffle else d).batch(256).prefetch(2)
    model.fit(ds(s["train"], True), validation_data=ds(s["valid"]), epochs=epochs, verbose=2,
              class_weight={0: 0.5 / (1 - pos), 1: 0.5 / pos},
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=1,
                                                          restore_best_weights=True)])

    def predict(texts):
        d = tf.data.Dataset.from_tensor_slices(tf.constant([str(t) for t in texts])).batch(1024)
        return model.predict(d, verbose=0).ravel()
    pv = predict(s["valid"].review)
    thr = best_threshold(s["valid"].label, pv)
    res = {"model": "TensorFlow/Keras CNN (embedding + Conv1D)", "threshold": round(thr, 3),
           "train_seconds": round(time.time() - t, 1), "tensorflow": tf.__version__}
    for k in ("valid", "test", "unseen"):
        res[k] = evaluate(s[k].label, predict(s[k].review), thr)
    return predict, res


def disagreements(model, df, n: int = 100, seed: int = C.SEED):
    """Reviews where the model is confident and disagrees with the star rating: either the rating or the model is
    wrong. Exported for manual review (the rating is a noisy proxy for the sentiment of the text)."""
    p = model.predict_proba(df.review)[:, 1]
    conf_wrong = df.assign(p_positive=p)[((p > 0.95) & (df.label == 0)) | ((p < 0.05) & (df.label == 1))]
    share = len(conf_wrong) / len(df)
    return conf_wrong.sample(min(n, len(conf_wrong)), random_state=seed), round(float(share), 4)
