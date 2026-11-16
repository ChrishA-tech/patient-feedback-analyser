import json

import numpy as np
import pytest

from pfa import data, memo, metrics, models, ner, topics


def test_clean_and_labels(raw_dir):
    df = data.load(raw_dir)
    assert "&#039;" not in df.review.iloc[0] and not df.review.iloc[0].startswith('"')
    assert "Unknown" in set(df.condition) and not df.condition.str.contains("</span>").any()
    assert set(df.label) <= {0, 1, -1} and (df.loc[df.rating >= 7, "label"] == 1).all()


def test_splits_no_leakage_and_unseen_conditions(splits):
    tr, te, un = splits["train"], splits["test"], splits["unseen"]
    assert not (set(te.review) & set(tr.review)) and not (set(un.review) & set(tr.review))
    assert not (set(un.condition) & set(tr.condition))
    assert splits["stats"]["unseen_conditions"] and splits["stats"]["sizes"]["test"] > 0


def test_metrics_and_threshold():
    y = np.array([0, 0, 1, 1, 1, 0] * 20)
    p = np.where(y == 1, 0.8, 0.3)
    e = metrics.evaluate(y, p)
    assert e["macro_f1"] == 1.0 and e["roc_auc"] == 1.0 and e["macro_f1_ci95"][0] == 1.0
    assert 0.3 < metrics.best_threshold(y, p) <= 0.8


def test_tfidf_model_and_disagreements(splits):
    m, res = models.fit_tfidf(splits)
    assert res["test"]["macro_f1"] > 0.9 and "unseen" in res
    sample, share = models.disagreements(m, splits["test"])
    assert 0 <= share < 0.2


def test_keras_model(splits):
    pytest.importorskip("tensorflow")
    _, res = models.fit_keras(splits, epochs=2, max_tokens=500, seq_len=32)
    assert res["test"]["macro_f1"] > 0.8


def test_ner_ruler_and_side_effect_odds(splits):
    pytest.importorskip("spacy")
    nlp = ner.build_nlp(["Sertraline", "Metformin"], use_scispacy=False)
    ents = ner.extract(nlp, ["Sertraline gave me terrible Nausea and headaches", "Metformin was fine"])
    labels = {(l, t) for l, t, *_ in ents[0]}
    assert ("DRUG", "sertraline") in labels and ("SIDE_EFFECT", "nausea") in labels and ("SIDE_EFFECT", "headache") in labels
    te = splits["test"].reset_index(drop=True)
    e = ner.extract(nlp, te.review.tolist())
    risk = ner.side_effect_risk(te, e, min_count=5)
    top = risk.set_index("side_effect")
    assert top.loc["nausea", "odds_ratio_negative"] > 5 and top.loc["nausea", "share_of_positive"] < 0.01
    assert len(ner.entity_review_sample(te, e, n=10)) == 10


def test_topics_nmf_fallback(splits):
    neg = splits["train"][splits["train"].label == 0]
    t = topics.themes(neg, n_sample=500, prefer="nmf", k_nmf=4)
    assert t["method"].startswith("NMF") and t["topics"] and abs(sum(x["share"] for x in t["topics"]) - 1) < 0.01


def test_memo_renders(splits):
    _, lr = models.fit_tfidf(splits)
    r = {"data": splits["stats"], "models": {"tfidf_lr": lr, "keras": None}, "disagreement_share": 0.02,
         "side_effects": [{"side_effect": "nausea", "share_of_negative": 0.3, "share_of_positive": 0.01,
                           "odds_ratio_negative": 30.0, "or_ci95": [10.0, 90.0]}],
         "topics": {"method": "NMF", "topics": [{"keywords": ["weight", "gain"], "share": 0.4, "top_conditions": ["A"]}]}}
    text = memo.render(r)
    assert "nausea" in text and "Not** clinical" in text


def test_sagemaker_inference_contract(tmp_path, splits):
    import importlib.util
    import joblib
    spec = importlib.util.spec_from_file_location("train", "aws/train.py")
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)
    (tmp_path / "data").mkdir()
    splits["train"][["review", "label"]].to_csv(tmp_path / "data/train.csv", index=False)
    splits["valid"][["review", "label"]].to_csv(tmp_path / "data/valid.csv", index=False)
    (tmp_path / "model").mkdir()

    class A:
        model_dir, train, output_data_dir = str(tmp_path / "model"), str(tmp_path / "data"), str(tmp_path / "out")
    tr.train(A)
    m = tr.model_fn(A.model_dir)
    body, ctype = tr.output_fn(tr.predict_fn(tr.input_fn(json.dumps({"review": "Awful nausea, never again"})), m))
    out = json.loads(body)
    assert ctype == "application/json" and set(out) == {"predicted_label", "probability"} and out["predicted_label"] == 0
    import pandas as pd
    base = pd.read_csv(tmp_path / "out/baseline.csv")
    assert list(base.columns) == ["prediction", "probability", "label"]
    assert json.load(open(tmp_path / "out/metrics.json"))["valid_macro_f1"] > 0.9


def test_ground_truth_format():
    from datetime import datetime
    from aws.gt import gt_key, gt_record
    r = gt_record("abc", 1)
    assert r == {"groundTruthData": {"data": "1", "encoding": "CSV"}, "eventMetadata": {"eventId": "abc"},
                 "eventVersion": "0"}
    assert gt_key("p/gt", datetime(2026, 11, 16, 9)) == "p/gt/2026/11/16/09/ground_truth.jsonl"


def test_run_local_end_to_end_and_readme(raw_dir, tmp_path, monkeypatch):
    pytest.importorskip("spacy")
    from pfa import config as C, data as D, report, run_local
    monkeypatch.setattr(C, "DATA", raw_dir)
    monkeypatch.setattr(C, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(C, "NER_SAMPLE", 300)
    monkeypatch.setattr(D, "download", lambda out=None: raw_dir)
    monkeypatch.setattr("pfa.ner.build_nlp.__defaults__", (False,))
    r = run_local.main(skip_keras=True, topic_method="nmf")
    assert (tmp_path / "results/memo.md").exists() and r["topics"]["method"].startswith("NMF")
    readme = tmp_path / "README.md"
    readme.write_text("a\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\nb")
    monkeypatch.setattr(C, "ROOT", tmp_path)
    report.fill()
    txt = readme.read_text()
    assert "TF-IDF" in txt and "Complaint themes" in txt and "old" not in txt


def test_report_includes_aws_section(tmp_path, monkeypatch, splits):
    from pfa import config as C, report
    _, lr = models.fit_tfidf(splits)
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(C, "RESULTS", tmp_path / "results")
    (tmp_path / "results/results.json").write_text(json.dumps({"data": splits["stats"], "models": {"tfidf_lr": lr},
                                                               "disagreement_share": 0.02}))
    (tmp_path / "results/aws.json").write_text(json.dumps({
        "region": "eu-west-2", "training_job": "patient-feedback-1", "training_billable_seconds": 180,
        "valid_metrics": {"valid_macro_f1": 0.91}, "approval": "Approved",
        "model_package_arn": "arn:aws:sagemaker:eu-west-2:1:model-package/patient-feedback-sentiment/1",
        "traffic_n": 500, "latency_ms_p50": 40.1, "latency_ms_p95": 80.2, "monitor_status": "CompletedWithViolations",
        "monitor_metrics": {"accuracy": 0.86, "f1": 0.88}, "violations": [{"metric": "f1"}], "cleaned_up_at": "x"}))
    txt = report.render()
    assert "patient-feedback-sentiment/1" in txt and "Approved" in txt and "violations: f1" in txt


def test_download_uses_local_zip(raw_dir, tmp_path, monkeypatch):
    import io, zipfile
    from pfa import config as C, data as D
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        for n in ("drugsComTrain_raw.tsv", "drugsComTest_raw.tsv"):
            z.write(raw_dir / n, n)
    with zipfile.ZipFile(tmp_path / "drug+reviews.zip", "w") as z:     # nested zip, as UCI ships it
        z.writestr("drugsCom_raw.zip", inner.getvalue())
    monkeypatch.setattr(C, "UCI_URLS", [])
    assert (D.download(tmp_path) / "drugsComTest_raw.tsv").exists()
