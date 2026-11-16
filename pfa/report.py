"""Fill the README results section from results/results.json and results/aws.json."""
import json
import re

from . import config as C

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _ci(m):
    return f"{m['macro_f1']} ({m['macro_f1_ci95'][0]}–{m['macro_f1_ci95'][1]})" if m.get("macro_f1_ci95") else str(m["macro_f1"])


def render() -> str:
    p = C.RESULTS / "results.json"
    if not p.exists():
        return "_No results yet._"
    r = json.loads(p.read_text())
    d = r["data"]
    L = [f"**Data:** {d['reviews']:,} reviews, {d['drugs']:,} drugs, {d['conditions']:,} conditions. Labels from the "
         f"10-star rating (≥7 positive, ≤4 negative; {d['neutral_excluded']:,} neutral 5–6 left out). "
         f"{d['test_rows_removed_as_duplicates_of_training']:,} test reviews were removed because the same text is "
         f"in training (the dataset lists one review under several drug names). Train / valid / test / unseen = "
         + " / ".join(f"{d['sizes'][k]:,}" for k in ("train", "valid", "test", "unseen"))
         + f"; *unseen* = {len(d['unseen_conditions'])} conditions never seen in training.", "",
         "| Model | Macro-F1, test (95% CI) | ROC-AUC, test | Negative-review recall | Macro-F1, unseen conditions | Train time |",
         "|---|---|---|---|---|---|"]
    for m in r["models"].values():
        if m:
            L.append(f"| {m['model']} | {_ci(m['test'])} | {m['test']['roc_auc']} | {m['test']['negative_recall']} | "
                     f"{_ci(m['unseen'])} | {m['train_seconds']:.0f} s |")
    L += ["", f"**Rating vs text:** the deployed model confidently contradicts the star rating on "
              f"{r['disagreement_share']:.1%} of test reviews (sample in `results/disagreement_sample.csv`)."]
    n = r.get("ner", {})
    L += ["", f"**Entities** ({n.get('model')}, {n.get('reviews', 0):,} test reviews): "
              + ", ".join(f"{k} {v:,}" for k, v in n.get("entities_by_label", {}).items())
              + f"; {n.get('reviews_with_side_effect', 0):.0%} of reviews mention a side effect."]
    prec = C.RESULTS / "entity_precision.json"
    if prec.exists():
        e = json.loads(prec.read_text())
        L.append(f"Hand-checked precision: {e['precision']:.0%} on {e['checked']} sampled entities.")
    else:
        L.append("_Entity precision not hand-checked yet (`results/entity_sample.csv`)._")
    se = r.get("side_effects", [])[:8]
    if se:
        L += ["", "| Side effect | In negative reviews | In positive reviews | Odds ratio (95% CI) |", "|---|---|---|---|"]
        L += [f"| {s['side_effect']} | {s['share_of_negative']:.1%} | {s['share_of_positive']:.1%} | "
              f"{s['odds_ratio_negative']} ({s['or_ci95'][0]}–{s['or_ci95'][1]}) |" for s in se]
    t = r.get("topics", {})
    if t.get("topics"):
        L += ["", f"**Complaint themes** ({t['method']}, {t['documents']:,} negative reviews"
              + (f", {t['outlier_share']:.0%} unassigned" if t.get("outlier_share") else "") + "):", ""]
        L += [f"- {', '.join(x['keywords'][:5])} — {x['share']:.0%}" for x in t["topics"][:8]]
    a = C.RESULTS / "aws.json"
    if a.exists():
        w = json.loads(a.read_text())
        mm = w.get("monitor_metrics") or {}
        L += ["", "**AWS SageMaker run** (" + str(w.get("region")) + "):", "",
              f"- Training job `{w.get('training_job')}` ({w.get('training_billable_seconds')} billable s): valid "
              f"macro-F1 {(w.get('valid_metrics') or {}).get('valid_macro_f1', 0):.4f}.",
              f"- Model Registry: `{'/'.join((w.get('model_package_arn') or '').split('/')[-2:])}` → **{w.get('approval')}** by the "
              f"macro-F1 ≥ 0.80 gate.",
              f"- Real-time endpoint with data capture: {w.get('traffic_n')} reviews from unseen conditions, latency "
              f"p50 {w.get('latency_ms_p50')} ms / p95 {w.get('latency_ms_p95')} ms (client side, London region).",
              f"- Model Monitor (model quality vs validation baseline): {w.get('monitor_status')}"
              + (f"; accuracy {mm.get('accuracy')}, F1 {mm.get('f1')}" if mm else "")
              + (f"; violations: {', '.join(v['metric'] for v in w.get('violations') or [])}" if w.get("violations") else "")
              + (f"; failure: {w.get('monitor_failure')}" if w.get("monitor_failure") else "") + ".",
              f"- Endpoint and schedule deleted after the run ({w.get('cleaned_up_at', 'not yet')})."]
    return "\n".join(L)


def fill():
    readme = C.ROOT / "README.md"
    s = readme.read_text()
    readme.write_text(re.sub(re.escape(START) + ".*?" + re.escape(END), START + "\n" + render() + "\n" + END, s, flags=re.S))


if __name__ == "__main__":
    fill()
