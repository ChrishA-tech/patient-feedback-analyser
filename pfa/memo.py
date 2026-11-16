"""One-page insights memo for a patient-experience / medicines team, generated from results.json."""


def _pct(x):
    return "n/a" if x is None else f"{x:.1%}" if x < 0.1 else f"{x:.0%}"


def render(r: dict) -> str:
    st, lr = r["data"], r["models"]["tfidf_lr"]
    ker = r["models"].get("keras")
    best = max([m for m in (lr, ker) if m], key=lambda m: m["test"]["macro_f1"])
    se = [x for x in r.get("side_effects", []) if x.get("linked_to_negative")][:5]
    topics = r.get("topics", {}).get("topics", [])[:5]
    lines = [
        "# Patient drug reviews: what drives negative experiences",
        "",
        f"**For:** patient-experience and medicines teams | **Data:** {st['reviews']:,} public Drugs.com reviews "
        f"({st['drugs']:,} drugs, {st['conditions']:,} conditions; UCI, CC BY 4.0) | **Not** clinical or hospital data",
        "",
        "## What we can do automatically",
        f"- Flag negative reviews: {best['model']} reaches macro-F1 {best['test']['macro_f1']} "
        f"(95% CI {best['test']['macro_f1_ci95'][0]}–{best['test']['macro_f1_ci95'][1]}) on held-out reviews and "
        f"{best['unseen']['macro_f1']} on {len(st['unseen_conditions'])} conditions it never saw in training.",
        f"- {_pct(r['disagreement_share'])} of test reviews get a confident prediction that contradicts the star "
        "rating; a sample is exported for human review (often the rating, not the text, is the odd one out).",
        "",
        "## What patients complain about (themes in negative reviews)",
        *[f"- **{', '.join(t['keywords'][:4])}** — {_pct(t['share'])} of negative reviews "
          f"(most often: {', '.join(t['top_conditions'][:2])})" for t in topics],
        f"- Method: {r.get('topics', {}).get('method', 'n/a')}.",
        "",
        "## Side effects most linked to bad experiences",
        *[f"- **{s['side_effect']}**: in {_pct(s['share_of_negative'])} of negative vs {_pct(s['share_of_positive'])} of "
          f"positive reviews (odds ratio {s['odds_ratio_negative']}, 95% CI {s['or_ci95'][0]}–{s['or_ci95'][1]})"
          for s in se],
        *([] if se else ["- None of the common side effects is significantly more frequent in negative reviews."]),
        "- Mentions that are the condition being treated (e.g. 'anxiety' in an Anxiety review) are excluded.",
        "",
        "## Recommendations",
        "1. Route reviews the model flags as negative and that mention a side effect to the medicines-safety inbox first.",
        "2. Track the top complaint themes monthly; a rising share is an early signal worth a closer look.",
        "3. Re-validate on your own feedback before use: performance drops on unseen conditions, so expect a drop on a "
        "new data source too, and monitor it (the SageMaker deployment does this with Model Monitor).",
        "",
        "## Caveats",
        "- Online reviews are self-selected and unverified; associations here are not evidence that a drug causes an "
        "effect. Ratings are a noisy proxy for sentiment.",
        "- Entity extraction is lexicon and model based; its precision is measured on a hand-checked sample only "
        "where stated in the README.",
    ]
    return "\n".join(lines) + "\n"
