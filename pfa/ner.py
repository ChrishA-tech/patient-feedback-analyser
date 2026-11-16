"""Entities with spaCy: drugs (EntityRuler from the dataset's drug names) and side effects (scispaCy's biomedical
DISEASE model when installed, plus a curated side-effect lexicon). Then: which side effects go with bad reviews."""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

SIDE_EFFECTS = [
    "nausea", "vomiting", "headache", "headaches", "migraine", "dizziness", "dizzy", "fatigue", "tired", "tiredness",
    "drowsiness", "drowsy", "insomnia", "weight gain", "weight loss", "gained weight", "hair loss", "acne", "rash",
    "itching", "anxiety", "panic attacks", "depression", "mood swings", "irritability", "cramps", "cramping",
    "bleeding", "spotting", "constipation", "diarrhea", "diarrhoea", "dry mouth", "sweating", "night sweats",
    "hot flashes", "low libido", "loss of appetite", "increased appetite", "bloating", "heartburn", "stomach pain",
    "abdominal pain", "chest pain", "palpitations", "high blood pressure", "blurred vision", "muscle pain",
    "joint pain", "back pain", "tremors", "shaking", "numbness", "tingling", "brain fog", "memory loss",
    "suicidal thoughts", "nightmares", "vivid dreams", "swelling", "breast tenderness", "yeast infection",
    "shortness of breath", "cough", "fever", "hives", "sore throat", "dehydration", "withdrawal",
]


def build_nlp(drug_names: Iterable[str], use_scispacy: bool = True):
    import spacy
    nlp, model = None, "spacy.blank('en') + EntityRuler"
    if use_scispacy:
        try:
            import scispacy  # noqa: F401  (registers the custom tokenizer)
            nlp = spacy.load("en_ner_bc5cdr_md", exclude=["parser", "tagger", "lemmatizer"])
            model = "scispaCy en_ner_bc5cdr_md (CHEMICAL, DISEASE) + EntityRuler"
        except Exception:
            nlp = None
    if nlp is None:
        nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler", before="ner" if "ner" in nlp.pipe_names else None,
                         config={"phrase_matcher_attr": "LOWER", "overwrite_ents": True})
    drugs = sorted({d.strip() for d in drug_names if isinstance(d, str) and 3 <= len(d.strip()) <= 40})
    ruler.add_patterns([{"label": "DRUG", "pattern": d} for d in drugs] +
                       [{"label": "SIDE_EFFECT", "pattern": s} for s in SIDE_EFFECTS])
    nlp.meta["pfa_model"] = model
    return nlp


def normalise(label: str, text: str) -> str:
    t = text.lower().strip()
    return {"headaches": "headache", "dizzy": "dizziness", "tired": "fatigue", "tiredness": "fatigue",
            "drowsy": "drowsiness", "gained weight": "weight gain", "diarrhoea": "diarrhea", "cramping": "cramps",
            "shaking": "tremors"}.get(t, t)


def extract(nlp, texts: List[str], batch_size: int = 256) -> List[List[tuple]]:
    out = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        ents = []
        for e in doc.ents:
            lab = {"CHEMICAL": "DRUG", "DISEASE": "CONDITION_OR_SIDE_EFFECT"}.get(e.label_, e.label_)
            ents.append((lab, normalise(lab, e.text), e.start_char, e.end_char))
        out.append(ents)
    return out


def _is_indication(term: str, condition: str) -> bool:
    """True when the 'side effect' is really the condition being treated (e.g. 'anxiety' in an Anxiety review)."""
    stem = term.split()[0][:6]
    return stem in str(condition).lower()


def side_effect_risk(df: pd.DataFrame, ents: List[List[tuple]], top: int = 15, min_count: int = 30) -> pd.DataFrame:
    """For each side effect: share of negative vs positive reviews mentioning it, and the odds ratio (95% CI).

    Mentions that match the review's own condition are dropped (in an Anxiety review, 'anxiety' is the reason for
    taking the drug, not a side effect). Without this, anxiety and depression look *protective*. Sorted by odds ratio;
    `linked_to_negative` is True only when the whole 95% CI is above 1."""
    neg, pos = (df.label == 0).to_numpy(), (df.label == 1).to_numpy()
    cond = df.condition.tolist() if "condition" in df else [""] * len(df)
    by, dropped = {}, Counter()
    for i, e in enumerate(ents):
        for se in {t for lab, t, *_ in e if lab in ("SIDE_EFFECT",)}:
            if _is_indication(se, cond[i]):
                dropped[se] += 1
                continue
            by.setdefault(se, np.zeros(len(df), bool))[i] = True
    rows = []
    for se, m in by.items():
        a, b = int((m & neg).sum()), int((~m & neg).sum())
        c, d = int((m & pos).sum()), int((~m & pos).sum())
        if a + c < min_count:
            continue
        orr = ((a + .5) * (d + .5)) / ((b + .5) * (c + .5))
        se_log = math.sqrt(1 / (a + .5) + 1 / (b + .5) + 1 / (c + .5) + 1 / (d + .5))
        lo, hi = math.exp(math.log(orr) - 1.96 * se_log), math.exp(math.log(orr) + 1.96 * se_log)
        rows.append({"side_effect": se, "mentions": a + c, "share_of_negative": round(a / max(neg.sum(), 1), 4),
                     "share_of_positive": round(c / max(pos.sum(), 1), 4), "odds_ratio_negative": round(orr, 2),
                     "or_ci95": [round(lo, 2), round(hi, 2)], "linked_to_negative": bool(lo > 1),
                     "dropped_as_indication": int(dropped[se])})
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    return out.sort_values("mentions", ascending=False).head(top).sort_values("odds_ratio_negative", ascending=False)


def entity_review_sample(df: pd.DataFrame, ents, n: int = 100, seed: int = 7) -> pd.DataFrame:
    """Random extracted entities with context, for a manual precision check (fill 'correct' with 1/0)."""
    rows = [{"review_index": int(i), "label": lab, "entity": t,
             "context": df.review.iloc[i][max(0, s - 60):e + 60], "correct": ""}
            for i, es in enumerate(ents) for lab, t, s, e in es]
    r = pd.DataFrame(rows)
    return r.sample(min(n, len(r)), random_state=seed) if len(r) else r


def summary(nlp, ents) -> Dict:
    c = Counter(lab for es in ents for lab, *_ in es)
    return {"model": nlp.meta.get("pfa_model"), "reviews": len(ents), "entities_by_label": dict(c),
            "reviews_with_side_effect": round(sum(any(l == "SIDE_EFFECT" for l, *_ in es) for es in ents)
                                              / max(len(ents), 1), 4)}
