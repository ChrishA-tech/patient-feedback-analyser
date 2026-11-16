"""Load the UCI Drugs.com reviews, clean them, label sentiment, remove leakage, and build the evaluation splits."""
from __future__ import annotations

import html
import io
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from . import config as C


def _extract(raw: bytes, out: Path) -> bool:
    """Unzip the TSVs (handles a zip nested inside the zip, as on some UCI mirrors)."""
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for n in z.namelist():
            if n.endswith(".tsv"):
                (out / Path(n).name).write_bytes(z.read(n))
            elif n.endswith(".zip"):
                _extract(z.read(n), out)
    return (out / "drugsComTrain_raw.tsv").exists()


def download(out: Path = None) -> Path:
    out = out or C.DATA
    out.mkdir(parents=True, exist_ok=True)
    if (out / "drugsComTrain_raw.tsv").exists():
        return out
    for z in sorted(out.glob("*.zip")) + sorted(Path.home().joinpath("Downloads").glob("drug*.zip")):
        if _extract(z.read_bytes(), out):          # a zip downloaded by hand
            return out
    for url in C.UCI_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            if _extract(urllib.request.urlopen(req, timeout=300).read(), out):
                print("downloaded", url)
                return out
        except Exception as e:
            print("failed", url, str(e)[:80])
    raise SystemExit(f"Could not download the dataset. Open {C.UCI_PAGE}, click Download, and put the zip in "
                     f"{out} (or leave it in ~/Downloads), then run again.")


def clean_text(t: str) -> str:
    t = html.unescape(str(t)).strip()
    t = t.strip('"').replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", t)


def load(folder: Path = None) -> pd.DataFrame:
    folder = folder or C.DATA
    parts = []
    for split in ("Train", "Test"):
        d = pd.read_csv(folder / f"drugsCom{split}_raw.tsv", sep="\t")
        d["source_split"] = split.lower()
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df = df.rename(columns={"drugName": "drug", "usefulCount": "useful_count"})
    df["review"] = df["review"].map(clean_text)
    df["condition"] = df["condition"].fillna("").astype(str).str.strip()
    # Scraping artefact: some 'conditions' are HTML fragments like '3</span> users found this comment helpful.'
    df.loc[df.condition.str.contains("</span>", regex=False) | (df.condition == ""), "condition"] = "Unknown"
    df["rating"] = df["rating"].astype(float)
    df["label"] = np.where(df.rating >= C.POS_MIN, 1, np.where(df.rating <= C.NEG_MAX, 0, -1))
    return df[["drug", "condition", "review", "rating", "label", "date", "useful_count", "source_split"]]


def build_splits(df: pd.DataFrame, seed: int = C.SEED) -> Dict:
    """train / valid / test (official test, seen conditions) / unseen (20 conditions never in training).

    The same review text often appears under several drug names (brand and generic), including across the
    official train and test files. Test or unseen rows whose text appears in training are removed (leakage)."""
    rng = np.random.default_rng(seed)
    lab = df[df.label >= 0].copy()
    counts = lab.condition[lab.condition != "Unknown"].value_counts()
    eligible = counts[(counts >= C.UNSEEN_MIN) & (counts <= C.UNSEEN_MAX)].index.to_numpy()
    unseen_conds = set(rng.choice(eligible, size=min(C.N_UNSEEN_CONDITIONS, len(eligible)), replace=False))
    is_unseen = lab.condition.isin(unseen_conds)

    pool = lab[~is_unseen & (lab.source_split == "train")]
    pool = pool.drop_duplicates("review")
    valid_mask = rng.random(len(pool)) < C.VALID_FRACTION
    train, valid = pool[~valid_mask], pool[valid_mask]
    valid = valid[~valid.review.isin(set(train.review))]

    seen_text = set(pool.review)
    test_all = lab[~is_unseen & (lab.source_split == "test")].drop_duplicates("review")
    test = test_all[~test_all.review.isin(seen_text)]
    unseen_all = lab[is_unseen].drop_duplicates("review")
    unseen = unseen_all[~unseen_all.review.isin(seen_text)]
    leak = {"test_rows_removed_as_duplicates_of_training": int(len(test_all) - len(test)),
            "unseen_rows_removed_as_duplicates_of_training": int(len(unseen_all) - len(unseen)),
            "duplicate_review_texts_in_full_dataset": int(df.review.duplicated().sum())}
    stats = {"reviews": int(len(df)), "labelled": int(len(lab)), "neutral_excluded": int((df.label < 0).sum()),
             "drugs": int(df.drug.nunique()), "conditions": int(df.condition.nunique()),
             "sizes": {k: int(len(v)) for k, v in (("train", train), ("valid", valid), ("test", test), ("unseen", unseen))},
             "positive_share_train": round(float(train.label.mean()), 3),
             "unseen_conditions": sorted(unseen_conds), **leak}
    return {"train": train, "valid": valid, "test": test, "unseen": unseen, "stats": stats}
