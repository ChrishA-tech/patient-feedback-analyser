import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

POS = ["This medication works great, my symptoms are gone and I feel amazing", "Best decision, life changing relief",
       "Very effective with no side effects at all, highly recommend", "Helped my pain a lot, very happy"]
NEG = ["Terrible nausea and headache, I stopped after a week", "Awful weight gain and anxiety, never again",
       "Did nothing for me and caused dizziness and insomnia", "Horrible acne and hair loss, worst drug ever"]


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    rng = np.random.default_rng(0)
    d = tmp_path_factory.mktemp("uci")
    conds = [f"Condition{i}" for i in range(30)]
    for split, n in (("Train", 3000), ("Test", 1000)):
        rows = []
        for i in range(n):
            pos = rng.random() < 0.7
            text = rng.choice(POS if pos else NEG) + f" (case {split}{i})"
            rows.append({"Unnamed: 0": i, "drugName": rng.choice(["Sertraline", "Levonorgestrel", "Metformin"]),
                         "condition": rng.choice(conds + ["3</span> users found this comment helpful."]),
                         "review": f'"{text} &#039;really&#039;"', "rating": rng.integers(7, 11) if pos else rng.integers(1, 5),
                         "date": "May 20, 2012", "usefulCount": int(rng.integers(0, 50))})
        # duplicate across files (brand/generic listing): leakage the splitter must remove
        if split == "Test":
            rows.append({**rows[0], "review": '"Terrible nausea and headache, I stopped after a week (case Train0) &#039;really&#039;"'})
        pd.DataFrame(rows).to_csv(d / f"drugsCom{split}_raw.tsv", sep="\t", index=False)
    return d


@pytest.fixture(scope="session")
def splits(raw_dir):
    from pfa import config as C, data
    C.UNSEEN_MIN, C.UNSEEN_MAX, C.N_UNSEEN_CONDITIONS = 50, 500, 3
    return data.build_splits(data.load(raw_dir))
