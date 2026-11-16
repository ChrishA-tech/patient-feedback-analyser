"""After filling the 'correct' column (1/0) in results/entity_sample.csv: python -m pfa.entity_precision"""
import json

import pandas as pd

from . import config as C

d = pd.read_csv(C.RESULTS / "entity_sample.csv")
d = d[d.correct.astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])]
if d.empty:
    raise SystemExit("Fill the 'correct' column with 1 (right entity and label) or 0 first.")
ok = d.correct.astype(float).astype(int)
out = {"checked": int(len(d)), "precision": round(float(ok.mean()), 3),
       "by_label": {k: round(float(v), 3) for k, v in ok.groupby(d.label).mean().items()}}
(C.RESULTS / "entity_precision.json").write_text(json.dumps(out, indent=2))
print(out)
