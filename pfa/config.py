from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
UCI_URLS = [  # UCI has moved files over the years: try each, newest first
    "https://archive.ics.uci.edu/static/public/462/drug+reviews+drugs+com.zip",
    "https://archive.ics.uci.edu/static/public/462/drug+review+dataset+drugs+com.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip",
]
UCI_PAGE = "https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com"
SEED = 7

# Sentiment labels from the 10-star rating: >= 7 positive, <= 4 negative, 5-6 left out as ambiguous.
POS_MIN, NEG_MAX = 7, 4
# Conditions held out of training entirely, to test generalisation to a new data source.
N_UNSEEN_CONDITIONS, UNSEEN_MIN, UNSEEN_MAX = 20, 200, 3000
VALID_FRACTION = 0.1

NER_SAMPLE, TOPIC_SAMPLE = 20000, 15000
