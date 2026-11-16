#!/usr/bin/env python3
"""One command on the Mac (system Python 3.9 is fine: standard library only).

  python3 run_and_push.py --local      # free: download data, train, entities, topics, memo, update README
  python3 run_and_push.py --aws        # SageMaker training/registry/endpoint/monitor run (~1 h), then clean-up
  python3 run_and_push.py --push-only  # commit + push to GitHub
"""
import base64
import getpass
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv312"
PY = VENV / "bin" / "python"
OWNER, REPO = "ChrishA-tech", "patient-feedback-analyser"
TRAILER = ("\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_01DVp71MJqKBK4sk34js4Jfb")
SCISPACY_MODEL = "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"


def sh(cmd, check=True):
    print("+", " ".join(map(str, cmd)))
    return subprocess.run(list(map(str, cmd)), cwd=ROOT, check=check).returncode


def uv():
    u = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
    if not Path(u).exists():
        sh(["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])
    return u


def setup(aws=False):
    u = uv()
    if not PY.exists():
        sh([u, "venv", "--python", "3.12", VENV])
    sh([u, "pip", "install", "--python", PY, "-r", "requirements.txt"] + (["-r", "requirements-aws.txt"] if aws else []))
    if not aws:
        # optional extras: the code falls back (and says so in the results) if they will not install
        if sh([u, "pip", "install", "--python", PY, "-r", "requirements-topics.txt"], check=False):
            print("BERTopic did not install: topics will use the NMF fallback")
        if sh([u, "pip", "install", "--python", PY, "scispacy==0.5.5", SCISPACY_MODEL], check=False):
            print("scispaCy did not install: entities will use the EntityRuler only")


def api(path, token, method="GET", body=None):
    req = urllib.request.Request("https://api.github.com" + path, method=method,
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"token {token}", "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {}


def push():
    sh([PY, "-m", "pfa.report"], check=False) if PY.exists() else None
    token = os.environ.get("GITHUB_TOKEN") or getpass.getpass("GitHub token (classic, repo scope; input hidden): ").strip()
    st, me = api("/user", token)
    if st != 200 or me.get("login") != OWNER:
        sys.exit(f"Token belongs to {me.get('login')!r}, expected {OWNER}. Stopping.")
    if api(f"/repos/{OWNER}/{REPO}", token)[0] == 404:
        api("/user/repos", token, "POST", {"name": REPO, "private": False, "description":
            "Patient drug reviews: sentiment (TF-IDF, TensorFlow), spaCy entities, BERTopic themes; AWS SageMaker "
            "training, Model Registry, endpoint and Model Monitor"})
        api(f"/repos/{OWNER}/{REPO}/topics", token, "PUT", {"names": [
            "nlp", "sentiment-analysis", "spacy", "tensorflow", "bertopic", "topic-modeling", "aws", "sagemaker",
            "mlops", "model-monitoring", "healthcare"]})
    if not (ROOT / ".git").exists():
        sh(["git", "init", "-b", "main"])
    sh(["git", "add", "-A"])
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode:
        sh(["git", "-c", "user.name=Chrish Pep Felix A", "-c", "user.email=chrishpepfelixa@gmail.com", "commit", "-m",
            "Patient feedback analyser: results" + TRAILER])
    auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    r = subprocess.run(["git", "-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {auth}", "push",
                        f"https://github.com/{OWNER}/{REPO}.git", "HEAD:main"], cwd=ROOT, capture_output=True, text=True)
    print((r.stdout + r.stderr).replace(token, "***").replace(auth, "***"))
    if r.returncode:
        sys.exit("Push failed (see above).")
    print(f"Done: https://github.com/{OWNER}/{REPO}\nNow delete the token at https://github.com/settings/tokens")


def main():
    a = set(sys.argv[1:])
    if "--local" in a:
        setup()
        sh([PY, "-m", "pytest", "-q"])
        sh([PY, "-m", "pfa.run_local"])
        sh([PY, "-m", "pfa.report"])
        print("\nLocal run done. Next: python3 run_and_push.py --aws   (or --push-only)")
    elif "--aws" in a:
        setup(aws=True)
        if sh([PY, "-c", "import boto3; print(boto3.client('sts').get_caller_identity()['Arn'])"], check=False):
            sys.exit("AWS credentials not set. Run `aws configure` (region eu-west-2) first.")
        sh([PY, "-m", "aws.run_sagemaker"])
        sh([PY, "-m", "pfa.report"])
        print("\nAWS run done and cleaned up. Next: python3 run_and_push.py --push-only")
    elif "--cleanup" in a:
        sh([PY, "-m", "aws.run_sagemaker", "--cleanup"])
    elif "--push-only" in a:
        push()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
