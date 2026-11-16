"""Model Monitor ground-truth format helpers (no AWS dependencies, so they are unit-tested everywhere)."""
from datetime import datetime


def gt_record(inference_id: str, label: int) -> dict:
    """One JSON line per inference, matched to captured data on InferenceId."""
    return {"groundTruthData": {"data": str(int(label)), "encoding": "CSV"},
            "eventMetadata": {"eventId": inference_id}, "eventVersion": "0"}


def gt_key(prefix: str, now: datetime) -> str:
    return f"{prefix}/{now:%Y/%m/%d/%H}/ground_truth.jsonl"
