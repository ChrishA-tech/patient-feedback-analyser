"""End-to-end SageMaker run: train -> register (Model Registry, approval gate) -> real-time endpoint with data
capture -> Model Monitor (model quality) on reviews from conditions the model never saw -> clean up.

    python -m aws.run_sagemaker            # full run (~45-60 min, cleans up the endpoint at the end)
    python -m aws.run_sagemaker --cleanup  # delete endpoint / schedule / model if a run was interrupted

Resumable: finished steps are recorded in results/aws_state.json and skipped on a re-run.
Costs: billed per second for a training job (ml.m5.large), the endpoint (ml.m5.large, about an hour) and two
processing jobs (ml.m5.xlarge). Check Billing > Free Tier credits after the run."""
from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("SAGEMAKER_SUPPRESS_V2_WARNING", "1")
import boto3  # noqa: E402
from botocore.exceptions import ClientError  # noqa: E402

from pfa import config as C  # noqa: E402
from pfa import data  # noqa: E402

REGION = os.environ.get("AWS_REGION", "eu-west-2")
PREFIX = "patient-feedback"
ROLE_NAME = "pfa-sagemaker-execution-role"
GROUP = "patient-feedback-sentiment"
ENDPOINT = "patient-feedback-sentiment"
SCHEDULE = "patient-feedback-model-quality"
# Override if your account's default service quotas reject one of these (new accounts sometimes start at 0):
#   PFA_HOST_INSTANCE=ml.t2.medium python -m aws.run_sagemaker
TRAIN_INSTANCE = os.environ.get("PFA_TRAIN_INSTANCE", "ml.m5.large")
HOST_INSTANCE = os.environ.get("PFA_HOST_INSTANCE", "ml.m5.large")
MONITOR_INSTANCE = os.environ.get("PFA_MONITOR_INSTANCE", "ml.m5.xlarge")
MIN_VALID_F1 = 0.80          # approval gate: the model is only approved for deployment above this
N_TRAFFIC = 500              # reviews sent to the endpoint from unseen conditions
STATE = C.RESULTS / "aws_state.json"
HERE = Path(__file__).resolve().parent


def state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(**kw):
    s = state()
    s.update(kw)
    C.RESULTS.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2, default=str))
    return s


def gt_record(inference_id: str, label: int) -> dict:
    """Model Monitor ground-truth format (one JSON line per inference, matched on InferenceId)."""
    return {"groundTruthData": {"data": str(int(label)), "encoding": "CSV"},
            "eventMetadata": {"eventId": inference_id}, "eventVersion": "0"}


def gt_key(prefix: str, now: datetime) -> str:
    return f"{prefix}/{now:%Y/%m/%d/%H}/ground_truth.jsonl"


def role_arn(iam):
    try:
        return iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {
            "Service": "sagemaker.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
        arn = iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust),
                              Description="SageMaker execution role for patient-feedback-analyser")["Role"]["Arn"]
        iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess")
        print("created IAM role; waiting 15 s for it to propagate")
        time.sleep(15)
        return arn


def cleanup(sm=None):
    sm = sm or boto3.client("sagemaker", region_name=REGION)
    for fn, kw in [(sm.delete_monitoring_schedule, {"MonitoringScheduleName": SCHEDULE}),
                   (sm.delete_endpoint, {"EndpointName": ENDPOINT})]:
        try:
            fn(**kw)
            print("deleted", kw)
        except Exception as e:
            print("skip", kw, str(e)[:90])
    s = state()
    for fn, kw in [(sm.delete_endpoint_config, {"EndpointConfigName": s.get("endpoint_config", "")}),
                   (sm.delete_model, {"ModelName": s.get("hosted_model", "")})]:
        if list(kw.values())[0]:
            try:
                fn(**kw)
                print("deleted", kw)
            except Exception as e:
                print("skip", kw, str(e)[:90])
    save(cleaned_up_at=datetime.now(timezone.utc).isoformat())


def main():
    import sagemaker
    from sagemaker import ModelPackage
    from sagemaker.model_metrics import MetricsSource, ModelMetrics
    from sagemaker.model_monitor import (CronExpressionGenerator, DataCaptureConfig, EndpointInput,
                                         ModelQualityMonitor)
    from sagemaker.model_monitor.dataset_format import DatasetFormat
    from sagemaker.sklearn import SKLearn

    boto = boto3.Session(region_name=REGION)
    sess = sagemaker.Session(boto_session=boto)
    sm, s3, rt = boto.client("sagemaker"), boto.client("s3"), boto.client("sagemaker-runtime")
    bucket = sess.default_bucket()
    role = role_arn(boto.client("iam"))
    base = f"s3://{bucket}/{PREFIX}"
    st = save(region=REGION, bucket=bucket, role=role, account=boto.client("sts").get_caller_identity()["Account"])

    # 1. data
    data.download()
    splits = data.build_splits(data.load())
    if "train_uri" not in st:
        tmp = C.DATA / "sagemaker"
        tmp.mkdir(parents=True, exist_ok=True)
        splits["train"][["review", "label"]].to_csv(tmp / "train.csv", index=False)
        splits["valid"][["review", "label"]].to_csv(tmp / "valid.csv", index=False)
        st = save(train_uri=sess.upload_data(str(tmp), bucket, f"{PREFIX}/data"))

    try:
        # 2. training job
        if "training_job" not in st:
            est = SKLearn(entry_point="train.py", source_dir=str(HERE), framework_version="1.2-1", py_version="py3",
                          instance_type=TRAIN_INSTANCE, instance_count=1, role=role, sagemaker_session=sess,
                          base_job_name="patient-feedback", output_path=f"{base}/models",
                          metric_definitions=[{"Name": k, "Regex": f"{k}=([0-9.]+)"}
                                              for k in ("valid_macro_f1", "valid_accuracy", "valid_auc", "threshold")])
            est.fit({"train": st["train_uri"]}, wait=True, logs="None")
            d = sm.describe_training_job(TrainingJobName=est.latest_training_job.name)
            st = save(training_job=d["TrainingJobName"], model_data=d["ModelArtifacts"]["S3ModelArtifacts"],
                      training_billable_seconds=d.get("BillableTimeInSeconds"),
                      training_metrics={m["MetricName"]: round(m["Value"], 4) for m in d.get("FinalMetricDataList", [])})
            print("trained", st["training_metrics"])

        # 3. baseline + metrics from the training job's output.tar.gz
        if "baseline_uri" not in st:
            out_key = st["model_data"].replace("model.tar.gz", "output.tar.gz").split(f"s3://{bucket}/")[1]
            raw = s3.get_object(Bucket=bucket, Key=out_key)["Body"].read()
            with tarfile.open(fileobj=io.BytesIO(raw)) as t:
                files = {Path(m.name).name: t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
            s3.put_object(Bucket=bucket, Key=f"{PREFIX}/monitor/baseline.csv", Body=files["baseline.csv"])
            s3.put_object(Bucket=bucket, Key=f"{PREFIX}/models/metrics.json", Body=files["metrics.json"])
            st = save(baseline_uri=f"{base}/monitor/baseline.csv", metrics_uri=f"{base}/models/metrics.json",
                      valid_metrics=json.loads(files["metrics.json"]))

        # 4. Model Registry with an approval gate
        if "model_package_arn" not in st:
            try:
                sm.create_model_package_group(ModelPackageGroupName=GROUP,
                                              ModelPackageGroupDescription="Patient review sentiment (TF-IDF + LR)")
            except ClientError as e:
                if "already exists" not in str(e).lower():
                    raise
            model = SKLearn.attach(st["training_job"], sagemaker_session=sess).create_model(
                entry_point="train.py", source_dir=str(HERE), role=role)
            pkg = model.register(content_types=["application/json"], response_types=["application/json"],
                                 inference_instances=[HOST_INSTANCE, "ml.t2.medium"], transform_instances=[HOST_INSTANCE],
                                 model_package_group_name=GROUP, approval_status="PendingManualApproval",
                                 description="TF-IDF + logistic regression sentiment for patient drug reviews",
                                 model_metrics=ModelMetrics(model_statistics=MetricsSource(
                                     content_type="application/json", s3_uri=st["metrics_uri"])))
            st = save(model_package_arn=pkg.model_package_arn)
        if "approval" not in st:
            f1 = st["valid_metrics"]["valid_macro_f1"]
            status = "Approved" if f1 >= MIN_VALID_F1 else "Rejected"
            sm.update_model_package(ModelPackageArn=st["model_package_arn"], ModelApprovalStatus=status,
                                    ApprovalDescription=f"valid macro-F1 {f1:.4f} vs gate {MIN_VALID_F1}")
            st = save(approval=status)
            if status != "Approved":
                sys.exit(f"Model rejected by the approval gate (valid macro-F1 {f1:.4f} < {MIN_VALID_F1}); not deploying.")

        # 5. real-time endpoint with data capture
        if "endpoint_up" not in st:
            mp = ModelPackage(role=role, model_package_arn=st["model_package_arn"], sagemaker_session=sess)
            mp.deploy(initial_instance_count=1, instance_type=HOST_INSTANCE, endpoint_name=ENDPOINT, wait=True,
                      data_capture_config=DataCaptureConfig(enable_capture=True, sampling_percentage=100,
                                                            destination_s3_uri=f"{base}/capture",
                                                            json_content_types=["application/json"],
                                                            sagemaker_session=sess))
            ep = sm.describe_endpoint(EndpointName=ENDPOINT)
            cfg = sm.describe_endpoint_config(EndpointConfigName=ep["EndpointConfigName"])
            st = save(endpoint_up=datetime.now(timezone.utc).isoformat(), endpoint_config=ep["EndpointConfigName"],
                      hosted_model=cfg["ProductionVariants"][0]["ModelName"])

        # 6. Model Monitor baseline (model quality) from the validation predictions
        mq = ModelQualityMonitor(role=role, instance_count=1, instance_type=MONITOR_INSTANCE, volume_size_in_gb=20,
                                 max_runtime_in_seconds=1800, sagemaker_session=sess)
        if "baseline_job" not in st:
            job = mq.suggest_baseline(baseline_dataset=st["baseline_uri"], dataset_format=DatasetFormat.csv(header=True),
                                      problem_type="BinaryClassification", inference_attribute="prediction",
                                      probability_attribute="probability", ground_truth_attribute="label",
                                      output_s3_uri=f"{base}/monitor/baseline-output", wait=True, logs=False)
            st = save(baseline_job=job.job_name, constraints_uri=f"{base}/monitor/baseline-output/constraints.json",
                      baseline_statistics=_read_json(s3, bucket, f"{PREFIX}/monitor/baseline-output/statistics.json"))

        # 7. traffic from conditions never seen in training, with ground truth
        if "traffic_sent" not in st:
            sample = splits["unseen"].sample(min(N_TRAFFIC, len(splits["unseen"])), random_state=C.SEED)
            lines, lat, correct = [], [], 0
            for review, label in zip(sample.review, sample.label):
                iid = str(uuid.uuid4())
                t = time.perf_counter()
                r = rt.invoke_endpoint(EndpointName=ENDPOINT, ContentType="application/json", Accept="application/json",
                                       Body=json.dumps({"review": review}), InferenceId=iid)
                lat.append((time.perf_counter() - t) * 1000)
                pred = json.loads(r["Body"].read())
                correct += int(pred["predicted_label"] == int(label))
                lines.append(json.dumps(gt_record(iid, label)))
            now = datetime.now(timezone.utc)
            s3.put_object(Bucket=bucket, Key=gt_key(f"{PREFIX}/ground-truth", now), Body="\n".join(lines).encode())
            lat.sort()
            st = save(traffic_sent=now.isoformat(), traffic_n=len(lines), client_accuracy=round(correct / len(lines), 4),
                      latency_ms_p50=round(lat[len(lat) // 2], 1), latency_ms_p95=round(lat[int(0.95 * (len(lat) - 1))], 1))
            print("traffic sent; waiting 6 min for data capture files to land in S3")
            time.sleep(360)

        # 8. one-off monitoring job (model quality on the captured traffic)
        if "monitor_status" not in st:
            mq.create_monitoring_schedule(
                monitor_schedule_name=SCHEDULE,
                endpoint_input=EndpointInput(endpoint_name=ENDPOINT, destination="/opt/ml/processing/input_data",
                                             inference_attribute="predicted_label", probability_attribute="probability"),
                ground_truth_input=f"{base}/ground-truth", problem_type="BinaryClassification",
                output_s3_uri=f"{base}/monitor/reports", constraints=st["constraints_uri"],
                schedule_cron_expression=CronExpressionGenerator.now(),
                data_analysis_start_time="-PT2H", data_analysis_end_time="-PT0H", enable_cloudwatch_metrics=True)
            ex = _wait_execution(sm)
            report = ex.get("ProcessingJobArn", "").split("/")[-1]
            out = _execution_outputs(sm, s3, bucket, ex)
            st = save(monitor_status=ex["MonitoringExecutionStatus"], monitor_failure=ex.get("FailureReason"),
                      monitor_job=report, **out)
        print(json.dumps({k: st.get(k) for k in ("training_metrics", "approval", "client_accuracy", "monitor_status",
                                                 "monitor_metrics", "violations")}, indent=2, default=str))
    finally:
        if "--keep" not in sys.argv:
            cleanup(sm)
    summarise()


def _read_json(s3, bucket, key):
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except Exception:
        return None


def _wait_execution(sm, timeout_min=45):
    t = time.time()
    while time.time() - t < timeout_min * 60:
        ex = sm.list_monitoring_executions(MonitoringScheduleName=SCHEDULE).get("MonitoringExecutionSummaries", [])
        if ex and ex[0]["MonitoringExecutionStatus"] in ("Completed", "CompletedWithViolations", "Failed", "Stopped"):
            return ex[0]
        print("monitoring:", ex[0]["MonitoringExecutionStatus"] if ex else "scheduled")
        time.sleep(60)
    raise TimeoutError("monitoring execution did not finish")


def _execution_outputs(sm, s3, bucket, ex):
    if not ex.get("ProcessingJobArn"):
        return {}
    job = sm.describe_processing_job(ProcessingJobName=ex["ProcessingJobArn"].split("/")[-1])
    uri = job["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
    key = uri.split(f"s3://{bucket}/")[1]
    stats = _read_json(s3, bucket, f"{key}/statistics.json") or {}
    viol = _read_json(s3, bucket, f"{key}/constraint_violations.json") or {}
    m = stats.get("binary_classification_metrics", {})
    return {"monitor_metrics": {k: round(v.get("value"), 4) for k, v in m.items() if isinstance(v, dict) and "value" in v},
            "violations": [{"metric": v.get("metric_name"), "description": v.get("description")}
                           for v in viol.get("violations", [])],
            "monitor_billable_note": "processing job on " + MONITOR_INSTANCE}


def summarise():
    s = state()
    keep = ["region", "training_job", "training_billable_seconds", "training_metrics", "valid_metrics",
            "model_package_arn", "approval", "traffic_n", "client_accuracy", "latency_ms_p50", "latency_ms_p95",
            "baseline_statistics", "monitor_status", "monitor_failure", "monitor_metrics", "violations",
            "cleaned_up_at"]
    (C.RESULTS / "aws.json").write_text(json.dumps({k: s.get(k) for k in keep}, indent=2, default=str))
    print("wrote results/aws.json")


if __name__ == "__main__":
    if "--cleanup" in sys.argv:
        cleanup()
    else:
        try:
            main()
        except ClientError as e:
            if "ResourceLimitExceeded" in str(e):
                print("\nAWS quota: your account's limit for this instance type is too low. Either request an increase "
                      "(Service Quotas > Amazon SageMaker) or re-run with another type, e.g.\n"
                      "  PFA_HOST_INSTANCE=ml.t2.medium PFA_TRAIN_INSTANCE=ml.m5.xlarge python -m aws.run_sagemaker\n"
                      "Finished steps are skipped on the re-run.")
            raise
