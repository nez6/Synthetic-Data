# Synthetic Test Data Creator

**Request a synthetic FHIR patient dataset that matches your population, validate it, train a model on it, and pull it from a deployed API.**

| | |
|---|---|
| **Live front end** | https://synthetic-data-ge.streamlit.app/ |
| **Live API** | https://stdc-api-ex2qikv6gq-ue.a.run.app ([interactive docs](https://stdc-api-ex2qikv6gq-ue.a.run.app/docs)) |
| **Dataset in the cloud** | [`gs://stdc-13890-data/synthea-pools/v1`](https://storage.googleapis.com/stdc-13890-data/synthea-pools/v1/manifest.json) (public, Google Cloud Storage) |

> All patients are synthetic (Synthea) and carry the HL7 `HTEST` security label. This is a teaching project, not a medical device.

Module Project 3 (Data/ML Engineering): a reproducible ML pipeline with cloud data storage, MLflow tracking, a Docker image, an API on Google Cloud Run and a Streamlit front end that calls it.

## Overview and goals

Healthcare teams need realistic test data that looks like *their* patients and *their* data-quality problems, without touching real records. This app:

1. **Profiles your data without sharing it.** You download `data_quality_profile.py`, run it on your own FHIR export and upload the resulting `profile.json`. It holds aggregate counts only, and small cells are suppressed.
2. **Selects matching Synthea patients.** It picks patients from pre-built pools by cohort and demographic quota, then injects your data-quality faults at your rates.
3. **Validates and repairs them.** Checks V01–V08 cover US Core, references, codes, dates and resource volumes. An independent audit and a statistical acceptance test follow.
4. **Trains a model.** It splits the data 60/20/20 and learns whether a record passes validation from **demographics only**.
5. **Delivers the dataset.** You get HTEST-labelled FHIR R4 (NDJSON), CSV, a zip with HL7 v2 messages, or one patient in any FHIR version.

```
GCS bucket ──download──▶ Synthea pools ─▶ select (cohort + demographic quota) ─▶ inject faults
   (00)                     (02)                                                      │
profile.json (01) ───────────────────────────────────────────────────────────────────┘
                                   ▼
                     validate + repair, audit (03) ─▶ ML 60/20/20 + MLflow ─▶ model_*.pkl
                                   ▼                                              ▼
                       delivery set (04)  ◀──────────  FastAPI on Cloud Run  ──▶ /predict
                                                         ▲
                                     Streamlit front end (Streamlit Community Cloud)
```

## Dataset

- **Source:** [Synthea](https://synthea.mitre.org) v4.0.0 ([GitHub](https://github.com/synthetichealth/synthea)), MITRE's open-source synthetic patient generator (Apache 2.0). We ran it with fixed seeds, Massachusetts demographics and 10 years of history.
- **Pools:** Pool **A** has 639 people aged 18–64 (seed 20261102). Pool **B** has 2,478 people aged 50–90 (seed 20261103). Each patient is one FHIR R4 Bundle, compacted to the clinical resource types in `config.yaml → synthea.keep_types`.
- **Cleaning:** Synthea writes about 9 MB per older patient, mostly billing resources. `src/pools.py` drops resource types we don't use, removes references that would dangle, and gzips each bundle.
- **Hosting:** each pool is a tar.gz in a public Google Cloud Storage bucket, next to `manifest.json` with SHA-256 checksums. `src/cloud_data.py` downloads and verifies them; that's step 00 of the pipeline.

## The ML task

| | |
|---|---|
| Task | Binary classification: does a patient record pass validation? |
| Features | `age`, `sex`, `race_ethnicity`, `cohort` (demographics only; the model never sees clinical values) |
| Targets | `passed_before` (**raw** model, data as generated) and `passed_after` (**validated** model, data as delivered) |
| Split | Per cohort: 60% train, 20% test (hyperparameter choice), 20% validation (final score). Ratios are set in `config.yaml → ml.split`, or per request |
| Candidates | Logistic regression (C = 0.1, 1, 10) and random forest (depth 3, 6; 200 trees), with one-hot encoding plus scaling or imputation |
| Selection metric | ROC AUC on the test split |
| Reported | Accuracy, balanced accuracy, ROC AUC and predicted vs actual pass rate on the validation split, plus permutation importance per feature |

**Why demographics only?** If a demographic group fails validation more often, a model trained on demographics can learn it. An ROC AUC well above 0.5 is a warning that data quality depends on who the patient is.

**Results from the default request** (363 cohort A patients and 132 cohort B patients tested):

| Model | Chosen | Validation accuracy | Balanced accuracy | ROC AUC |
|---|---|---|---|---|
| Raw (before validation) | constant "fail" | 100% | n/a | n/a |
| Validated (after validation) | random forest, depth 3 | 90.0% | 86.5% | 0.901 |

**What the difference means:**
- **Raw model:** every raw Synthea record fails at least one check, so there's nothing to learn and the model predicts the constant "fail".
- **Validated model:** after validation, 91% of cohort A passes but only 11% of cohort B does. Cohort B patients are older, have complex care and have more resources than the targets allow. So the model predicts mostly from `cohort` and `age`, which shows up in the ML tab's importance chart.

That finding is about the data, not about the model: the cohort B targets are hard to meet. The assignment doesn't grade the model's accuracy, so it's kept simple.

## Experiment tracking (MLflow)

Every run logs to the MLflow experiment `synthetic_test_data_ml`:
- **Parent run:** request name, delivery size, features, split ratios and patients per cohort; final and baseline pass rates per cohort; and the artifacts `pass_rates.png`, `ml_report.json` and `ml_dataset.csv`.
- **Nested runs `raw` and `validated`:** target, chosen model and its hyperparameters; accuracy, balanced accuracy and ROC AUC on the train, test and validation splits; and the pickled model.

```bash
make mlflow        # http://localhost:5000
```

## Cloud services used

| Service | Used for |
|---|---|
| **Google Cloud Storage** (`gs://stdc-13890-data`) | Hosts the cleaned dataset (Synthea pools + checksum manifest), public-read |
| **Google Cloud Build** | Builds the Docker image from this repo on `gcloud run deploy --source .` |
| **Google Artifact Registry** | Stores the built image |
| **Google Cloud Run** (`stdc-api`, us-east1) | Runs the API container, publicly accessible |
| **Streamlit Community Cloud** | Hosts the front end; it calls the Cloud Run API over HTTPS (`API_URL` secret) |

## Setup and usage (local)

```bash
git clone https://github.com/nez6/Synthetic-Data.git && cd Synthetic-Data
python3 -m venv .venv && source .venv/bin/activate
make install          # pip install -r requirements.txt
make data-download    # 00 · fetch + verify the Synthea pools from Google Cloud Storage
```

### Train the model
```bash
make train            # = python -m src.pipeline train
```
This downloads the data if needed, then runs the default request: selection, validation, the 60/20/20 split, model training and MLflow logging. Results land in `data/runs/<run_id>/` (the `ml/` folder holds both models). To train on a custom request instead, use the **📝 Request** tab or `POST /runs`.

### Build and test the API locally
```bash
make run              # API on :8000 (+ HL7 listener on :2575) and dashboard on :8501
make test             # 63 tests: validation, FHIR versions, HL7, cloud download, API endpoints

curl localhost:8000/health
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"age": 72, "sex": "female", "race_ethnicity": "Hispanic/Latino", "cohort": "B"}'
# → {"run_id": "...", "model": "validated", "probability_pass": 0.04, "predicted_pass": false, ...}
```

### Run everything in Docker
```bash
docker build -t stdc .
docker run --rm -v "$PWD/data:/app/data" stdc train                 # training entry point
docker run --rm -p 8080:8080 stdc serve                             # serving entry point → http://localhost:8080/docs
docker run --rm -p 8501:8501 -e API_URL=http://host.docker.internal:8080 stdc ui   # front end
```
On first start, `serve` downloads the data from Cloud Storage and trains a model in the background. `/health` reports its progress (`bootstrap.stage`: downloading → training → ready).

## Deploy to the cloud

**1. Data (once, or after rebuilding the pools).** This needs `gcloud` signed in to a project with billing enabled.
```bash
gcloud storage buckets create gs://<bucket> --location us-east1 --uniform-bucket-level-access
gcloud storage buckets add-iam-policy-binding gs://<bucket> --member=allUsers --role=roles/storage.objectViewer
make data-upload      # set cloud.bucket in config.yaml first
```

**2. API on Cloud Run.** This builds the Dockerfile with Cloud Build.
```bash
gcloud services enable run.googleapis.com storage.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
make deploy-api       # gcloud run deploy stdc-api --source . --region us-east1 --allow-unauthenticated \
                      #   --memory 8Gi --cpu 2 --no-cpu-throttling --max-instances 1 --timeout 900
curl -X POST https://<service-url>/predict -H 'content-type: application/json' -d '{"age": 45, "sex": "male", "cohort": "A"}'
```
- `--max-instances 1`: runs and jobs live in the container, so a single instance keeps them consistent.
- `--no-cpu-throttling`: lets background runs keep working between requests.

**3. Front end on Streamlit Community Cloud.**
1. Go to [share.streamlit.io](https://share.streamlit.io) and choose **Create app**.
2. Pick this repo and branch, with main file `interface/frontend.py`.
3. Under **Advanced settings**, choose Python 3.12 and set the secret `API_URL = "https://<service-url>"`.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | JSON `{age, sex, race_ethnicity, cohort, model?, run_id?}` → probability the record passes validation |
| GET | `/health` | Pools, runs, start-up state, HL7 listener |
| GET | `/tools/data_quality_profile.py` | Download the profiling script |
| GET | `/requests/default` · POST `/requests/from-profile` | Default request / request filled from a profile |
| POST | `/runs` → GET `/jobs/{id}` | Submit a request (runs in the background) and poll it |
| GET | `/runs`, `/runs/{id}`, `/runs/{id}/full` | Results: cohort scores, checks, per-patient records, ML report |
| GET | `/runs/{id}/dataset?format=ndjson\|csv\|zip` | Download the delivered dataset |
| GET | `/runs/{id}/patients/{pid}?fhir_version=R5` | One delivered patient in DSTU2, STU3, R4, R4B or R5 |
| GET | `/runs/{id}/plot.png`, `/runs/{id}/delivered-records`, `/runs/{id}/patients/{pid}/before-after` | Data for the dashboard |
| POST | `/pools/eligible` · GET `/pools/profile` | Eligible patients per cohort / demo profile of the pools |
| POST | `/hl7/send` · GET/DELETE `/hl7/inbox` | Relay an ORU^R01 to the MLLP listener over HTTPS; read or clear its inbox |
| GET/POST | `/fhir/{version}/metadata`, `/fhir/{version}/$convert` | CapabilityStatement; convert a Bundle to another FHIR version |

## Project structure

```
config.yaml                 pipeline configuration: cloud bucket, split ratios, quality bounds, acceptance, Synthea pools
Dockerfile                  one image; entry points: train · serve · ui
Makefile                    install · data-download · train · run · test · docker-* · deploy-api
src/
  cloud_data.py             00 package / upload / download + verify the dataset (GCS)
  pipeline.py               train · serve · ui entry points; start-up download and training
  request.py                01 the request (cohorts, targets, demographics, faults, split)
  pools.py, selection.py    02 Synthea pools; eligibility, demographic quotas, fault injection
  stdc/                     03 validation V01–V08 and repairs, independent audit, statistics
  runner.py                 01–04 one request end to end; run storage and exports
  mlpipe.py                 03 split 60/20/20, train and tune, evaluate, MLflow logging
  inference.py              load a run's model and predict
  app.py                    FastAPI service
  fhir/, hl7v2/             FHIR multi-version builder/parser/validator; HL7 v2 + MLLP listener
interface/
  frontend.py, views/       Streamlit dashboard (one module per tab)
  client.py                 HTTP client: the dashboard reads and writes everything through the API
tools/data_quality_profile.py   the script users run on their own data (standard library only)
tests/                      pytest suite and 12 Synthea fixture patients
.github/workflows/ci.yml    tests on every push and pull request
```

## Ethics, privacy and limitations

- **Privacy:** no real patient data is used or stored. The profiling script runs on the user's system and returns aggregate counts only. Counts below 11 are suppressed, and fewer than 11 patients is refused. Every delivered record carries a synthetic identifier and the HL7 `HTEST` label, and records without the synthetic identifier are never modified.
- **Representativeness:** Synthea simulates Massachusetts demographics and disease models. Quotas can match a requested race/ethnicity mix only where the pool has enough eligible patients, and the shortfall is reported per group. Synthetic patients don't reproduce the documentation gaps, coding habits or care inequities of a real EHR. That's why faults are injected explicitly.
- **Bias check, not bias fix:** the demographics-only model is there to *detect* whether data quality depends on who the patient is. A high ROC AUC (as for cohort B here) means the delivered set under-represents some groups. Downstream tests could then miss failures that affect them.
- **The model is deliberately simple** and isn't a clinical tool. Predictions describe synthetic data quality, never a real person.
- **Operational limits:** runs live in the Cloud Run container, so they're lost when the instance is recycled (a new one re-downloads the data and retrains). The HL7 MLLP port is only reachable inside the container, which is why `POST /hl7/send` relays to it over HTTPS.

## Verification

Checked on 24 Nov 2026:

| Check | Result |
|---|---|
| Tests (`make test`) | 63 passed |
| `docker build` from a fresh clone of `main` (Google Cloud Shell, Docker 27) | Built in 161 s; `docker run … serve` started, downloaded the pools from GCS and served `/health` |
| Cloud Run (same Dockerfile, built by Cloud Build) | Start-up downloads the data and trains in ~6 min; `/health` reports `ready` |
| Live `/predict` | `{"age":72,"sex":"female","race_ethnicity":"Hispanic/Latino","cohort":"B"}` → `{"model":"validated","probability_pass":0.0411,"predicted_pass":false}`. Same value as a local run, so the pipeline is reproducible |
| Live front end | Sidebar shows the API as reachable; the ML tab's Predict calls the deployed `/predict` |

> The API scales to zero after ~15 idle minutes. The first request afterwards waits ~6 minutes while it re-downloads the data and retrains; the front end's sidebar shows the progress.

Docker Desktop doesn't run on the author's Mac (macOS 13), so the local container test was done in Google Cloud Shell. Its `serve` run was cut off by a Cloud Shell sign-out during the 577 MB download; the full serve path, from data download through training to `/predict`, is shown by Cloud Run, which runs the same image.

## Collaboration

This is a solo project (no partner). Work happens on feature branches and is merged into `main` through pull requests:
- PR #1 (`feature/custom-dataset-creator`) holds the pipeline, cloud ingestion, Docker, deployment and front end.
- This verification section was added on `docs/verification`.

CI runs the tests on every PR.
