PY ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python)
IMAGE ?= stdc
PROJECT ?= stdc-13890
REGION ?= us-east1
SERVICE ?= stdc-api

.PHONY: install data-download data-upload train run api ui listener send mlflow test \
        docker-build docker-train docker-serve deploy-api

install:        ## Python dependencies
	$(PY) -m pip install -r requirements.txt

data-download:  ## 00 · fetch the Synthea pools from Google Cloud Storage (verified by SHA-256)
	$(PY) -m src.cloud_data download

data-upload:    ## maintainers: package data/pools and upload to gs://<bucket> (needs gcloud)
	$(PY) -m src.cloud_data upload

train:          ## download data if needed, run the default request: validate → 60/20/20 → models → MLflow
	$(PY) -m src.pipeline train

run:            ## API + HL7 listener + dashboard, all at once
	./scripts/run_all.sh

api:            ## FastAPI on :8000 (also starts the HL7 MLLP listener on :2575)
	$(PY) -m uvicorn src.app:app --reload --port 8000

ui:             ## Streamlit dashboard (API_URL=... to point it at a deployed API)
	$(PY) -m streamlit run interface/frontend.py

listener:       ## standalone HL7 listener (if not running the API)
	$(PY) -m src.hl7v2.listener

send:           ## send 5 random HL7 ORU^R01 messages to the listener
	$(PY) -m src.hl7v2.send --count 5

mlflow:         ## MLflow UI on :5000
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

test:
	$(PY) -m pytest -q

docker-build:
	docker build -t $(IMAGE) .

docker-train:   ## training entry point in the container (data and MLflow land in ./data)
	docker run --rm -v "$(PWD)/data:/app/data" $(IMAGE) train

docker-serve:   ## serving entry point in the container, on http://localhost:8080
	docker run --rm -p 8080:8080 $(IMAGE) serve

deploy-api:     ## build with Cloud Build and deploy to Cloud Run (see README → Deploy)
	gcloud run deploy $(SERVICE) --source . --project $(PROJECT) --region $(REGION) \
	  --allow-unauthenticated --memory 8Gi --cpu 2 --no-cpu-throttling \
	  --min-instances 0 --max-instances 1 --timeout 900
