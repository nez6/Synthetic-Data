PY ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python)

.PHONY: install data train run api ui listener send mlflow samples test

install:
	$(PY) -m pip install -r requirements.txt

data:        ## generate synthetic patients and export FHIR Bundles
	$(PY) -m src.data_ingestion

train:       ## train from FHIR Bundles (generates data if missing)
	$(PY) -m src.train

run:         ## API + HL7 listener + dashboard, all at once
	./scripts/run_all.sh

api:         ## FastAPI on :8000 (also starts the HL7 MLLP listener on :2575)
	$(PY) -m uvicorn src.app:app --reload --port 8000

ui:
	$(PY) -m streamlit run interface/frontend.py

listener:    ## standalone HL7 listener (if not running the API)
	$(PY) -m src.hl7v2.listener

send:        ## send 5 random HL7 ORU^R01 messages to the listener
	$(PY) -m src.hl7v2.send --count 5

mlflow:
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

samples:
	$(PY) -m scripts.make_samples

test:
	$(PY) -m pytest -q
