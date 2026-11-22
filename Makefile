.PHONY: install train api ui test mlflow

install:
	pip install -r requirements.txt

train:
	python -m src.train

api:
	uvicorn src.app:app --reload --port 8000

ui:
	streamlit run interface/frontend.py

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

test:
	pytest -q
