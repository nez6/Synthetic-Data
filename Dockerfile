# Synthetic Test Data Creator: one image for the whole project.
#
#   docker build -t stdc .
#   docker run --rm -p 8080:8080 stdc            # serve: API + HL7 listener; on first start it downloads the
#                                                # Synthea pools from Google Cloud Storage and trains a model
#   docker run --rm -v "$PWD/data:/app/data" stdc train   # data ingestion from GCS → pipeline → MLflow run
#   docker run --rm -p 8501:8501 -e API_URL=https://<api> stdc ui   # the Streamlit front end
#
# Cloud Run sets $PORT; the API listens on it (default 8080 here).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY config.yaml Makefile ./
COPY src ./src
COPY interface ./interface
COPY tools ./tools
COPY .streamlit ./.streamlit

RUN useradd --create-home --uid 1000 app && mkdir -p /app/data && chown -R app /app
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health',timeout=4)"

ENTRYPOINT ["python", "-m", "src.pipeline"]
CMD ["serve"]
