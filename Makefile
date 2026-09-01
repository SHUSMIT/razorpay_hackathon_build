# Linux/macOS convenience wrapper. On Windows use: python run.py <target>
PY ?= python

.PHONY: setup data tune train assess context api app test all clean

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.prepare

tune:
	$(PY) -m src.tune --trials 30

train:
	$(PY) -m src.train

assess:
	$(PY) -m src.assess

context:
	$(PY) -m src.context

api:
	$(PY) -m uvicorn src.api:app --host 127.0.0.1 --port 8000

app:
	$(PY) -m streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -q tests

all: data tune train assess context test

clean:
	rm -rf data/processed/*.parquet models/*.json models/*.cbm models/*.joblib \
	       reports/*.png reports/*.json reports/*.csv \
	       reports/audit_log.jsonl reports/review_queue.jsonl
