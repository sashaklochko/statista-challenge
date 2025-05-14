.PHONY: prepare-env run-server query-api ready clean load-data run-test

# Removes the existing virtualenv, creates a new one, install dependencies.
prepare-env:
	rm -rf .venv
	python3 -m venv .venv
	.venv/bin/pip install -U pip wheel
	.venv/bin/pip install -r requirements.txt

# Start the server locally
run-server:
	.venv/bin/uvicorn app.server:app --host 0.0.0.0 --port 9000 --reload

# Load data into Elasticsearch
load-data:
	.venv/bin/python scripts/insert_data.py

test-query:
	.venv/bin/python test_query.py

run-test:
	PYTHONPATH=. pytest app/tests/