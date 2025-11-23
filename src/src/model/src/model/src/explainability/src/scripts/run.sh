#!/usr/bin/env bash
set -e

# Create virtual env optional
python -m venv .venv || true
source .venv/bin/activate || true

pip install -r requirements.txt

# train a quick model on sample data
python -m src.model.train --data data/sample_transactions.csv

# run server
uvicorn src.api:app --host 0.0.0.0 --port 8000
