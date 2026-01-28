# DCGrid Forecast: Data-centre load growth + grid stress indicators

Goal: **forecast data-centre load growth signals and flag grid stress before it becomes obvious in planning docs.**

This repo is **Ireland-first** (because Ireland has unusually transparent public data on data-centre electricity demand), but the pipeline is designed to generalise to other regions.

## What you get
- A reproducible pipeline that produces an **hourly feature table**:
  - system demand (MW)
  - renewables (wind/solar) + variability
  - day-ahead price spikes and **negative-price frequency** (optional)
  - a **data-centre load proxy** derived from official quarterly DC consumption
  - a transparent **stress_score** indicator
- Models:
  - **XGBoost** baseline (default)
  - **Temporal Fusion Transformer (TFT)** deep-learning model (optional)
  - **Prophet** daily baseline (optional)
- Serving + observability:
  - **FastAPI** `/forecast` and `/stress/alerts`
  - **Streamlit** dashboard
  - **MLflow** tracking (optional; included in local compose)
  - MinIO (S3) for model artifacts
- Deployment:
  - Local dev stack via Docker Compose
  - A **single-node Kubernetes cluster in a Docker container (k3s)** + manifests

## Public data used (core)
- **CSO Ireland**: quarterly metered electricity consumption by data centres (2015–2023)  
- **EirGrid**: quarter-hourly system data spreadsheet (demand + wind + solar)  
- **Ember (optional)**: hourly day-ahead prices for European countries

See: `docs/DATA_SOURCES.md`.

## Quickstart (local, no Kubernetes)
```bash
cp .env.example .env
make venv
make compose-up

# build features and train model
docker compose -f infra/docker-compose.local.yaml run --rm ingest
docker compose -f infra/docker-compose.local.yaml run --rm train

# API: http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

## Quickstart (Kubernetes-in-Docker via k3s)
```bash
cp .env.example .env

# Start k3s (single node) inside Docker
make k3s-up

# Build local images
./infra/k3s/build-images.sh

# Point kubectl at the generated kubeconfig
export KUBECONFIG="$(pwd)/infra/k3s/kubeconfig/kubeconfig.yaml"

# Bootstrap + deploy
make k3s-bootstrap
make k3s-deploy

kubectl -n dcgrid get pods
```

## How the data-centre load proxy works (important)
The CSO provides **quarterly data-centre electricity consumption (GWh)**.  
We compute, per quarter:

`dc_share = dc_gwh / total_system_gwh_in_quarter`

Then for each hour:

`dc_load_mw ≈ demand_mw * dc_share`

This is a **proxy**, not a metered DC load time-series. It’s still useful for:
- growth tracking (quarter-to-quarter)
- correlating stress events (prices/renewables) with inferred DC share

If you have direct substation / metered DC load data, swap it in and the rest of the pipeline stays the same.

## Next steps
- Add ENTSO-E ingestion for other bidding zones (requires API key).
- Replace the proxy with **direct DC load measurements** (where available).
- Add a congestion proxy (flows vs NTC, redispatch, curtailment) if you have data.
