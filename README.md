# dcgrid-forecast

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.1xx-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Postgres](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![MinIO](https://img.shields.io/badge/MinIO-S3%20compatible-C72E49?logo=minio&logoColor=white)](https://min.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.x-0194E2)](https://mlflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)](https://xgboost.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Forecasting](https://img.shields.io/badge/PyTorch%20Forecasting-TFT-purple)](https://pytorch-forecasting.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-k3s-326CE5?logo=kubernetes&logoColor=white)](https://k3s.io/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.x-D71F00)](https://www.sqlalchemy.org/)

![image](image.png)

Forecast **data-centre load** and **grid stress** (Ireland-first) using public grid/market data, store features in Postgres, track experiments in MLflow, persist model artifacts to MinIO (S3), expose predictions via FastAPI, and explore results in a Streamlit dashboard.

> Repo: `git@github.com:hlosukwakha/dcgrid-forecast.git`

---

## Project tree

> The exact tree may vary slightly as you iterate. This reflects the current layout implied by the Makefile + scripts.

```text
dcgrid-forecast/
├─ services/
│  └─ dcgrid/
│     ├─ api/
│     │  ├─ Dockerfile
│     │  └─ main.py                    # FastAPI app
│     ├─ dashboard/
│     │  ├─ Dockerfile
│     │  └─ app.py                     # Streamlit UI
│     ├─ models/
│     │  ├─ tft.py                     # Temporal Fusion Transformer training
│     │  └─ prophet_baseline.py         # Optional baseline
│     ├─ scripts/
│     │  ├─ ingest.py                  # Build feature table in Postgres
│     │  └─ train.py                   # Train + log + store model artifact
│     ├─ db.py                         # SQLAlchemy engine + schema init
│     ├─ storage.py                    # MinIO/S3 helpers (put_bytes, etc.)
│     └─ config.py                     # Reads env vars (REGION, MODEL_NAME, etc.)
├─ infra/
│  ├─ docker-compose.local.yaml        # Local stack (compose)
│  ├─ mlflow/
│  │  └─ Dockerfile                    # MLflow image with pg + s3 deps
│  └─ k3s/
│     ├─ docker-compose.k3s.yaml        # k3s-in-docker
│     ├─ kubeconfig/
│     │  └─ kubeconfig.yaml             # pinned kubeconfig
│     ├─ bootstrap.sh                   # generates + applies manifests
│     ├─ build-images.sh                # build + import :local images into k3s
│     └─ manifests/
│        ├─ generated/                  # generated YAML (source of truth for k3s)
│        └─ exported/                   # optional "single-file" export (if you keep it)
├─ deploy/
│  └─ k8s/                              # optional static manifests (if you keep them)
├─ requirements.txt
├─ Makefile
├─ .env                                # local config (not committed)
└─ README.md
```

---

## What this project does (blog-style)

AI workloads are power-hungry, and the constraints are physical: **grid capacity, price signals, and renewable variability**. This project turns that reality into a practical engineering pipeline:

1. **Ingest** public grid/system data (Ireland-first) into a structured feature table.
2. **Engineer features** that correlate with stress and dispatch constraints (demand, wind/solar, renewables share, price effects, volatility, ramp rate, etc.).
3. **Train forecasting models** that predict a data-centre load proxy and/or stress-relevant signals:
   - **XGBoost** for a strong, fast tabular baseline.
   - **Temporal Fusion Transformer (TFT)** for sequence modeling with known/unknown covariates.
4. **Track experiments** with **MLflow** (metrics, params, run history).
5. **Persist model artifacts** into **MinIO (S3-compatible)** so your API can load the latest model without baking it into the container image.
6. **Serve predictions** via **FastAPI** and visualize recent history + forecasts in **Streamlit**.

### Where it’s useful
- **Data-centre site selection and capacity planning** (what happens to load/stress under different demand profiles?).
- **Grid hosting capacity research** (stress and renewables variability).
- **Energy-aware scheduling prototypes** (what would you do if prices spike or renewables crash?).
- **Academic or hackathon projects** that need an end-to-end MLOps loop (DB → MLflow → artifact store → API → UI).

---

## Tech stack: what each component does

- **Postgres**: “source of truth” feature store for training + dashboard queries (e.g. `dcgrid.features_hourly`).
- **MinIO (S3)**: artifact store for trained models (e.g. `models/IE/tft.pkl`) and any intermediate datasets if you choose.
- **MLflow**: experiment tracking UI + run metadata (params/metrics), and optionally model registry later.
- **Python services**
  - **Ingest script**: downloads/reads raw data + computes hourly features → Postgres.
  - **Train script**: reads features → trains model → logs to MLflow → writes model artifact to MinIO.
- **FastAPI**: serves prediction endpoints + Swagger docs.
- **Streamlit dashboard**: quick exploration of features and forecasts.
- **Docker Compose**: easiest “single machine” full stack.
- **k3s (Kubernetes)**: reproducible “cluster-like” environment that runs the same stack and uses Jobs for ingest/train.

---

## Dataset + data-centre load proxy

### Primary dataset (Ireland-first)
The project uses public system data from **EirGrid** (e.g. the “System Data Qtr Hourly” Excel publication) and derives a clean hourly time series.

- Source typically configured via `EIRGRID_SYSTEM_DATA_URL` (Excel `.xlsx`).
- Ingest converts quarter-hourly (or other native frequency) to **hourly** features.

### How the data-centre load proxy works (conceptual)
If you don’t have real data-centre telemetry, you still need a *supervised target* that behaves like data-centre demand:

A common approach is to build a proxy that:
- scales with **system demand** (baseline correlation),
- responds to **price** and **renewable availability** (stress/price sensitivity),
- includes **smooth persistence** (data centres don’t usually jump like residential loads),
- optionally includes **a “stress” coupling** (when grid stress rises, you can model load shaping or curtailment).

In practice, you can compute something like:

- `dc_load_mw = base_share * demand_mw + alpha * ramp_rate_mw_h + beta * price_effect + noise`
- with clipping/smoothing and guardrails.

> Your `dcgrid.features_hourly` table already includes the target column `dc_load_mw`, along with stress-related features such as `stress_score`.

### Feature table
The dashboard and trainer typically read:

- `demand_mw`, `wind_mw`, `solar_mw`, `renew_mw`, `renewable_share`
- `price_eur_mwh`, `neg_price_freq_7d`, `price_spike_flag`
- `ramp_rate_mw_h`, `renew_var_24h`
- `dc_load_mw` (target / proxy)
- `stress_score`

---

## Quickstart

You can run the project in two ways:

1. **Docker Compose** (fastest, recommended for day-to-day dev)
2. **k3s in Docker** (more “cluster-like”; uses Jobs for ingest/train)

### 0) Clone the repo

```bash
git clone git@github.com:hlosukwakha/dcgrid-forecast.git
cd dcgrid-forecast
```

Create a local `.env`:

```bash
cp .env.example .env 2>/dev/null || true
# or create manually
```

Typical `.env` keys:
- `REGION=IE`
- `MODEL_NAME=xgb` or `tft`
- `FORECAST_HORIZON_H=168`
- `EIRGRID_SYSTEM_DATA_URL=...`

---

## Option A: Docker Compose workflow

### Start the stack

```bash
make compose-up
```

### Ingest data (build features table)

```bash
make data
```

### Train a model (runs inside the compose environment)

```bash
make train
```

### UIs (Compose)
Depending on your compose ports, you should have:

- **API docs**: `http://localhost:8080/api/docs`
- **MLflow**: `http://localhost:9090`
- **Dashboard**: `http://localhost:8080` (or `http://localhost:8501` if you expose Streamlit directly)

> If your dashboard is on `8501` locally, keep using that. If you reverse-proxy through a single port, `8080` may host the UI.

Stop everything:

```bash
make compose-down
```

---

## Option B: k3s workflow (cluster-like)

### Start k3s

```bash
make k3s-up
```

### Build + import images into k3s containerd
This avoids `ImagePullBackOff` for `dcgrid/*:local`.

```bash
make k3s-images
```

### Bootstrap (generate + apply manifests)
`bootstrap.sh` generates manifests into `infra/k3s/manifests/generated/` and applies them.

```bash
make k3s-bootstrap
```

### Run ingest + train as Jobs
```bash
make k3s-ingest
make k3s-train
```

### Check status
```bash
make k3s-status
make k3s-nodes
```

---

## Port-forwarding

### Why you need port-forwarding in k3s
k3s services are **ClusterIP** by default: they are reachable *inside the cluster* but not from your host OS browser. Port-forwarding creates a secure tunnel from your laptop to the service.

### Common port-forward commands

```bash
# API
kubectl --kubeconfig "$PWD/infra/k3s/kubeconfig/kubeconfig.yaml" -n dcgrid port-forward svc/dcgrid-api 8080:8000

# Dashboard
kubectl --kubeconfig "$PWD/infra/k3s/kubeconfig/kubeconfig.yaml" -n dcgrid port-forward svc/dcgrid-dashboard 8501:8501

# MLflow (service port 5000)
kubectl --kubeconfig "$PWD/infra/k3s/kubeconfig/kubeconfig.yaml" -n dcgrid port-forward svc/mlflow 9090:5000

# MinIO (S3 + console)
kubectl --kubeconfig "$PWD/infra/k3s/kubeconfig/kubeconfig.yaml" -n dcgrid port-forward svc/minio 9000:9000
kubectl --kubeconfig "$PWD/infra/k3s/kubeconfig/kubeconfig.yaml" -n dcgrid port-forward svc/minio 9001:9001
```

---

## Testing / sanity checks

### Verify kube context is pinned (avoid “old context”)
Your Makefile pins kubeconfig with `--kubeconfig ...`. You can still sanity-check:

```bash
kubectl --kubeconfig "$PWD/infra/k3s/kubeconfig/kubeconfig.yaml" get nodes
kubectl config get-contexts   # may show something else; ignore if you always pass --kubeconfig
```

### Check services + DNS inside cluster
```bash
kubectl --kubeconfig "$PWD/infra/k3s/kubeconfig/kubeconfig.yaml" -n dcgrid get svc
kubectl --kubeconfig "$PWD/infra/k3s/kubeconfig/kubeconfig.yaml" -n dcgrid get endpoints
```

### Check Postgres table exists + row count
```bash
kubectl --kubeconfig "$PWD/infra/k3s/kubeconfig/kubeconfig.yaml" -n dcgrid exec -it deploy/postgres --   psql -U dcgrid -d dcgrid -c "\dt dcgrid.*"

kubectl --kubeconfig "$PWD/infra/k3s/kubeconfig/kubeconfig.yaml" -n dcgrid exec -it deploy/postgres --   psql -U dcgrid -d dcgrid -c "select count(*) from dcgrid.features_hourly;"
```

### Check model artifacts in MinIO
If you have a MinIO client or browse the console (9001), look for:
- `models/IE/xgb.pkl` or `models/IE/tft.pkl`

### Validate manifests
```bash
kubectl --kubeconfig "$PWD/infra/k3s/kubeconfig/kubeconfig.yaml" apply --dry-run=client -f infra/k3s/manifests/generated/
```

---

## Config files: what they’re for

- `.env`  
  Runtime configuration for region/model/horizon and secrets (local dev).

- `requirements.txt`  
  Python dependencies for host dev and container builds.

- `infra/docker-compose.local.yaml`  
  Local dev stack (postgres + minio + mlflow + api + dashboard + jobs).

- `infra/mlflow/Dockerfile`  
  MLflow image with the extras needed for:
  - Postgres backend store (`psycopg2-binary`)
  - S3/MinIO artifact store (`boto3`)

- `infra/k3s/docker-compose.k3s.yaml`  
  Runs a k3s “cluster” in Docker.

- `infra/k3s/kubeconfig/kubeconfig.yaml`  
  Pinned kubeconfig so commands don’t accidentally use another cluster context.

- `infra/k3s/build-images.sh`  
  Builds local `dcgrid/*:local` images and imports them into k3s containerd.

- `infra/k3s/bootstrap.sh`  
  Generates + applies k3s manifests into `infra/k3s/manifests/generated/`.

- `infra/k3s/manifests/generated/*.yaml`  
  Generated manifests (namespace, secrets, deployments, services, jobs, PVCs).

- `deploy/k8s/*.yaml` (optional)  
  Static manifests. If you keep generated manifests as the single source of truth,
  consider removing or clearly marking this directory to avoid duplication.

---

## Scripts, queries, and YAMLs (high level)

### Scripts
- `dcgrid.scripts.ingest`
  - Downloads/reads raw system data
  - Computes derived features
  - Ensures schema (creates `dcgrid` schema + tables)
  - Inserts/updates `dcgrid.features_hourly`

- `dcgrid.scripts.train`
  - Loads features with SQL
  - Trains selected model (`MODEL_NAME`)
  - Logs params/metrics to MLflow
  - Saves pickled artifact to MinIO under `models/{REGION}/{MODEL_NAME}.pkl`

### Key SQL query
Training reads features similar to:

```sql
SELECT ts_utc, demand_mw, wind_mw, solar_mw, renew_mw, renewable_share,
       price_eur_mwh, neg_price_freq_7d, price_spike_flag, ramp_rate_mw_h, renew_var_24h,
       dc_load_mw, stress_score
FROM dcgrid.features_hourly
WHERE region = :r
ORDER BY ts_utc ASC;
```

### YAML files
Generated manifests typically include:
- `00-namespace.yaml`: namespace
- `10-secret.yaml`: env secrets (db + minio)
- `20-postgres.yaml`: PVC + Deployment + Service
- `30-minio.yaml`: PVC + Deployment + Service
- `40-api.yaml`: API Deployment + Service
- `50-dashboard.yaml`: Dashboard Deployment + Service
- `60-job-ingest.yaml`: ingest Job (waits for Postgres)
- `70-job-train.yaml`: training Job (waits for Postgres/MinIO/MLflow)
- `80-mlflow.yaml`: MLflow Deployment + Service (recommended)

---

## Makefile: how to use it

Common flows:

### Local dev (compose)
```bash
make compose-up
make data
make train
```

### k3s (cluster-like)
```bash
make k3s-up
make k3s-images
make k3s-bootstrap
make k3s-ingest
make k3s-train
```

### One-shot (k3s)
```bash
make k3s-run
```

---

## UIs

These are the primary endpoints you listed:

- **API docs**: `http://localhost:8080/api/docs`  
  Swagger/OpenAPI docs for FastAPI.

- **MLflow UI**: `http://localhost:9090`  
  Track experiments, params, metrics, and runs.

- **App UI**: `http://localhost:8080`  
  A landing UI depending on your routing:
  - Either Streamlit is exposed at 8080, or
  - 8080 serves API and dashboard is on 8501.

If your dashboard runs on 8501 (most common), use:

- **Streamlit**: `http://localhost:8501`

---

## Troubleshooting (common errors)

### 1) `ImagePullBackOff` for `dcgrid/*:local`
Cause: k3s can’t pull local tags from Docker Hub.

Fix:
```bash
make k3s-images
kubectl --kubeconfig "$PWD/infra/k3s/kubeconfig/kubeconfig.yaml" -n dcgrid get pods
```

### 2) “Old context” / wrong cluster
Fix: always use the pinned kubeconfig (Makefile does this already):

```bash
export KUBECONFIG="$PWD/infra/k3s/kubeconfig/kubeconfig.yaml"
kubectl get nodes
```

### 3) Dashboard error: “relation dcgrid.features_hourly does not exist”
Cause: ingest job didn’t run or connected to the wrong Postgres (host vs cluster).

Fix:
- Run ingest Job in k3s: `make k3s-ingest`
- Port-forward the *correct* service if you’re testing locally.
- Verify table inside the cluster Postgres:
  ```bash
  kubectl --kubeconfig "$PWD/infra/k3s/kubeconfig/kubeconfig.yaml" -n dcgrid exec -it deploy/postgres --     psql -U dcgrid -d dcgrid -c "\dt dcgrid.*"
  ```

### 4) `connection refused` to Postgres from a Job
Cause: Postgres not ready yet or Service endpoints not wired.

Fix:
- Ensure Postgres pod is `Running` + endpoints exist:
  ```bash
  kubectl --kubeconfig "$PWD/infra/k3s/kubeconfig/kubeconfig.yaml" -n dcgrid get pods
  kubectl --kubeconfig "$PWD/infra/k3s/kubeconfig/kubeconfig.yaml" -n dcgrid get endpoints postgres
  ```
- Use initContainer `pg_isready` (recommended for Jobs).

### 5) MLflow service resolves but port-forward fails (“connection refused” inside pod)
Cause: MLflow container not listening on `0.0.0.0:5000` or it crashed early.

Fix:
```bash
kubectl --kubeconfig "$PWD/infra/k3s/kubeconfig/kubeconfig.yaml" -n dcgrid get pods
kubectl --kubeconfig "$PWD/infra/k3s/kubeconfig/kubeconfig.yaml" -n dcgrid logs deploy/mlflow --tail=200
kubectl --kubeconfig "$PWD/infra/k3s/kubeconfig/kubeconfig.yaml" -n dcgrid describe pod -l app=mlflow
```

### 6) TFT training errors about categoricals or NaNs
- TFT requires categoricals to be **string/categorified**, and **no NaNs** in real-valued features.
- Fix by:
  - converting `hour/dow/month` to strings (categoricals)
  - filling NaNs (and optionally adding `_is_missing` indicator columns)
  - dropping columns that are fully NA

---

## Extending the project

Ideas:
- Add more regions (GB, NL, DE) with region-specific data sources and unify via a `region` dimension.
- Add proper **model registry** (MLflow Model Registry or custom pointer in MinIO).
- Add a **scheduled pipeline** (CronJob in k3s) for daily ingest + retraining.
- Replace proxy `dc_load_mw` with real telemetry when available.
- Add stress/constraint labels (e.g., scarcity hours, curtailment flags) and train classification heads.

---

## Signature

Built by **@hlosukwakha**  
Cloud / Data / Observability / Energy-aware AI infrastructure
