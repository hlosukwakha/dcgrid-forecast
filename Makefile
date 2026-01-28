SHELL := /bin/bash

.PHONY: help venv fmt lint test data train serve compose-up compose-down k3s-up k3s-down k3s-bootstrap k3s-deploy k3s-status

help:
	@echo "Targets:"
	@echo "  venv           Create local venv and install deps"
	@echo "  data           Download + build feature table"
	@echo "  train          Train model (XGBoost or TFT)"
	@echo "  serve          Run FastAPI locally"
	@echo "  compose-up     Local stack (postgres+minio+mlflow+api+dashboard)"
	@echo "  compose-down   Stop local stack"
	@echo "  k3s-up         Start single-node Kubernetes (k3s) in Docker"
	@echo "  k3s-bootstrap  Install ingress + namespace + secrets"
	@echo "  k3s-deploy     Deploy workloads to k3s"
	@echo "  k3s-status     kubectl get pods -A"

venv:
	python -m venv .dcgrid
	. .dcgrid/bin/activate && pip install -U pip && pip install -r requirements.txt

data:
	. .dcgrid/bin/activate && python -m dcgrid.scripts.ingest

train:
	. .dcgrid/bin/activate && python -m dcgrid.scripts.train

serve:
	. .dcgrid/bin/activate && uvicorn dcgrid.api.main:app --host $${API_HOST:-0.0.0.0} --port $${API_PORT:-8000} --reload

compose-up:
	docker compose -f infra/docker-compose.local.yaml up -d --build

compose-down:
	docker compose -f infra/docker-compose.local.yaml down -v

k3s-up:
	docker compose -f infra/k3s/docker-compose.k3s.yaml up -d

k3s-down:
	docker compose -f infra/k3s/docker-compose.k3s.yaml down -v

k3s-bootstrap:
	./infra/k3s/bootstrap.sh

k3s-deploy:
	kubectl apply -f deploy/k8s/

k3s-status:
	kubectl get pods -A
