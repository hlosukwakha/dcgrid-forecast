SHELL := /bin/bash

# --- k3s kubeconfig pinned to this repo (prevents "old context") ---
KUBECONFIG_PATH := $(CURDIR)/infra/k3s/kubeconfig/kubeconfig.yaml
KUBECTL := kubectl --kubeconfig $(KUBECONFIG_PATH)

# --- local python ---
VENV := .dcgrid
PY := . $(VENV)/bin/activate && PYTHONPATH=$(CURDIR)/services

.PHONY: help venv fmt lint test \
        data data-host data-docker \
        train train-host train-docker \
        serve \
        compose-up compose-down \
        k3s-up k3s-down k3s-bootstrap k3s-deploy k3s-status k3s-nodes \
        k3s-images k3s-apply k3s-ingest k3s-train k3s-restart-api \
        k3s-run

help:
	@echo "Targets:"
	@echo "  venv             Create local venv and install deps"
	@echo "  data-host        Run ingest on host (expects Postgres reachable on localhost)"
	@echo "  data-docker      Run ingest in docker-compose stack"
	@echo "  train-host       Train on host (expects DB reachable on localhost)"
	@echo "  train-docker     Train in docker-compose stack"
	@echo "  serve            Run FastAPI locally (host)"
	@echo "  compose-up       Local stack (postgres+minio+mlflow+api+dashboard)"
	@echo "  compose-down     Stop local stack"
	@echo "  k3s-up           Start k3s in Docker"
	@echo "  k3s-images       Build + import local images into k3s containerd"
	@echo "  k3s-bootstrap    Generate/apply manifests (namespace+secrets+workloads+jobs)"
	@echo "  k3s-ingest       Run ingest Job in k3s"
	@echo "  k3s-train        Run train Job in k3s"
	@echo "  k3s-status       kubectl get pods -n dcgrid"
	@echo "  k3s-nodes        kubectl get nodes"

venv:
	python -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip && pip install -r requirements.txt

# -----------------------------
# Local / host workflows
# -----------------------------

# Keep `data` as an alias for the docker mode (more consistent with your env vars)
data: data-docker

data-host:
	$(PY) POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=5432 python -m dcgrid.scripts.ingest

data-docker:
	docker compose --env-file .env -f infra/docker-compose.local.yaml run --rm ingest

train: train-docker

train-host:
	$(PY) POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=5432 python -m dcgrid.scripts.train

train-docker:
	docker compose --env-file .env -f infra/docker-compose.local.yaml run --rm train

serve:
	$(PY) uvicorn dcgrid.api.main:app --host $${API_HOST:-0.0.0.0} --port $${API_PORT:-8000} --reload

compose-up:
	docker compose --env-file .env -f infra/docker-compose.local.yaml up -d --build

compose-down:
	docker compose -f infra/docker-compose.local.yaml down -v

# -----------------------------
# k3s workflows
# -----------------------------

k3s-up:
	docker compose -f infra/k3s/docker-compose.k3s.yaml up -d

k3s-down:
	docker compose -f infra/k3s/docker-compose.k3s.yaml down -v

# Build+import images into k3s (avoids ImagePullBackOff for :local)
k3s-images:
	./infra/k3s/build-images.sh

# Apply generated manifests (bootstrap should generate them first)
k3s-apply:
	$(KUBECTL) apply -f infra/k3s/manifests/generated/

k3s-bootstrap:
	APPLY_MANIFESTS=0 ./infra/k3s/bootstrap.sh
	$(MAKE) k3s-apply

k3s-deploy: k3s-apply

k3s-status:
	$(KUBECTL) -n dcgrid get pods

k3s-nodes:
	$(KUBECTL) get nodes

# Run batch jobs in k3s
k3s-ingest:
	$(KUBECTL) -n dcgrid delete job dcgrid-ingest --ignore-not-found
	$(KUBECTL) -n dcgrid apply -f infra/k3s/manifests/generated/60-job-ingest.yaml
	$(KUBECTL) -n dcgrid wait --for=condition=Ready pod -l job-name=dcgrid-ingest --timeout=900s || true
	$(KUBECTL) -n dcgrid logs -f job/dcgrid-ingest --all-containers=true


k3s-train:
	$(KUBECTL) -n dcgrid delete job dcgrid-train --ignore-not-found
	$(KUBECTL) -n dcgrid apply -f infra/k3s/manifests/generated/70-job-train.yaml
	@echo "Waiting for train pod to appear..."
	@POD=""; \
	for i in $$(seq 1 60); do \
	  POD=$$($(KUBECTL) -n dcgrid get pod -l job-name=dcgrid-train -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true); \
	  if [ -n "$$POD" ]; then break; fi; \
	  sleep 1; \
	done; \
	if [ -z "$$POD" ]; then echo "Train pod did not appear"; exit 1; fi; \
	echo "Pod: $$POD"; \
	echo "Waiting for any container to start..."; \
	for i in $$(seq 1 120); do \
	  PHASE=$$($(KUBECTL) -n dcgrid get pod $$POD -o jsonpath='{.status.phase}' 2>/dev/null || true); \
	  if [ "$$PHASE" = "Running" ] || [ "$$PHASE" = "Succeeded" ] || [ "$$PHASE" = "Failed" ]; then break; fi; \
	  sleep 1; \
	done; \
	echo "Streaming logs (all containers)"; \
	$(KUBECTL) -n dcgrid logs -f $$POD --all-containers=true || true; \
	echo "Job status:"; \
	$(KUBECTL) -n dcgrid get job dcgrid-train -o wide; \
	$(KUBECTL) -n dcgrid get pod -l job-name=dcgrid-train -o wide




k3s-restart-api:
	$(KUBECTL) -n dcgrid rollout restart deploy/dcgrid-api

# One-shot: bring up cluster, load images, deploy, ingest, train
k3s-run: k3s-up k3s-images k3s-bootstrap k3s-ingest k3s-train

# Port forwarding for k3s services (mlflow, minio, api, dashboard)
k3s-forward:
	./infra/k3s/port-forward.sh
