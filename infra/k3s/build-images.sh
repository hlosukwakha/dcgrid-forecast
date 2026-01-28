#!/usr/bin/env bash
set -euo pipefail

# Build local images to be used by k3s (single-node).
docker build -t dcgrid/api:local -f services/dcgrid/api/Dockerfile .
docker build -t dcgrid/dashboard:local -f services/dcgrid/dashboard/Dockerfile .
docker build -t dcgrid/worker:local -f services/dcgrid/worker.Dockerfile .
echo "Built images: dcgrid/api:local dcgrid/dashboard:local dcgrid/worker:local"
