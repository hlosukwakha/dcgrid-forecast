#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Find k3s node container (docker-compose usually names it k3s-k3s-1)
K3S_NODE_CONTAINER="${K3S_NODE_CONTAINER:-$(docker ps --format '{{.Names}}' | grep -E '(^|-)k3s-1$|k3s-k3s-1' | head -n 1)}"
if [[ -z "${K3S_NODE_CONTAINER}" ]]; then
  echo "ERROR: could not find k3s node container (expected something like k3s-k3s-1)."
  echo "Run: docker ps | grep -i k3s"
  exit 1
fi
echo "Using k3s node container: ${K3S_NODE_CONTAINER}"

echo "==> Building local images (Docker)..."
docker build -t dcgrid/api:local       -f services/dcgrid/api/Dockerfile .
docker build -t dcgrid/dashboard:local -f services/dcgrid/dashboard/Dockerfile .
docker build -t dcgrid/worker:local    -f services/dcgrid/worker.Dockerfile .
docker build -t dcgrid/mlflow:local    -f infra/mlflow/Dockerfile .

echo "Built images:"
echo "  - dcgrid/api:local"
echo "  - dcgrid/dashboard:local"
echo "  - dcgrid/worker:local"
echo "  - dcgrid/mlflow:local"

echo
echo "==> Importing images into k3s containerd..."
for img in dcgrid/api:local dcgrid/dashboard:local dcgrid/worker:local dcgrid/mlflow:local; do
  echo "Importing ${img} ..."
  docker save "${img}" | docker exec -i "${K3S_NODE_CONTAINER}" /bin/ctr -n k8s.io images import -
done

echo
echo "==> Verifying images inside k3s (crictl)..."
docker exec -i "${K3S_NODE_CONTAINER}" sh -lc \
  "/bin/crictl images | egrep 'dcgrid/(api|dashboard|worker|mlflow)\\s+local' || true"

echo
echo "Done."
