#!/usr/bin/env bash
set -euo pipefail

KUBECONFIG="$(pwd)/infra/k3s/kubeconfig/kubeconfig.yaml"

if [ ! -f "$KUBECONFIG" ]; then
  echo "kubeconfig not found yet at $KUBECONFIG"
  echo "Run: make k3s-up  (then wait a few seconds) and re-run this script."
  exit 1
fi

export KUBECONFIG="$KUBECONFIG"

kubectl create namespace dcgrid --dry-run=client -o yaml | kubectl apply -f -

# Basic secrets (for demo)
kubectl -n dcgrid create secret generic dcgrid-env --from-literal=POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-dcgrid}" --dry-run=client -o yaml | kubectl apply -f -

echo "Bootstrap complete: namespace=dcgrid"
