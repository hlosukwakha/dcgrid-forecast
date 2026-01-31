#!/usr/bin/env bash
set -euo pipefail

KUBECONFIG="${KUBECONFIG:-$(pwd)/infra/k3s/kubeconfig/kubeconfig.yaml}"
export KUBECONFIG

NAMESPACE="${NAMESPACE:-dcgrid}"

# Local ports (change if you have collisions)
DASH_LOCAL_PORT="${DASH_LOCAL_PORT:-8501}"
API_LOCAL_PORT="${API_LOCAL_PORT:-8000}"

echo "Using kubeconfig: $KUBECONFIG"
echo "Port-forwarding:"
echo "  dashboard -> http://localhost:${DASH_LOCAL_PORT}"
echo "  api       -> http://localhost:${API_LOCAL_PORT}"
echo ""
echo "Press Ctrl+C to stop."

# Ensure targets exist before forwarding (optional but helpful)
kubectl -n "$NAMESPACE" rollout status deploy/dcgrid-dashboard --timeout=120s || true
kubectl -n "$NAMESPACE" rollout status deploy/dcgrid-api --timeout=120s || true

# Run both forwards in background and wait
kubectl -n "$NAMESPACE" port-forward deploy/dcgrid-dashboard "${DASH_LOCAL_PORT}:8501" &
PF1=$!
kubectl -n "$NAMESPACE" port-forward deploy/dcgrid-api "${API_LOCAL_PORT}:8000" &
PF2=$!

cleanup() {
  echo "Stopping port-forwards..."
  kill "$PF1" "$PF2" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait
