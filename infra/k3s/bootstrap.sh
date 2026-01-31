#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# dcgrid k3s bootstrap
# - Generates manifests into infra/k3s/manifests/generated
# - Optionally applies them if APPLY_MANIFESTS=1
# - Writes a helper file you can source to pin kubectl to the right kubeconfig
# ------------------------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Inputs
KUBECONFIG_PATH="${KUBECONFIG_PATH:-$ROOT_DIR/infra/k3s/kubeconfig/kubeconfig.yaml}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"

# Output
OUT_DIR="${OUT_DIR:-$ROOT_DIR/infra/k3s/manifests/generated}"
NAMESPACE="${NAMESPACE:-dcgrid}"

# Behavior toggles
APPLY_MANIFESTS="${APPLY_MANIFESTS:-0}"          # 1 => kubectl apply -f generated/
WRITE_CONTEXT_HELPER="${WRITE_CONTEXT_HELPER:-1}" # 1 => write infra/k3s/use-k3s.env helper

# ------------------------------------------------------------------------------
# Preflight
# ------------------------------------------------------------------------------
if [[ ! -f "$KUBECONFIG_PATH" ]]; then
  echo "kubeconfig not found yet at: $KUBECONFIG_PATH"
  echo "Run: make k3s-up  (then wait a few seconds) and re-run this script."
  exit 1
fi

# Load .env if present
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

mkdir -p "$OUT_DIR"

KUBECTL=(kubectl --kubeconfig "$KUBECONFIG_PATH")

# ------------------------------------------------------------------------------
# Defaults (can be overridden by ENV_FILE / environment)
# ------------------------------------------------------------------------------
REGION="${REGION:-IE}"

POSTGRES_DB="${POSTGRES_DB:-dcgrid}"
POSTGRES_USER="${POSTGRES_USER:-dcgrid}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-dcgrid}"

MINIO_BUCKET="${MINIO_BUCKET:-dcgrid}"
MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minioadmin}"
MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-minioadmin}"

MODEL_NAME="${MODEL_NAME:-xgb}"
FORECAST_HORIZON_H="${FORECAST_HORIZON_H:-168}"

# Optional (may be empty)
EIRGRID_SYSTEM_DATA_URL="${EIRGRID_SYSTEM_DATA_URL:-}"

# Internal service DNS (cluster-side)
POSTGRES_HOST="postgres"
POSTGRES_PORT="5432"
MINIO_ENDPOINT="http://minio:9000"
MLFLOW_HOST="mlflow"
MLFLOW_PORT="5000"
MLFLOW_TRACKING_URI="http://${MLFLOW_HOST}:${MLFLOW_PORT}"

# Local images imported into k3s containerd
API_IMAGE="${API_IMAGE:-dcgrid/api:local}"
DASHBOARD_IMAGE="${DASHBOARD_IMAGE:-dcgrid/dashboard:local}"
WORKER_IMAGE="${WORKER_IMAGE:-dcgrid/worker:local}"
MLFLOW_IMAGE="${MLFLOW_IMAGE:-dcgrid/mlflow:local}"

# ------------------------------------------------------------------------------
# Option A: write helper to pin kubectl context (avoid old host context)
# ------------------------------------------------------------------------------
if [[ "$WRITE_CONTEXT_HELPER" == "1" ]]; then
  cat > "$ROOT_DIR/infra/k3s/use-k3s.env" <<EOF
# Source this to force kubectl to use the dcgrid k3s kubeconfig
export KUBECONFIG="$KUBECONFIG_PATH"
alias k="kubectl --kubeconfig $KUBECONFIG_PATH"
EOF
  echo "Wrote helper: $ROOT_DIR/infra/k3s/use-k3s.env"
  echo "Use: source infra/k3s/use-k3s.env"
fi

echo
echo "Using kubeconfig: $KUBECONFIG_PATH"
echo "Generating manifests -> $OUT_DIR"
echo "Namespace: $NAMESPACE"
echo

# ------------------------------------------------------------------------------
# Generate manifests
# ------------------------------------------------------------------------------

cat > "$OUT_DIR/00-namespace.yaml" <<YAML
apiVersion: v1
kind: Namespace
metadata:
  name: ${NAMESPACE}
YAML

cat > "$OUT_DIR/10-secret.yaml" <<YAML
apiVersion: v1
kind: Secret
metadata:
  name: dcgrid-env
  namespace: ${NAMESPACE}
type: Opaque
stringData:
  POSTGRES_DB: "${POSTGRES_DB}"
  POSTGRES_USER: "${POSTGRES_USER}"
  POSTGRES_PASSWORD: "${POSTGRES_PASSWORD}"
  MINIO_ACCESS_KEY: "${MINIO_ACCESS_KEY}"
  MINIO_SECRET_KEY: "${MINIO_SECRET_KEY}"
YAML

cat > "$OUT_DIR/20-postgres.yaml" <<YAML
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pgdata
  namespace: ${NAMESPACE}
spec:
  accessModes: ["ReadWriteOnce"]
  resources:
    requests:
      storage: 1Gi
  storageClassName: local-path
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: ${NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels: { app: postgres }
  template:
    metadata:
      labels: { app: postgres }
    spec:
      containers:
        - name: postgres
          image: postgres:16
          ports:
            - containerPort: 5432
          env:
            - name: POSTGRES_DB
              valueFrom: { secretKeyRef: { name: dcgrid-env, key: POSTGRES_DB } }
            - name: POSTGRES_USER
              valueFrom: { secretKeyRef: { name: dcgrid-env, key: POSTGRES_USER } }
            - name: POSTGRES_PASSWORD
              valueFrom: { secretKeyRef: { name: dcgrid-env, key: POSTGRES_PASSWORD } }
          volumeMounts:
            - name: pgdata
              mountPath: /var/lib/postgresql/data
      volumes:
        - name: pgdata
          persistentVolumeClaim:
            claimName: pgdata
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: ${NAMESPACE}
spec:
  selector: { app: postgres }
  ports:
    - name: pg
      port: 5432
      targetPort: 5432
  type: ClusterIP
YAML

cat > "$OUT_DIR/30-minio.yaml" <<YAML
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: miniodata
  namespace: ${NAMESPACE}
spec:
  accessModes: ["ReadWriteOnce"]
  resources:
    requests:
      storage: 2Gi
  storageClassName: local-path
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minio
  namespace: ${NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels: { app: minio }
  template:
    metadata:
      labels: { app: minio }
    spec:
      containers:
        - name: minio
          image: minio/minio:RELEASE.2024-12-18T13-15-44Z
          args: ["server", "/data", "--console-address", ":9001"]
          ports:
            - containerPort: 9000
            - containerPort: 9001
          env:
            - name: MINIO_ROOT_USER
              valueFrom: { secretKeyRef: { name: dcgrid-env, key: MINIO_ACCESS_KEY } }
            - name: MINIO_ROOT_PASSWORD
              valueFrom: { secretKeyRef: { name: dcgrid-env, key: MINIO_SECRET_KEY } }
          volumeMounts:
            - name: miniodata
              mountPath: /data
      volumes:
        - name: miniodata
          persistentVolumeClaim:
            claimName: miniodata
---
apiVersion: v1
kind: Service
metadata:
  name: minio
  namespace: ${NAMESPACE}
spec:
  selector: { app: minio }
  ports:
    - name: s3
      port: 9000
      targetPort: 9000
    - name: console
      port: 9001
      targetPort: 9001
  type: ClusterIP
YAML

# --- MLflow (uses Postgres backend + MinIO artifact store) ---
cat > "$OUT_DIR/35-mlflow.yaml" <<YAML
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mlflowdata
  namespace: ${NAMESPACE}
spec:
  accessModes: ["ReadWriteOnce"]
  resources:
    requests:
      storage: 1Gi
  storageClassName: local-path
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: ${NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels: { app: mlflow }
  template:
    metadata:
      labels: { app: mlflow }
    spec:
      containers:
        - name: mlflow
          image: ${MLFLOW_IMAGE}
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 5000
          env:
            - name: MLFLOW_BACKEND_STORE_URI
              value: "postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"
            - name: MLFLOW_DEFAULT_ARTIFACT_ROOT
              value: "s3://${MINIO_BUCKET}/mlflow"
            - name: AWS_ACCESS_KEY_ID
              valueFrom: { secretKeyRef: { name: dcgrid-env, key: MINIO_ACCESS_KEY } }
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom: { secretKeyRef: { name: dcgrid-env, key: MINIO_SECRET_KEY } }
            - name: MLFLOW_S3_ENDPOINT_URL
              value: "${MINIO_ENDPOINT}"
            - name: AWS_DEFAULT_REGION
              value: "us-east-1"
          command: ["mlflow"]
          args:
            - "server"
            - "--host=0.0.0.0"
            - "--port=5000"
            - "--backend-store-uri=\$(MLFLOW_BACKEND_STORE_URI)"
            - "--default-artifact-root=\$(MLFLOW_DEFAULT_ARTIFACT_ROOT)"
          volumeMounts:
            - name: mlflowdata
              mountPath: /mlflow
      volumes:
        - name: mlflowdata
          persistentVolumeClaim:
            claimName: mlflowdata
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow
  namespace: ${NAMESPACE}
spec:
  selector: { app: mlflow }
  ports:
    - name: http
      port: 5000
      targetPort: 5000
  type: ClusterIP
YAML

# --- API ---
# Optional EIRGRID_SYSTEM_DATA_URL env only if set
EIRGRID_ENV_BLOCK=""
if [[ -n "$EIRGRID_SYSTEM_DATA_URL" ]]; then
  EIRGRID_ENV_BLOCK=$(cat <<EOF
            - name: EIRGRID_SYSTEM_DATA_URL
              value: "${EIRGRID_SYSTEM_DATA_URL}"
EOF
)
fi

cat > "$OUT_DIR/40-api.yaml" <<YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dcgrid-api
  namespace: ${NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels: { app: dcgrid-api }
  template:
    metadata:
      labels: { app: dcgrid-api }
    spec:
      containers:
        - name: api
          image: ${API_IMAGE}
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
          env:
            - name: REGION
              value: "${REGION}"
            - name: MODEL_NAME
              value: "${MODEL_NAME}"
            - name: FORECAST_HORIZON_H
              value: "${FORECAST_HORIZON_H}"
            - name: POSTGRES_HOST
              value: "${POSTGRES_HOST}"
            - name: POSTGRES_PORT
              value: "${POSTGRES_PORT}"
            - name: POSTGRES_DB
              valueFrom: { secretKeyRef: { name: dcgrid-env, key: POSTGRES_DB } }
            - name: POSTGRES_USER
              valueFrom: { secretKeyRef: { name: dcgrid-env, key: POSTGRES_USER } }
            - name: POSTGRES_PASSWORD
              valueFrom: { secretKeyRef: { name: dcgrid-env, key: POSTGRES_PASSWORD } }
            - name: MINIO_ENDPOINT
              value: "${MINIO_ENDPOINT}"
            - name: MINIO_ACCESS_KEY
              valueFrom: { secretKeyRef: { name: dcgrid-env, key: MINIO_ACCESS_KEY } }
            - name: MINIO_SECRET_KEY
              valueFrom: { secretKeyRef: { name: dcgrid-env, key: MINIO_SECRET_KEY } }
            - name: MINIO_BUCKET
              value: "${MINIO_BUCKET}"
            - name: MLFLOW_TRACKING_URI
              value: "${MLFLOW_TRACKING_URI}"
${EIRGRID_ENV_BLOCK}
---
apiVersion: v1
kind: Service
metadata:
  name: dcgrid-api
  namespace: ${NAMESPACE}
spec:
  selector: { app: dcgrid-api }
  ports:
    - name: http
      port: 8000
      targetPort: 8000
  type: ClusterIP
YAML

# --- Dashboard ---
cat > "$OUT_DIR/50-dashboard.yaml" <<YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dcgrid-dashboard
  namespace: ${NAMESPACE}
spec:
  replicas: 1
  selector:
    matchLabels: { app: dcgrid-dashboard }
  template:
    metadata:
      labels: { app: dcgrid-dashboard }
    spec:
      containers:
        - name: dashboard
          image: ${DASHBOARD_IMAGE}
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8501
          command: ["streamlit"]
          args:
            - "run"
            - "/app/services/dcgrid/dashboard/app.py"
            - "--server.address=0.0.0.0"
            - "--server.port=8501"
          env:
            - name: REGION
              value: "${REGION}"
            - name: POSTGRES_HOST
              value: "${POSTGRES_HOST}"
            - name: POSTGRES_PORT
              value: "${POSTGRES_PORT}"
            - name: POSTGRES_DB
              valueFrom: { secretKeyRef: { name: dcgrid-env, key: POSTGRES_DB } }
            - name: POSTGRES_USER
              valueFrom: { secretKeyRef: { name: dcgrid-env, key: POSTGRES_USER } }
            - name: POSTGRES_PASSWORD
              valueFrom: { secretKeyRef: { name: dcgrid-env, key: POSTGRES_PASSWORD } }
---
apiVersion: v1
kind: Service
metadata:
  name: dcgrid-dashboard
  namespace: ${NAMESPACE}
spec:
  selector: { app: dcgrid-dashboard }
  ports:
    - name: http
      port: 8501
      targetPort: 8501
  type: ClusterIP
YAML

# ------------------------------------------------------------------------------
# Apply (optional)
# ------------------------------------------------------------------------------
echo "Generated manifests:"
ls -1 "$OUT_DIR" | sed 's/^/  - /'

if [[ "$APPLY_MANIFESTS" == "1" ]]; then
  echo
  echo "Applying generated manifests..."
  "${KUBECTL[@]}" apply -f "$OUT_DIR"

  echo
  echo "Pods:"
  "${KUBECTL[@]}" -n "$NAMESPACE" get pods -o wide
else
  echo
  echo "Not applying manifests (APPLY_MANIFESTS=0)."
  echo "To apply:"
  echo "  kubectl --kubeconfig $KUBECONFIG_PATH apply -f $OUT_DIR"
fi

echo
echo "Access (recommended port-forwards):"
echo "  kubectl --kubeconfig $KUBECONFIG_PATH -n $NAMESPACE port-forward svc/dcgrid-api 8000:8000"
echo "  kubectl --kubeconfig $KUBECONFIG_PATH -n $NAMESPACE port-forward svc/dcgrid-dashboard 8501:8501"
echo "  kubectl --kubeconfig $KUBECONFIG_PATH -n $NAMESPACE port-forward svc/mlflow 5000:5000"
echo "  kubectl --kubeconfig $KUBECONFIG_PATH -n $NAMESPACE port-forward svc/minio 9000:9000"
echo "  kubectl --kubeconfig $KUBECONFIG_PATH -n $NAMESPACE port-forward svc/minio 9001:9001"
