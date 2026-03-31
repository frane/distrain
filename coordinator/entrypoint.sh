#!/bin/bash
set -e

# ============================================================
# Distrain Coordinator entrypoint
#
# Starts MinIO (internal S3), bootstraps v0 if needed, then
# runs the coordinator. Nodes connect via external URLs.
# ============================================================

# --- Defaults ---
COORDINATOR_PORT="${COORDINATOR_PORT:-8000}"
S3_ACCESS_KEY="${S3_ACCESS_KEY:-minioadmin}"
S3_SECRET_KEY="${S3_SECRET_KEY:-minioadmin}"
S3_BUCKET="${S3_BUCKET:-distrain-training}"
S3_REGION="${S3_REGION:-us-east-1}"
S3_EXTERNAL_ENDPOINT="${S3_EXTERNAL_ENDPOINT:-}"
COORDINATOR_EXTERNAL_URL="${COORDINATOR_EXTERNAL_URL:-}"
MIN_CONTRIBUTIONS="${MIN_CONTRIBUTIONS:-0}"
MIN_WEIGHT="${MIN_WEIGHT:-0}"
MAX_STALENESS="${MAX_STALENESS:-30}"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
KEEP_VERSIONS="${KEEP_VERSIONS:-3}"
PRESET="${PRESET:-tiny}"

MINIO_DATA_DIR="/data/minio"
MINIO_PORT=9000

# --- SSH (optional, for RunPod-style access) ---
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p /root/.ssh
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    /usr/sbin/sshd 2>/dev/null || true
    echo "[entrypoint] SSH enabled"
fi

# --- Start MinIO ---
mkdir -p "$MINIO_DATA_DIR"
export MINIO_ROOT_USER="$S3_ACCESS_KEY"
export MINIO_ROOT_PASSWORD="$S3_SECRET_KEY"

echo "[entrypoint] Starting MinIO on :${MINIO_PORT}..."
minio server "$MINIO_DATA_DIR" --address ":${MINIO_PORT}" --console-address ":9001" &
MINIO_PID=$!

# Wait for MinIO to be ready (up to 30s)
echo "[entrypoint] Waiting for MinIO..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:${MINIO_PORT}/minio/health/live >/dev/null 2>&1; then
        echo "[entrypoint] MinIO ready (${i}s)"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "[entrypoint] ERROR: MinIO failed to start within 30s"
        exit 1
    fi
    sleep 1
done

# --- Bootstrap v0 checkpoint if none exists ---
# Generate a temporary node.toml for the bootstrap command (points to local MinIO)
BOOTSTRAP_TOML="/tmp/bootstrap-node.toml"
cat > "$BOOTSTRAP_TOML" << EOF
coordinator_url = "http://localhost:${COORDINATOR_PORT}"
api_key = ""
gpu_device = -1
target_push_interval_secs = 60.0
min_inner_steps = 10
max_inner_steps = 500
cache_dir = "/tmp/distrain-cache"
max_cache_gb = 5
batch_size = 4
seq_len = 512

[storage]
endpoint = "http://localhost:${MINIO_PORT}"
bucket = "${S3_BUCKET}"
access_key_id = "${S3_ACCESS_KEY}"
secret_access_key = "${S3_SECRET_KEY}"
region = "${S3_REGION}"
EOF

# Check if v0 checkpoint already exists.
# MinIO stores objects as files under MINIO_DATA_DIR/<bucket>/<key>.
# Check the accumulator (always created alongside checkpoint by bootstrap).
echo "[entrypoint] Checking for existing checkpoint..."
if [ -f "${MINIO_DATA_DIR}/${S3_BUCKET}/accumulator/current.json" ]; then
    echo "[entrypoint] Checkpoint already exists, skipping bootstrap"
else
    echo "[entrypoint] No checkpoint found. Bootstrapping v0 with preset '${PRESET}'..."
    export RUST_LOG="${RUST_LOG:-info}"
    distrain-node bootstrap --config "$BOOTSTRAP_TOML" --preset "$PRESET"
    echo "[entrypoint] Bootstrap complete"
fi

rm -f "$BOOTSTRAP_TOML"

# --- Start Coordinator ---
# Coordinator connects to MinIO on localhost. Nodes get the external endpoint via /config.
export R2_ENDPOINT="http://localhost:${MINIO_PORT}"
export R2_BUCKET="$S3_BUCKET"
export R2_ACCESS_KEY_ID="$S3_ACCESS_KEY"
export R2_SECRET_ACCESS_KEY="$S3_SECRET_KEY"
export PORT="$COORDINATOR_PORT"
export HOST="0.0.0.0"
export MIN_CONTRIBUTIONS="$MIN_CONTRIBUTIONS"
export MIN_WEIGHT="$MIN_WEIGHT"
export MAX_STALENESS="$MAX_STALENESS"
export VOCAB_SIZE="$VOCAB_SIZE"
export KEEP_VERSIONS="$KEEP_VERSIONS"

# Set external storage endpoint for /config response to nodes
if [ -n "$S3_EXTERNAL_ENDPOINT" ]; then
    export S3_EXTERNAL_ENDPOINT="$S3_EXTERNAL_ENDPOINT"
    echo "[entrypoint] Nodes will use S3 endpoint: $S3_EXTERNAL_ENDPOINT"
else
    echo "[entrypoint] WARNING: S3_EXTERNAL_ENDPOINT not set. Nodes will get localhost:${MINIO_PORT} from /config."
    echo "[entrypoint]          Set S3_EXTERNAL_ENDPOINT=http://<PUBLIC_IP>:${MINIO_PORT} for remote nodes."
fi

export RUST_LOG="${RUST_LOG:-info}"

echo "[entrypoint] Starting coordinator on :${COORDINATOR_PORT}..."
echo "[entrypoint] Config: MIN_CONTRIBUTIONS=${MIN_CONTRIBUTIONS} MIN_WEIGHT=${MIN_WEIGHT} MAX_STALENESS=${MAX_STALENESS} VOCAB_SIZE=${VOCAB_SIZE} KEEP_VERSIONS=${KEEP_VERSIONS} PRESET=${PRESET}"

# Run coordinator in foreground. If it exits, the container stops.
exec coordinator
