#!/bin/bash
set -e

# SSH setup (for RunPod access)
mkdir -p /root/.ssh
if [ -n "$PUBLIC_KEY" ]; then
  echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi
/usr/sbin/sshd 2>/dev/null || true

# Generate node.toml from environment variables
COORDINATOR_URL="${COORDINATOR_URL:-http://localhost:8000}"
S3_ENDPOINT="${S3_ENDPOINT:-http://localhost:9000}"
S3_BUCKET="${S3_BUCKET:-distrain-training}"
S3_ACCESS_KEY="${S3_ACCESS_KEY:-minioadmin}"
S3_SECRET_KEY="${S3_SECRET_KEY:-minioadmin}"
S3_REGION="${S3_REGION:-us-east-1}"
GPU_DEVICE="${GPU_DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MIN_INNER_STEPS="${MIN_INNER_STEPS:-50}"
MAX_INNER_STEPS="${MAX_INNER_STEPS:-500}"
PUSH_INTERVAL="${PUSH_INTERVAL:-60.0}"

cat > /workspace/node.toml << EOF
coordinator_url = "${COORDINATOR_URL}"
api_key = ""
gpu_device = ${GPU_DEVICE}
target_push_interval_secs = ${PUSH_INTERVAL}
min_inner_steps = ${MIN_INNER_STEPS}
max_inner_steps = ${MAX_INNER_STEPS}
cache_dir = "/workspace/cache"
max_cache_gb = 20
batch_size = ${BATCH_SIZE}
seq_len = 512
force_batch_size = ${BATCH_SIZE}

[storage]
endpoint = "${S3_ENDPOINT}"
bucket = "${S3_BUCKET}"
access_key_id = "${S3_ACCESS_KEY}"
secret_access_key = "${S3_SECRET_KEY}"
region = "${S3_REGION}"
EOF

echo "Generated /workspace/node.toml"
cat /workspace/node.toml

export RUST_LOG="${RUST_LOG:-info}"

# Start training (foreground, tee to log file for SSH debugging)
exec distrain-node start --config /workspace/node.toml 2>&1 | tee /workspace/distrain-node.log
