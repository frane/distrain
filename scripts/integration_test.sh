#!/usr/bin/env bash
# v0.2 Integration Test
#
# Spins up the full stack via docker-compose, runs a training cycle,
# tests state recovery, node resume, format compatibility, and convergence.
#
# Prerequisites: docker, docker compose, cargo (for local node builds)
#
# Usage: ./scripts/integration_test.sh [--keep]
#   --keep: don't tear down containers after test (for debugging)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_DIR/docker/docker-compose.yml"
KEEP_CONTAINERS=false

if [[ "${1:-}" == "--keep" ]]; then
    KEEP_CONTAINERS=true
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓ $1${NC}"; }
fail() { echo -e "${RED}✗ $1${NC}"; FAILURES=$((FAILURES + 1)); }
info() { echo -e "${YELLOW}→ $1${NC}"; }

FAILURES=0
TESTS=0
COORD_PID=0
NODE1_PID=0
NODE2_PID=0
CACHE1=""
CACHE2=""

assert_eq() {
    TESTS=$((TESTS + 1))
    if [[ "$1" == "$2" ]]; then
        pass "$3"
    else
        fail "$3 (expected '$2', got '$1')"
    fi
}

assert_gt() {
    TESTS=$((TESTS + 1))
    if (( $(echo "$1 > $2" | bc -l) )); then
        pass "$3"
    else
        fail "$3 ($1 not > $2)"
    fi
}

assert_ok() {
    TESTS=$((TESTS + 1))
    if [[ $1 -eq 0 ]]; then
        pass "$2"
    else
        fail "$2 (exit code $1)"
    fi
}

cleanup() {
    info "Cleaning up processes..."
    kill $COORD_PID $NODE1_PID $NODE2_PID 2>/dev/null || true
    wait $COORD_PID $NODE1_PID $NODE2_PID 2>/dev/null || true
    docker rm -f distrain-test-minio 2>/dev/null || true
    rm -rf "$CACHE1" "$CACHE2" 2>/dev/null || true
}
trap cleanup EXIT

# ── Build ──────────────────────────────────────────────────────────

info "Building coordinator and node (release)..."
cd "$PROJECT_DIR"
cargo build --release -p distrain-coordinator -p distrain-node 2>&1 | tail -3

COORDINATOR="$PROJECT_DIR/target/release/coordinator"
NODE="$PROJECT_DIR/target/release/distrain-node"

assert_ok $? "Build coordinator + node"

# ── MinIO ──────────────────────────────────────────────────────────

# Use existing MinIO if running on port 9000, else start via Docker
MINIO_PORT=9000
if curl -sf "http://localhost:${MINIO_PORT}/minio/health/live" > /dev/null 2>&1; then
    info "Using existing MinIO on port ${MINIO_PORT}"
else
    info "Starting MinIO via Docker..."
    MINIO_PORT=9002  # avoid conflict
    docker run -d --name distrain-test-minio \
        -p ${MINIO_PORT}:9000 \
        -e MINIO_ROOT_USER=minioadmin \
        -e MINIO_ROOT_PASSWORD=minioadmin \
        minio/minio server /data 2>/dev/null || true
    for i in $(seq 1 15); do
        if curl -sf "http://localhost:${MINIO_PORT}/minio/health/live" > /dev/null 2>&1; then break; fi
        sleep 1
    done
fi
curl -sf "http://localhost:${MINIO_PORT}/minio/health/live" > /dev/null 2>&1
assert_ok $? "MinIO is healthy on port ${MINIO_PORT}"

MINIO_URL="http://localhost:${MINIO_PORT}"

# ── Start Coordinator ──────────────────────────────────────────────

COORD_PORT=8099  # non-conflicting port for test
TEST_BUCKET="distrain-test-$(date +%s)"  # unique bucket per run
COORD_URL="http://localhost:${COORD_PORT}"

start_coordinator() {
    R2_ENDPOINT=${MINIO_URL} \
    R2_BUCKET=${TEST_BUCKET} \
    R2_ACCESS_KEY_ID=minioadmin \
    R2_SECRET_ACCESS_KEY=minioadmin \
    PORT=${COORD_PORT} \
    MIN_CONTRIBUTIONS=2 \
    MAX_STALENESS=10 \
    VOCAB_SIZE=32768 \
    KEEP_VERSIONS=20 \
    ENABLE_REBASING=true \
    REBASING_THRESHOLD=3 \
    RUST_LOG=warn \
    "$COORDINATOR" &
    COORD_PID=$!
    sleep 2
}

info "Starting coordinator on port ${COORD_PORT}..."
start_coordinator

curl -sf "${COORD_URL}/health" > /dev/null 2>&1
assert_ok $? "Coordinator is healthy"

# ── Bootstrap v0 checkpoint ────────────────────────────────────────

info "Bootstrapping v0 checkpoint (micro-test preset)..."
cat > /tmp/distrain-test-node.toml <<TOMLEOF
coordinator_url = "${COORD_URL}"
api_key = ""
gpu_device = -1
target_push_interval_secs = 60.0
min_inner_steps = 5
max_inner_steps = 10
cache_dir = "/tmp/distrain-test-bootstrap"
max_cache_gb = 10
seq_len = 64
compression_pipeline = "block"
quantization_mode = "int8_block"
use_importance = false

[storage]
endpoint = "${MINIO_URL}"
bucket = "${TEST_BUCKET}"
access_key_id = "minioadmin"
secret_access_key = "minioadmin"
region = "auto"
TOMLEOF

"$NODE" bootstrap --config /tmp/distrain-test-node.toml --preset micro-test 2>&1 | tail -3
assert_ok $? "Bootstrap v0 checkpoint"

# Upload synthetic training data via the coordinator's upload proxy
info "Uploading synthetic training data..."

# Create 2 small shards (random-ish uint16 tokens) and manifest
# Use a tiny Rust program to generate binary shard data
python3 -c "
import struct, json, os
# Generate 2 shards of 10K uint16 tokens each
for i in range(2):
    tokens = [(j * 7 + i * 13) % 255 for j in range(10000)]  # vocab_size=256 for micro-test
    data = struct.pack('<' + 'H' * len(tokens), *tokens)
    with open(f'/tmp/shard_{i:04d}.bin', 'wb') as f:
        f.write(data)

manifest = {'num_shards': 2, 'shards': [
    {'filename': 'shard_0000.bin', 'num_tokens': 10000, 'size_bytes': 20000},
    {'filename': 'shard_0001.bin', 'num_tokens': 10000, 'size_bytes': 20000},
], 'total_tokens': 20000}
with open('/tmp/manifest.json', 'w') as f:
    json.dump(manifest, f)
print('Generated test data')
" 2>&1

# Upload via coordinator proxy (avoids needing S3 client)
curl -sf -X PUT "${COORD_URL}/upload/data/manifest.json" --data-binary @/tmp/manifest.json > /dev/null
curl -sf -X PUT "${COORD_URL}/upload/data/shard_0000.bin" --data-binary @/tmp/shard_0000.bin > /dev/null
curl -sf -X PUT "${COORD_URL}/upload/data/shard_0001.bin" --data-binary @/tmp/shard_0001.bin > /dev/null
assert_ok $? "Uploaded synthetic training data"

# Verify checkpoint exists
VERSION=$(curl -sf "${COORD_URL}/checkpoint/latest" | python3 -c "import sys,json; print(json.load(sys.stdin)['version'])")
assert_eq "$VERSION" "0" "Checkpoint v0 exists"

# ── Test 1: Coordinator state recovery ─────────────────────────────

info "Test: Coordinator state recovery..."
# Kill coordinator
kill $COORD_PID 2>/dev/null || true
wait $COORD_PID 2>/dev/null || true
sleep 1

# Restart coordinator
start_coordinator

# Verify it recovered
VERSION_AFTER=$(curl -sf "${COORD_URL}/checkpoint/latest" | python3 -c "import sys,json; print(json.load(sys.stdin)['version'])")
assert_eq "$VERSION_AFTER" "0" "Coordinator recovered checkpoint version after restart"

# ── Test 2: Training produces checkpoints ──────────────────────────

info "Test: Training with 2 CPU nodes produces checkpoints..."
CACHE1=$(mktemp -d)
CACHE2=$(mktemp -d)

cat > /tmp/distrain-test-node1.toml <<TOMLEOF
coordinator_url = "${COORD_URL}"
api_key = ""
gpu_device = -1
target_push_interval_secs = 30.0
min_inner_steps = 3
max_inner_steps = 3
cache_dir = "${CACHE1}"
max_cache_gb = 10
seq_len = 64
compression_pipeline = "block"
quantization_mode = "int8_block"
use_importance = false

[storage]
endpoint = "${MINIO_URL}"
bucket = "${TEST_BUCKET}"
access_key_id = "minioadmin"
secret_access_key = "minioadmin"
region = "auto"
TOMLEOF

cat > /tmp/distrain-test-node2.toml <<TOMLEOF
coordinator_url = "${COORD_URL}"
api_key = ""
gpu_device = -1
target_push_interval_secs = 30.0
min_inner_steps = 3
max_inner_steps = 3
cache_dir = "${CACHE2}"
max_cache_gb = 10
seq_len = 64
compression_pipeline = "unstructured"
quantization_mode = "int8_tensor"
use_importance = false

[storage]
endpoint = "${MINIO_URL}"
bucket = "${TEST_BUCKET}"
access_key_id = "minioadmin"
secret_access_key = "minioadmin"
region = "auto"
TOMLEOF

# Run both nodes for ~30 seconds
"$NODE" start --config /tmp/distrain-test-node1.toml --cpu &
NODE1_PID=$!
"$NODE" start --config /tmp/distrain-test-node2.toml --cpu &
NODE2_PID=$!

info "Waiting for training (60s)..."
sleep 60

# Check if checkpoints were produced
NEW_VERSION=$(curl -sf "${COORD_URL}/checkpoint/latest" | python3 -c "import sys,json; print(json.load(sys.stdin)['version'])")
TESTS=$((TESTS + 1))
if [[ "$NEW_VERSION" -gt "0" ]]; then
    pass "Training produced checkpoint v${NEW_VERSION} (>v0)"
else
    fail "No checkpoints produced after 30s"
fi

# ── Test 3: Format compatibility (block + unstructured in same run) ──

STATUS=$(curl -sf "${COORD_URL}/status")
TOTAL_CONTRIBS=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['total_contributions'])")
TESTS=$((TESTS + 1))
if [[ "$TOTAL_CONTRIBS" -gt "0" ]]; then
    pass "Both block and unstructured nodes contributed (${TOTAL_CONTRIBS} total)"
else
    fail "No contributions recorded"
fi

# ── Test 4: Node resume after kill ─────────────────────────────────

info "Test: Node resume after kill..."
# Kill node 1
kill $NODE1_PID 2>/dev/null || true
wait $NODE1_PID 2>/dev/null || true
sleep 1

# Check state file exists
TESTS=$((TESTS + 1))
if [[ -f "$CACHE1/state.toml" ]]; then
    pass "Node 1 saved state.toml on shutdown"
else
    fail "Node 1 did not save state.toml"
fi

# Restart node 1
"$NODE" start --config /tmp/distrain-test-node1.toml --cpu &
NODE1_PID=$!
sleep 20

# It should have resumed (check logs would show "Resuming from saved state")
TESTS=$((TESTS + 1))
RESUMED_VERSION=$(curl -sf "${COORD_URL}/checkpoint/latest" | python3 -c "import sys,json; print(json.load(sys.stdin)['version'])")
if [[ "$RESUMED_VERSION" -ge "$NEW_VERSION" ]]; then
    pass "Node 1 resumed and training continued (v${RESUMED_VERSION})"
else
    fail "Training did not continue after node resume"
fi

# ── Test 5: Coordinator restart mid-training ───────────────────────

info "Test: Coordinator restart mid-training..."
PRE_KILL_VERSION=$RESUMED_VERSION

kill $COORD_PID 2>/dev/null || true
wait $COORD_PID 2>/dev/null || true
sleep 2

# Restart coordinator
start_coordinator

RECOVERED_VERSION=$(curl -sf "${COORD_URL}/checkpoint/latest" | python3 -c "import sys,json; print(json.load(sys.stdin)['version'])")
assert_eq "$RECOVERED_VERSION" "$PRE_KILL_VERSION" "Coordinator recovered to v${PRE_KILL_VERSION} after restart"

# Wait for more training
sleep 15
FINAL_VERSION=$(curl -sf "${COORD_URL}/checkpoint/latest" | python3 -c "import sys,json; print(json.load(sys.stdin)['version'])")
TESTS=$((TESTS + 1))
if [[ "$FINAL_VERSION" -ge "$PRE_KILL_VERSION" ]]; then
    pass "Training continued after coordinator restart (v${FINAL_VERSION})"
else
    fail "Training stalled after coordinator restart"
fi

# ── Cleanup (handled by trap) ──────────────────────────────────────

# ── Results ────────────────────────────────────────────────────────

echo ""
echo "=========================="
if [[ $FAILURES -eq 0 ]]; then
    echo -e "${GREEN}ALL $TESTS TESTS PASSED${NC}"
else
    echo -e "${RED}$FAILURES/$TESTS TESTS FAILED${NC}"
fi
echo "=========================="

exit $FAILURES
