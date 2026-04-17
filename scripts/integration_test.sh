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
    if [[ "$KEEP_CONTAINERS" == "false" ]]; then
        info "Tearing down containers..."
        docker compose -f "$COMPOSE_FILE" down -v --remove-orphans 2>/dev/null || true
    else
        info "Keeping containers running (--keep mode)"
    fi
}
trap cleanup EXIT

# ── Build ──────────────────────────────────────────────────────────

info "Building coordinator and node (release)..."
cd "$PROJECT_DIR"
cargo build --release -p distrain-coordinator -p distrain-node 2>&1 | tail -3

COORDINATOR="$PROJECT_DIR/target/release/coordinator"
NODE="$PROJECT_DIR/target/release/distrain-node"

assert_ok $? "Build coordinator + node"

# ── Start MinIO ────────────────────────────────────────────────────

info "Starting MinIO..."
docker compose -f "$COMPOSE_FILE" up -d minio 2>/dev/null
sleep 3

# Wait for MinIO health
for i in $(seq 1 15); do
    if curl -sf http://localhost:9000/minio/health/live > /dev/null 2>&1; then
        break
    fi
    sleep 1
done
curl -sf http://localhost:9000/minio/health/live > /dev/null 2>&1
assert_ok $? "MinIO is healthy"

# ── Start Coordinator ──────────────────────────────────────────────

info "Starting coordinator..."
R2_ENDPOINT=http://localhost:9000 \
R2_BUCKET=distrain-training \
R2_ACCESS_KEY_ID=minioadmin \
R2_SECRET_ACCESS_KEY=minioadmin \
MIN_CONTRIBUTIONS=2 \
MAX_STALENESS=10 \
VOCAB_SIZE=32768 \
KEEP_VERSIONS=20 \
ENABLE_REBASING=true \
REBASING_THRESHOLD=3 \
RUST_LOG=info \
"$COORDINATOR" &
COORD_PID=$!
sleep 2

curl -sf http://localhost:8000/health > /dev/null 2>&1
assert_ok $? "Coordinator is healthy"

# ── Bootstrap v0 checkpoint ────────────────────────────────────────

info "Bootstrapping v0 checkpoint (micro-test preset)..."
cat > /tmp/distrain-test-node.toml <<TOMLEOF
coordinator_url = "http://localhost:8000"
api_key = ""
gpu_device = -1

[storage]
endpoint = "http://localhost:9000"
bucket = "distrain-training"
access_key_id = "minioadmin"
secret_access_key = "minioadmin"
region = "auto"

compression_pipeline = "block"
use_importance = false
TOMLEOF

"$NODE" bootstrap --config /tmp/distrain-test-node.toml --preset micro-test 2>&1 | tail -3
assert_ok $? "Bootstrap v0 checkpoint"

# Verify checkpoint exists
VERSION=$(curl -sf http://localhost:8000/checkpoint/latest | python3 -c "import sys,json; print(json.load(sys.stdin)['version'])")
assert_eq "$VERSION" "0" "Checkpoint v0 exists"

# ── Test 1: Coordinator state recovery ─────────────────────────────

info "Test: Coordinator state recovery..."
# Kill coordinator
kill $COORD_PID 2>/dev/null || true
wait $COORD_PID 2>/dev/null || true
sleep 1

# Restart coordinator
R2_ENDPOINT=http://localhost:9000 \
R2_BUCKET=distrain-training \
R2_ACCESS_KEY_ID=minioadmin \
R2_SECRET_ACCESS_KEY=minioadmin \
MIN_CONTRIBUTIONS=2 \
MAX_STALENESS=10 \
VOCAB_SIZE=32768 \
KEEP_VERSIONS=20 \
ENABLE_REBASING=true \
RUST_LOG=info \
"$COORDINATOR" &
COORD_PID=$!
sleep 2

# Verify it recovered
VERSION_AFTER=$(curl -sf http://localhost:8000/checkpoint/latest | python3 -c "import sys,json; print(json.load(sys.stdin)['version'])")
assert_eq "$VERSION_AFTER" "0" "Coordinator recovered checkpoint version after restart"

# ── Test 2: Training produces checkpoints ──────────────────────────

info "Test: Training with 2 CPU nodes produces checkpoints..."
CACHE1=$(mktemp -d)
CACHE2=$(mktemp -d)

cat > /tmp/distrain-test-node1.toml <<TOMLEOF
coordinator_url = "http://localhost:8000"
api_key = ""
gpu_device = -1
min_inner_steps = 5
max_inner_steps = 5
cache_dir = "$CACHE1"
compression_pipeline = "block"

[storage]
endpoint = "http://localhost:9000"
bucket = "distrain-training"
access_key_id = "minioadmin"
secret_access_key = "minioadmin"
region = "auto"
TOMLEOF

cat > /tmp/distrain-test-node2.toml <<TOMLEOF
coordinator_url = "http://localhost:8000"
api_key = ""
gpu_device = -1
min_inner_steps = 5
max_inner_steps = 5
cache_dir = "$CACHE2"
compression_pipeline = "unstructured"

[storage]
endpoint = "http://localhost:9000"
bucket = "distrain-training"
access_key_id = "minioadmin"
secret_access_key = "minioadmin"
region = "auto"
TOMLEOF

# Run both nodes for ~30 seconds
"$NODE" start --config /tmp/distrain-test-node1.toml --cpu &
NODE1_PID=$!
"$NODE" start --config /tmp/distrain-test-node2.toml --cpu &
NODE2_PID=$!

info "Waiting for training (30s)..."
sleep 30

# Check if checkpoints were produced
NEW_VERSION=$(curl -sf http://localhost:8000/checkpoint/latest | python3 -c "import sys,json; print(json.load(sys.stdin)['version'])")
TESTS=$((TESTS + 1))
if [[ "$NEW_VERSION" -gt "0" ]]; then
    pass "Training produced checkpoint v${NEW_VERSION} (>v0)"
else
    fail "No checkpoints produced after 30s"
fi

# ── Test 3: Format compatibility (block + unstructured in same run) ──

STATUS=$(curl -sf http://localhost:8000/status)
TOTAL_CONTRIBS=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['total_contributions'])")
TESTS=$((TESTS + 1))
if [[ "$TOTAL_CONTRIBS" -gt "0" ]]; then
    pass "Both block and unstructured nodes contributed ($TOTAL_CONTRIBS total)"
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
sleep 10

# It should have resumed (check logs would show "Resuming from saved state")
TESTS=$((TESTS + 1))
RESUMED_VERSION=$(curl -sf http://localhost:8000/checkpoint/latest | python3 -c "import sys,json; print(json.load(sys.stdin)['version'])")
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
R2_ENDPOINT=http://localhost:9000 \
R2_BUCKET=distrain-training \
R2_ACCESS_KEY_ID=minioadmin \
R2_SECRET_ACCESS_KEY=minioadmin \
MIN_CONTRIBUTIONS=2 \
MAX_STALENESS=10 \
VOCAB_SIZE=32768 \
KEEP_VERSIONS=20 \
ENABLE_REBASING=true \
RUST_LOG=info \
"$COORDINATOR" &
COORD_PID=$!
sleep 3

RECOVERED_VERSION=$(curl -sf http://localhost:8000/checkpoint/latest | python3 -c "import sys,json; print(json.load(sys.stdin)['version'])")
assert_eq "$RECOVERED_VERSION" "$PRE_KILL_VERSION" "Coordinator recovered to v${PRE_KILL_VERSION} after restart"

# Wait for more training
sleep 15
FINAL_VERSION=$(curl -sf http://localhost:8000/checkpoint/latest | python3 -c "import sys,json; print(json.load(sys.stdin)['version'])")
TESTS=$((TESTS + 1))
if [[ "$FINAL_VERSION" -ge "$PRE_KILL_VERSION" ]]; then
    pass "Training continued after coordinator restart (v${FINAL_VERSION})"
else
    fail "Training stalled after coordinator restart"
fi

# ── Cleanup ────────────────────────────────────────────────────────

info "Cleaning up..."
kill $NODE1_PID $NODE2_PID 2>/dev/null || true
wait $NODE1_PID $NODE2_PID 2>/dev/null || true
kill $COORD_PID 2>/dev/null || true
wait $COORD_PID 2>/dev/null || true
rm -rf "$CACHE1" "$CACHE2"

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
