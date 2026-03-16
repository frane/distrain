# Distrain — build commands per platform
# Install: cargo install just

default:
    @just --list

# ── Core ──────────────────────────────────────
test:
    cargo test --workspace --exclude distrain-wasm
test-wasm:
    cargo check -p distrain-wasm --target wasm32-unknown-unknown
lint:
    cargo clippy --workspace --exclude distrain-wasm -- -D warnings
check:
    cargo check --workspace --exclude distrain-wasm

# ── Coordinator ───────────────────────────────
build-coordinator:
    cargo build --release -p distrain-coordinator
run-coordinator:
    cargo run -p distrain-coordinator
docker-coordinator:
    docker build -t distrain-coordinator -f coordinator/Dockerfile .

# ── Node CLI (Linux/macOS/Windows) ────────────
build-node:
    cargo build --release -p distrain-node
run-node config="node.toml":
    cargo run -p distrain-node -- start --config {{config}}
bootstrap config="node.toml" preset="tiny":
    cargo run -p distrain-node -- bootstrap --config {{config}} --preset {{preset}}
docker-node:
    docker build -t distrain-node -f node/cli/Dockerfile .

# ── Shared UI ─────────────────────────────────
sync-ui:
    cp node/ui/distrain-ui.css node/ui/distrain-ui.js node/ui/distrain-ui.html node/desktop/frontend/
    cp node/ui/distrain-ui.css node/ui/distrain-ui.js node/ui/distrain-ui.html node/browser/web/

# ── Node Desktop (Tauri — macOS/Windows) ──────
build-desktop: sync-ui
    cd node/desktop/src-tauri && cargo tauri build
dev-desktop: sync-ui
    cd node/desktop/src-tauri && cargo tauri dev

# ── Node Browser (WebAssembly) ────────────────
build-wasm: sync-ui
    RUSTFLAGS=--cfg=web_sys_unstable_apis wasm-pack build node/browser/wasm --target web --out-dir ../web/pkg
serve-browser port="8080": sync-ui
    cd node/browser/web && python3 -m http.server {{port}}

# ── Node Mobile (iOS/Android via FFI) ─────────
build-ffi:
    cargo build --release -p distrain-ffi
build-ios-rust:
    node/ios/build-rust.sh release
build-ios:
    node/ios/build-rust.sh release
generate-ios:
    cd node/ios && xcodegen generate
open-ios:
    open node/ios/DistrainNode.xcodeproj
build-android:
    node/android/build-rust.sh release

# ── Docker Compose (full local stack) ────────
stack-up:
    cd docker && docker compose up --build -d
stack-down:
    cd docker && docker compose down
stack-logs:
    cd docker && docker compose logs -f

# ── Scripts ──────────────────────────────────
prepare-data dataset="fineweb-edu-10bt" output-dir="data/fineweb":
    source .venv/bin/activate && python scripts/prepare_data.py {{dataset}} --output-dir {{output-dir}} --upload
prepare-test:
    source .venv/bin/activate && python scripts/prepare_data.py test --output-dir data/test --num-shards 10 --upload

# ── All ──────────────────────────────────────
build-all: build-coordinator build-node build-ffi
