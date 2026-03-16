#!/usr/bin/env bash
# Build the Distrain FFI library for Android targets.
#
# Prerequisites:
#   cargo install cargo-ndk
#   Android NDK installed (set ANDROID_NDK_HOME)
#
# Usage:
#   ./build-rust.sh [release|debug]

set -euo pipefail

MODE=${1:-release}
FLAGS=""
if [ "$MODE" = "release" ]; then
    FLAGS="--release"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."
OUTPUT_DIR="$SCRIPT_DIR/app/src/main/jniLibs"

echo "Building distrain-ffi for Android ($MODE)..."

cd "$PROJECT_ROOT"

# Build for all major Android ABIs
cargo ndk \
    -t arm64-v8a \
    -t armeabi-v7a \
    -t x86_64 \
    -o "$OUTPUT_DIR" \
    build $FLAGS -p distrain-ffi

echo "Libraries written to $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"/*/libdistrain_ffi.so 2>/dev/null || echo "(no .so files found — check build output)"
