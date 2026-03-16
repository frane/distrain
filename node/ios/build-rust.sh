#!/usr/bin/env bash
# Build the Distrain FFI library for iOS targets.
#
# Prerequisites:
#   rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios
#
# Usage:
#   ./build-rust.sh [release|debug]

set -euo pipefail

MODE=${1:-release}
FLAGS=""
PROFILE="debug"
if [ "$MODE" = "release" ]; then
    FLAGS="--release"
    PROFILE="release"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/../.."
OUTPUT_DIR="$SCRIPT_DIR/libs"

echo "Building distrain-ffi for iOS ($MODE)..."

cd "$PROJECT_ROOT"

# Build for device (arm64)
cargo build $FLAGS -p distrain-ffi --target aarch64-apple-ios

# Build for simulator (arm64 Apple Silicon + x86_64 Intel)
cargo build $FLAGS -p distrain-ffi --target aarch64-apple-ios-sim
cargo build $FLAGS -p distrain-ffi --target x86_64-apple-ios

# Create universal simulator library
mkdir -p "$OUTPUT_DIR"

cp "target/aarch64-apple-ios/$PROFILE/libdistrain_ffi.a" "$OUTPUT_DIR/libdistrain_ffi-ios.a"

lipo -create \
    "target/aarch64-apple-ios-sim/$PROFILE/libdistrain_ffi.a" \
    "target/x86_64-apple-ios/$PROFILE/libdistrain_ffi.a" \
    -output "$OUTPUT_DIR/libdistrain_ffi-sim.a"

# Copy header
cp "$PROJECT_ROOT/node/ffi/distrain.h" "$OUTPUT_DIR/"

echo "Libraries:"
ls -la "$OUTPUT_DIR"/*.a
echo "Header: $OUTPUT_DIR/distrain.h"
echo ""
echo "Add to Xcode project:"
echo "  1. Drag libdistrain_ffi-ios.a into the project"
echo "  2. Add distrain.h as Objective-C Bridging Header"
echo "  3. Link against: libdistrain_ffi-ios.a (device) or libdistrain_ffi-sim.a (simulator)"
