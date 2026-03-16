FROM rust:1.93-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock* ./
COPY core/ core/
COPY node/cli/ node/cli/

RUN cargo build --release -p distrain-node

# Runtime — pure Rust, no Python dependency
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/distrain-node /usr/local/bin/distrain-node

CMD ["distrain-node", "start", "--config", "/etc/distrain/node.toml"]
