FROM rust:1.93-bookworm AS builder

WORKDIR /app
COPY . .

RUN cargo build --release -p distrain-node

# Runtime — pure Rust, no Python dependency
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libvulkan1 openssh-server \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd

ENV VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
ENV XDG_RUNTIME_DIR=/tmp

COPY --from=builder /app/target/release/distrain-node /usr/local/bin/distrain-node

CMD ["distrain-node", "start", "--config", "/etc/distrain/node.toml"]
