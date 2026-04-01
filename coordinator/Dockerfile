FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
COPY . .

# Build with CUDA for GPU aggregation. Pin CUDA 12.4 for driver compat.
ENV CUDARC_CUDA_VERSION=12040
RUN cargo build --release -p distrain-coordinator --features cuda -p distrain-node --features cuda

# Runtime — CUDA devel (cubecl compiles kernels at runtime via NVRTC)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates openssh-server curl \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd

COPY --from=builder /app/target/release/coordinator /usr/local/bin/coordinator
COPY --from=builder /app/target/release/distrain-node /usr/local/bin/distrain-node

# Install MinIO server (amd64)
RUN curl -sSL -o /usr/local/bin/minio https://dl.min.io/server/minio/release/linux-amd64/minio \
    && chmod +x /usr/local/bin/minio

COPY coordinator/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000 9000 22

CMD ["/entrypoint.sh"]
