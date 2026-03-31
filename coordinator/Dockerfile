FROM rust:1.93-bookworm AS builder

WORKDIR /app
COPY . .

# Build coordinator and node CLI (node needed for bootstrap)
RUN cargo build --release -p distrain-coordinator -p distrain-node

# Runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates openssh-server curl libgomp1 \
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
