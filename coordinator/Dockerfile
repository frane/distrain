FROM rust:1.93-bookworm AS builder

WORKDIR /app
COPY . .

RUN cargo build --release -p distrain-coordinator

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/coordinator /usr/local/bin/coordinator

ENV PORT=8000 HOST=0.0.0.0
EXPOSE 8000

CMD ["coordinator"]
