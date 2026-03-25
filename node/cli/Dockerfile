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

# Entrypoint: setup SSH keys from RunPod env, start sshd, keep alive
RUN echo '#!/bin/bash\n\
mkdir -p /root/.ssh\n\
if [ -n "$PUBLIC_KEY" ]; then\n\
  echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys\n\
  chmod 600 /root/.ssh/authorized_keys\n\
fi\n\
/usr/sbin/sshd\n\
exec sleep infinity' > /start.sh && chmod +x /start.sh

CMD ["/start.sh"]
