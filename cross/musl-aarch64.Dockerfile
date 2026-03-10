FROM ghcr.io/cross-rs/aarch64-unknown-linux-musl:0.2.5

RUN apt-get update && apt-get install -y g++-aarch64-linux-gnu && rm -rf /var/lib/apt/lists/*
