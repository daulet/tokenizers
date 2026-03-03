FROM ghcr.io/cross-rs/x86_64-unknown-linux-musl:main

RUN apt-get update && apt-get install -y g++-x86-64-linux-gnu && rm -rf /var/lib/apt/lists/*
