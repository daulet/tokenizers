# syntax=docker/dockerfile:1.3

FROM rust:1.87 as builder-rust
ARG TARGETPLATFORM
WORKDIR /workspace
COPY ./benches ./benches
COPY ./src ./src
COPY ./Cargo.toml ./Cargo.toml
COPY ./Cargo.lock ./Cargo.lock
RUN cargo build --release

FROM golang:1.21 as builder-go
ARG DOCKER_TARGETPLATFORM
WORKDIR /workspace
COPY ./release/go.mod   .
COPY ./release/main.go  .
# can't rely on tokenizers module because latest release doesn't include recent changes
COPY tokenizer.go       ./tokenizers/
COPY tokenizers.h       ./tokenizers/
COPY --from=builder-rust \
    /workspace/target/release/libtokenizers.a \
    ./tokenizers/lib/${DOCKER_TARGETPLATFORM}/
COPY ./test/data        ./test/data
RUN go run -ldflags="-extldflags '-L./tokenizers/lib/${DOCKER_TARGETPLATFORM}'" .
