# syntax=docker/dockerfile:1.3

FROM rust:1.69 as builder-rust
ARG TARGETPLATFORM
WORKDIR /workspace
COPY ./lib .
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=${TARGETPLATFORM} \
    --mount=type=cache,target=/root/target,id=${TARGETPLATFORM} \
    cargo build --release

FROM golang:1.19 as builder-go
ARG TARGETPLATFORM
WORKDIR /workspace
COPY --from=builder-rust /workspace/target/release/libtokenizers.a .
COPY ./release .
COPY ./test/data ./test/data
RUN --mount=type=cache,target=/root/.cache/go-build \
    --mount=type=cache,target=/var/cache/go,id=${TARGETPLATFORM} \
    CGO_ENABLED=1 CGO_LDFLAGS="-Wl,--copy-dt-needed-entries" go run main.go
