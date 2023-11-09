# syntax=docker/dockerfile:1.3

FROM rust:1.71 as builder-rust
ARG TARGETPLATFORM
WORKDIR /workspace
COPY ./src ./src
COPY ./Cargo.toml ./Cargo.toml
COPY ./Cargo.lock ./Cargo.lock
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=${TARGETPLATFORM} \
    --mount=type=cache,target=/root/target,id=${TARGETPLATFORM} \
    cargo build --release

FROM golang:1.21 as builder-go
ARG TARGETPLATFORM
WORKDIR /workspace
COPY ./release .
COPY ./test/data ./test/data
RUN go mod download
COPY --from=builder-rust \
    /workspace/target/release/libtokenizers.a \
    /go/pkg/mod/github.com/daulet/tokenizers@v0.5.1/libtokenizers.a
RUN CGO_ENABLED=1 CGO_LDFLAGS="-Wl,--copy-dt-needed-entries" go run main.go
