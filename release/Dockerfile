# syntax=docker/dockerfile:1.3

FROM rust:1.71 as builder-rust
ARG TARGETPLATFORM
WORKDIR /workspace
COPY ./src ./src
COPY ./Cargo.toml ./Cargo.toml
COPY ./Cargo.lock ./Cargo.lock
RUN cargo build --release

FROM golang:1.21 as builder-go
ARG TARGETPLATFORM
WORKDIR /workspace
COPY ./release .
COPY ./test/data ./test/data
RUN go mod download
COPY --from=builder-rust \
    /workspace/target/release/libtokenizers.a \
    /go/pkg/mod/github.com/daulet/tokenizers@v0.6.0/libtokenizers.a
RUN go run main.go
