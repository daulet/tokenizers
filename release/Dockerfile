# syntax=docker/dockerfile:1.4

FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/rust:1.87 as builder-rust
ARG TARGETPLATFORM
WORKDIR /workspace
COPY ./benches ./benches
COPY ./src ./src
COPY ./Cargo.toml ./Cargo.toml
COPY ./Cargo.lock ./Cargo.lock
RUN cargo build --release

FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/library/golang:1.21.5 as builder-go
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
RUN go mod tidy
CMD go run -ldflags="-extldflags '-L./tokenizers/lib/${DOCKER_TARGETPLATFORM}'" .
