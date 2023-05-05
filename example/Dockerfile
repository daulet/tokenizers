# syntax=docker/dockerfile:1.3

FROM golang:1.19 as builder-go
ARG TARGETPLATFORM
WORKDIR /workspace
RUN curl -fsSL https://github.com/daulet/tokenizers/releases/latest/download/libtokenizers.$(echo ${TARGETPLATFORM} | tr / -).tar.gz | tar xvz
COPY ./example .
COPY ./test/data ./test/data
RUN --mount=type=cache,target=/root/.cache/go-build \
    --mount=type=cache,target=/var/cache/go,id=${TARGETPLATFORM} \
    CGO_ENABLED=1 CGO_LDFLAGS="-Wl,--copy-dt-needed-entries" go run main.go
