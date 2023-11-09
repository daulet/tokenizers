# syntax=docker/dockerfile:1.3

FROM golang:1.21 as builder-go
ARG TARGETPLATFORM
WORKDIR /workspace
RUN curl -fsSL https://github.com/daulet/tokenizers/releases/latest/download/libtokenizers.$(echo ${TARGETPLATFORM} | tr / -).tar.gz | tar xvz
COPY ./example .
COPY ./test/data ./test/data
RUN go mod download
COPY ./libtokenizers.a /go/pkg/mod/github.com/daulet/tokenizers@v0.5.1/libtokenizers.a
# mounting Go cache won't work since we mutate it above
RUN CGO_ENABLED=1 CGO_LDFLAGS="-Wl,--copy-dt-needed-entries" go run main.go
