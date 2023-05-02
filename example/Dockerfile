FROM rust:1.69 as builder-rust
WORKDIR /workspace
COPY ./lib .
RUN cargo build --release

FROM golang:1.19 as builder-go
WORKDIR /workspace
COPY --from=builder-rust /workspace/target/release/libtokenizers.a .
COPY ./example .
COPY ./test/data ./test/data
RUN CGO_ENABLED=1 CGO_LDFLAGS="-Wl,--copy-dt-needed-entries" go run main.go
