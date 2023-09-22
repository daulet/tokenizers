# Tokenizers

Go bindings for the [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) library.

## Installation

`make build` to build `libtokenizers.a` that you need to run your application that uses bindings.

### Using pre-built binaries

Build your Go application using pre-built native binaries: `docker build --platform=linux/amd64 -f example/Dockerfile .`

Available binaries:
* [darwin-arm64](https://github.com/daulet/tokenizers/releases/latest/download/libtokenizers.darwin-arm64.tar.gz)
* [linux-arm64](https://github.com/daulet/tokenizers/releases/latest/download/libtokenizers.linux-arm64.tar.gz)
* [linux-amd64](https://github.com/daulet/tokenizers/releases/latest/download/libtokenizers.linux-amd64.tar.gz)

## Getting started

TLDR: [working example](example/main.go).

Load a tokenizer from a JSON config:
```go
import "github.com/daulet/tokenizers"

tk, err := tokenizers.FromFile("./data/bert-base-uncased.json")
if err != nil {
    return err
}
// release native resources
defer tk.Close()
```

Encode text and decode tokens:
```go
fmt.Println("Vocab size:", tk.VocabSize())
// Vocab size: 30522
fmt.Println(tk.Encode("brown fox jumps over the lazy dog", false))
// [2829 4419 14523 2058 1996 13971 3899] [brown fox jumps over the lazy dog]
fmt.Println(tk.Encode("brown fox jumps over the lazy dog", true))
// [101 2829 4419 14523 2058 1996 13971 3899 102] [[CLS] brown fox jumps over the lazy dog [SEP]]
fmt.Println(tk.Decode([]uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899}, true))
// brown fox jumps over the lazy dog
```

## Benchmarks
```bash
go test . -bench=. -benchmem -benchtime=10s

goos: darwin
goarch: arm64
pkg: github.com/daulet/tokenizers
BenchmarkEncodeNTimes-10     	  996556	     11851 ns/op	     116 B/op	       6 allocs/op
BenchmarkEncodeNChars-10      1000000000	     2.446 ns/op	       0 B/op	       0 allocs/op
BenchmarkDecodeNTimes-10     	 7286056	      1657 ns/op	     112 B/op	       4 allocs/op
BenchmarkDecodeNTokens-10    	65191378	     211.0 ns/op	       7 B/op	       0 allocs/op
PASS
ok  	github.com/daulet/tokenizers	126.681s
```

## Cross compilation
- `cargo install cross --git https://github.com/cross-rs/cross`
- `make release`
