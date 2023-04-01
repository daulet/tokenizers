# Tokenizers

Go bindings for the [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) library.

## Installation
```bash
// TODO figure out distribution
make build
```

## Getting started

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
// [2829 4419 14523 2058 1996 13971 3899]
fmt.Println(tk.Encode("brown fox jumps over the lazy dog", true))
// [101 2829 4419 14523 2058 1996 13971 3899 102]
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
