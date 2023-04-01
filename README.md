# Tokenizers

Go bindings for the [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) library.

## Installation
```bash
// TODO figure out distribution
```

## Getting started
```go
import "github.com/daulet/tokenizers"

tk, err := tokenizers.FromFile("./data/bert-base-uncased.json")
if err != nil {
    return err
}
// release native resources
defer tk.Close()
fmt.Println(tk.Encode("brown fox jumps over the lazy dog"))
// [2829 4419 14523 2058 1996 13971 3899]
```

## Benchmarks
```bash
go test . -bench=. -benchmem -benchtime=10s

goos: darwin
goarch: arm64
pkg: github.com/daulet/tokenizer
BenchmarkEncodeNTimes-10     	  985678	     12023 ns/op	     132 B/op	       7 allocs/op
BenchmarkEncodeNChars-10      1000000000	     2.442 ns/op	       0 B/op	       0 allocs/op
BenchmarkDecodeNTimes-10     	 6762982	      1767 ns/op	     128 B/op	       5 allocs/op
BenchmarkDecodeNTokens-10    	65058678	     219.8 ns/op	       7 B/op	       0 allocs/op
PASS
ok  	github.com/daulet/tokenizer	69.993s
```
