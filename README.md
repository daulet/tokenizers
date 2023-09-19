# Tokenizers

Go bindings for the [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) library.

## Installation

* Option 1: `make build` to build `libtokenizers.a` that you need to run your application that uses bindings.
* Option 2: Here: [import-this-package-in-your-projects](https://github.com/sunhailin-Leo/tokenizers#import-this-package-in-your-projects)

**Both two options, you need a Rust environment to build `libtokenizers.a`.**

## Getting started

TLDR: [working example](example/main.go).

Load a tokenizer from a JSON config:
```go
import "github.com/sunhailin-Leo/tokenizers"

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
// &{[2829 4419 14523 2058 1996 13971 3899] [] [] [] [brown fox jumps over the lazy dog] []}
fmt.Println(tk.Encode("brown fox jumps over the lazy dog", true))
// &{[101 2829 4419 14523 2058 1996 13971 3899 102] [] [] [] [[CLS] brown fox jumps over the lazy dog [SEP]] []}
fmt.Println(tk.Decode([]uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899}, true))
// brown fox jumps over the lazy dog
```

Encode Result Struct:
```go
type Offset struct {
	Start uint32
	End   uint32
}

type TokenizerResult struct {
	TokenIds          []uint32
	TypeIds           []uint32
	SpecialTokensMask []uint32
	AttentionMask     []uint32
	Tokens            []string
	Offsets           []Offset
}
```

## Benchmarks
```bash
go test . -bench=. -benchmem -benchtime=10s

goos: darwin
goarch: amd64
pkg: github.com/sunhailin-Leo/tokenizers
cpu: Intel(R) Core(TM) i7-8850H CPU @ 2.60GHz
BenchmarkEncodeNTimes-12                  288102             39637 ns/op             440 B/op         14 allocs/op
BenchmarkEncodeWithOptionNTimes-12        316530             38884 ns/op             552 B/op         16 allocs/op
BenchmarkDecodeNTimes-12                  714762             17110 ns/op              96 B/op          3 allocs/op
PASS
ok      github.com/sunhailin-Leo/tokenizers     38.997s
```

## Import this package in your projects
1、Find your project package main
```go
package main

//go:generate make -C $GOPATH/pkg/mod/github.com/sunhailin-!leo/tokenizers@v0.6.0 
// The last version tag must follow go.mod version.

// Your project imports.
import ""

func main() {
	// Your project code here
}
```
2、Before your first run or build, you must run `go generate` in where did you want to execute `go run / build`
