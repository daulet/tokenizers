# tokenizer

Go bindings for the [HuggingFace Tokenizers library](https://github.com/huggingface/tokenizers).

## Getting started
```go
import "github.com/daulet/tokenizer"

tk := tokenizer.FromFile("./data/bert-base-uncased.json")
// release native resources
defer tk.Close()
fmt.Println(tk.Encode("brown fox jumps over the lazy dog"))
```

## Benchmarks
```bash
go test . -bench=. -benchmem -benchtime=10s

goos: darwin
goarch: arm64
pkg: github.com/daulet/tokenizer
BenchmarkEncode-10    	  911226	     12031 ns/op	     164 B/op	       8 allocs/op
BenchmarkDecode-10    	 6719450	      1735 ns/op	     128 B/op	       5 allocs/op
PASS
ok  	github.com/daulet/tokenizer	24.775s
```
