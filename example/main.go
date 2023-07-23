package main

import (
	"fmt"

	"github.com/daulet/tokenizers"
)

func main() {
	tk, err := tokenizers.FromFile("./test/data/bert-base-uncased.json")
	if err != nil {
		panic(err)
	}
	// release native resources
	defer tk.Close()
	fmt.Println("Vocab size:", tk.VocabSize())
	// Vocab size: 30522
	encoding := tk.Encode("brown fox jumps over the lazy dog", false)
	fmt.Println(encoding.IDs, encoding.Tokens)
	// [2829 4419 14523 2058 1996 13971 3899] [brown fox jumps over the lazy dog]
	encoding = tk.Encode("brown fox jumps over the lazy dog", true)
	fmt.Println(encoding.IDs, encoding.Tokens)
	// [101 2829 4419 14523 2058 1996 13971 3899 102] [[CLS] brown fox jumps over the lazy dog [SEP]]
	fmt.Println(tk.Decode([]uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899}, true))
	// brown fox jumps over the lazy dog
}
