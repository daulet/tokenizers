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
	fmt.Println(tk.Encode("brown fox jumps over the lazy dog"))
	// [2829 4419 14523 2058 1996 13971 3899]
	fmt.Println(tk.Decode(tk.Encode("brown fox jumps over the lazy dog")))
	// brown fox jumps over the lazy dog
}
