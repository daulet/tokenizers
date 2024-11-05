package main

import (
	"fmt"
	"log"

	"github.com/daulet/tokenizers"
)

func simple() error {
	tk, err := tokenizers.FromFile("../test/data/bert-base-uncased.json")
	if err != nil {
		return err
	}
	// release native resources
	defer tk.Close()

	fmt.Println("Vocab size:", tk.VocabSize())
	// Vocab size: 30522
	fmt.Println(tk.Encode("brown fox jumps over the lazy dog", false))
	// [2829 4419 14523 2058 1996 13971 3899] [brown fox jumps over the lazy dog]
	fmt.Println(tk.Encode("brown fox jumps over the lazy dog", true))
	// [101 2829 4419 14523 2058 1996 13971 3899 102] [[CLS] brown fox jumps over the lazy dog [SEP]]
	fmt.Println(tk.Decode([]uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899}, true))
	// brown fox jumps over the lazy dog
	return nil
}

func advanced() error {
	// Load tokenizer from local config file
	tk, err := tokenizers.FromFile("../test/data/bert-base-uncased.json")
	if err != nil {
		return err
	}
	defer tk.Close()

	// Load pretrained tokenizer from HuggingFace
	tokenizerPath := "./.cache/tokenizers/google-bert/bert-base-uncased"
	tkFromHf, err := tokenizers.FromPretrained("google-bert/bert-base-uncased", &tokenizerPath, nil)
	if err != nil {
		return err
	}
	defer tkFromHf.Close()

	// Encode with specific options
	encodeOptions := []tokenizers.EncodeOption{
		tokenizers.WithReturnTypeIDs(),
		tokenizers.WithReturnAttentionMask(),
		tokenizers.WithReturnTokens(),
		tokenizers.WithReturnOffsets(),
		tokenizers.WithReturnSpecialTokensMask(),
	}
	// Or simply:
	// encodeOptions = append(encodeOptions, tokenizers.WithReturnAllAttributes())

	// regardless of how the tokenizer was initialized, the output is the same
	for _, tkzr := range []*tokenizers.Tokenizer{tk, tkFromHf} {
		encodingResponse := tkzr.EncodeWithOptions("brown fox jumps over the lazy dog", true, encodeOptions...)
		fmt.Println(encodingResponse.IDs)
		// [101 2829 4419 14523 2058 1996 13971 3899 102]
		fmt.Println(encodingResponse.TypeIDs)
		// [0 0 0 0 0 0 0 0 0]
		fmt.Println(encodingResponse.SpecialTokensMask)
		// [1 0 0 0 0 0 0 0 1]
		fmt.Println(encodingResponse.AttentionMask)
		// [1 1 1 1 1 1 1 1 1]
		fmt.Println(encodingResponse.Tokens)
		// [[CLS] brown fox jumps over the lazy dog [SEP]]
		fmt.Println(encodingResponse.Offsets)
		// [[0 0] [0 5] [6 9] [10 15] [16 20] [21 24] [25 29] [30 33] [0 0]]
	}
	return nil
}

func main() {
	if err := simple(); err != nil {
		log.Fatal(err)
	}
	if err := advanced(); err != nil {
		log.Fatal(err)
	}
}
