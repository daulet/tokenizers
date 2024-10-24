package main

import (
	"fmt"
	"github.com/daulet/tokenizers"
)

func main() {
	tk, err := tokenizers.FromFile("../test/data/bert-base-uncased.json")
	if err != nil {
		panic(err)
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

	var encodeOptions []tokenizers.EncodeOption
	encodeOptions = append(encodeOptions, tokenizers.WithReturnTypeIDs())
	encodeOptions = append(encodeOptions, tokenizers.WithReturnAttentionMask())
	encodeOptions = append(encodeOptions, tokenizers.WithReturnTokens())
	encodeOptions = append(encodeOptions, tokenizers.WithReturnOffsets())
	encodeOptions = append(encodeOptions, tokenizers.WithReturnSpecialTokensMask())

	// Or just basically
	// encodeOptions = append(encodeOptions, tokenizers.WithReturnAllAttributes())

	encodingResponse := tk.EncodeWithOptions("brown fox jumps over the lazy dog", true, encodeOptions...)
	fmt.Println(encodingResponse.IDs)
	// [2829 4419 14523 2058 1996 13971 3899]
	fmt.Println(encodingResponse.TypeIDs)
	// [0 0 0 0 0 0 0]
	fmt.Println(encodingResponse.SpecialTokensMask)
	// [0 0 0 0 0 0 0]
	fmt.Println(encodingResponse.AttentionMask)
	// [1 1 1 1 1 1 1]
	fmt.Println(encodingResponse.Tokens)
	// [brown fox jumps over the lazy dog]
	fmt.Println(encodingResponse.Offsets)
	// [[0 5] [6 9] [10 15] [16 20] [21 24] [25 29] [30 33]]

	tokenizerPath := "../huggingface-tokenizers/google-bert/bert-base-uncased"
	tkFromHf, errHf := tokenizers.LoadTokenizerFromHuggingFace("google-bert/bert-base-uncased", &tokenizerPath, nil)
	if errHf != nil {
		panic(errHf)
	}
	// release native resources
	defer tkFromHf.Close()

	encodingResponseHf := tkFromHf.EncodeWithOptions("brown fox jumps over the lazy dog", true, encodeOptions...)
	fmt.Println(encodingResponseHf.IDs)
	// [101 2829 4419 14523 2058 1996 13971 3899 102]
	fmt.Println(encodingResponseHf.TypeIDs)
	// [0 0 0 0 0 0 0 0 0]
	fmt.Println(encodingResponseHf.SpecialTokensMask)
	// [1 0 0 0 0 0 0 0 1]
	fmt.Println(encodingResponseHf.AttentionMask)
	// [1 1 1 1 1 1 1 1 1]
	fmt.Println(encodingResponseHf.Tokens)
	// [[CLS] brown fox jumps over the lazy dog [SEP]]
	fmt.Println(encodingResponseHf.Offsets)
	// [[0 0] [0 5] [6 9] [10 15] [16 20] [21 24] [25 29] [30 33] [0 0]]
}
