package main

import (
	"fmt"
	"log"
	"strings"

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
	tkFromHf, err := tokenizers.FromPretrained("google-bert/bert-base-uncased", tokenizers.WithCacheDir("./.cache/tokenizers"))
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

func tiktoken() error {
	// Define the pattern for tiktoken tokenization (same as used in Rust tests)
	pattern := strings.Join([]string{
		`[\p{Han}]+`,
		`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
		`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
		`\p{N}{1,3}`,
		` ?[^\s\p{L}\p{N}]+[\r\n]*`,
		`\s*[\r\n]+`,
		`\s+(?!\S)`,
		`\s+`,
	}, "|")

	fmt.Println("=== Tiktoken Tokenizer Example ===")

	// Create a tiktoken tokenizer
	tk, err := tokenizers.FromTiktoken(
		"../test/data/kimi-k2-instruct/tiktoken.model",
		"../test/data/kimi-k2-instruct/tokenizer_config.json",
		pattern,
	)
	if err != nil {
		return err
	}
	defer tk.Close()

	// Test text with both English and Chinese
	text := "Hello, world! 你好，世界！"
	fmt.Printf("\nOriginal text: %s\n", text)

	// Encode the text
	ids, tokens := tk.Encode(text, false)
	fmt.Printf("Token IDs: %v\n", ids)
	if len(tokens) > 0 {
		fmt.Printf("Tokens: %v\n", tokens)
	}

	// Decode back to text
	decoded := tk.Decode(ids, false)
	fmt.Printf("Decoded text: %s\n", decoded)

	// Display vocab size
	vocabSize := tk.VocabSize()
	fmt.Printf("\nVocab size: %d\n", vocabSize)

	// Test with special tokens
	fmt.Println("\n--- Testing with special tokens ---")
	idsWithSpecial, _ := tk.Encode(text, true)
	fmt.Printf("Token IDs (with special): %v\n", idsWithSpecial)

	decodedWithSpecial := tk.Decode(idsWithSpecial, false)
	fmt.Printf("Decoded (with special): %s\n", decodedWithSpecial)

	decodedSkipSpecial := tk.Decode(idsWithSpecial, true)
	fmt.Printf("Decoded (skip special): %s\n", decodedSkipSpecial)

	// Test encoding text that contains special tokens
	fmt.Println("\n--- Testing text containing special tokens ---")
	textWithSpecialTokens := "[BOS] Hello, world! [EOS]"
	fmt.Printf("Text with special tokens: %s\n", textWithSpecialTokens)

	// Encode with special tokens enabled - should recognize [BOS] and [EOS] as special
	idsSpecial, _ := tk.Encode(textWithSpecialTokens, true)
	fmt.Printf("Token IDs (special tokens enabled): %v\n", idsSpecial)

	// Encode with special tokens disabled - should treat [BOS] and [EOS] as regular text
	idsNoSpecial, _ := tk.Encode(textWithSpecialTokens, false)
	fmt.Printf("Token IDs (special tokens disabled): %v\n", idsNoSpecial)

	return nil
}

func main() {
	if err := simple(); err != nil {
		log.Fatal(err)
	}
	if err := advanced(); err != nil {
		log.Fatal(err)
	}
	if err := tiktoken(); err != nil {
		log.Fatal(err)
	}
}
