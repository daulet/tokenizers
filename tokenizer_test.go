package tokenizer_test

import (
	"math/rand"
	"testing"

	"github.com/daulet/tokenizer"

	"github.com/stretchr/testify/assert"
)

// TODO test for leaks
// TODO expose as an API
const vocabSize = 30522

func TestEncode(t *testing.T) {
	tk := tokenizer.FromFile("./test/data/bert-base-uncased.json")
	defer tk.Close()
	tokens := tk.Encode("brown fox jumps over the lazy dog")
	assert.Equal(t, []uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899}, tokens)
}

func TestDecode(t *testing.T) {
	tk := tokenizer.FromFile("./test/data/bert-base-uncased.json")
	defer tk.Close()
	str := tk.Decode([]uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899})
	assert.Equal(t, "brown fox jumps over the lazy dog", str)
}

func BenchmarkEncodeNTimes(b *testing.B) {
	tk := tokenizer.FromFile("./test/data/bert-base-uncased.json")
	defer tk.Close()
	expected := []uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokens := tk.Encode("brown fox jumps over the lazy dog")
		assert.Equal(b, expected, tokens)
	}
}

func BenchmarkEncodeNChars(b *testing.B) {
	tk := tokenizer.FromFile("./test/data/bert-base-uncased.json")
	defer tk.Close()
	input := make([]rune, 0, b.N)
	for i := 0; i < b.N; i++ {
		input = append(input, rune(rand.Uint32()%vocabSize))
	}
	str := string(input)
	b.ResetTimer()
	tokens := tk.Encode(str)
	assert.Greater(b, len(tokens), 0)
}

func BenchmarkDecodeNTimes(b *testing.B) {
	tk := tokenizer.FromFile("./test/data/bert-base-uncased.json")
	defer tk.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		str := tk.Decode([]uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899})
		assert.Equal(b, "brown fox jumps over the lazy dog", str)
	}
}

func BenchmarkDecodeNTokens(b *testing.B) {
	tk := tokenizer.FromFile("./test/data/bert-base-uncased.json")
	defer tk.Close()
	input := make([]uint32, 0, b.N)
	for i := 0; i < b.N; i++ {
		input = append(input, rand.Uint32()%vocabSize)
	}
	b.ResetTimer()
	text := tk.Decode(input)
	// a token is one or more characters
	assert.Greater(b, len(text), b.N)
}
