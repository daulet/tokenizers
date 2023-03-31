package tokenizer_test

import (
	"testing"

	"github.com/daulet/tokenizer"

	"github.com/stretchr/testify/assert"
)

// TODO test for leaks
// TODO add benchmark for longer input

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

func BenchmarkEncode(b *testing.B) {
	tk := tokenizer.FromFile("./test/data/bert-base-uncased.json")
	defer tk.Close()
	expected := []uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokens := tk.Encode("brown fox jumps over the lazy dog")
		assert.Equal(b, expected, tokens)
	}
}

func BenchmarkDecode(b *testing.B) {
	tk := tokenizer.FromFile("./test/data/bert-base-uncased.json")
	defer tk.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		str := tk.Decode([]uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899})
		assert.Equal(b, "brown fox jumps over the lazy dog", str)
	}
}
