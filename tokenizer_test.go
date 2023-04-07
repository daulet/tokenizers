package tokenizers_test

import (
	_ "embed"
	"math/rand"
	"testing"

	"github.com/daulet/tokenizers"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

//go:embed test/data/sentence-transformers-labse.json
var embeddedBytes []byte

// TODO test for leaks

func TestInvalidConfigPath(t *testing.T) {
	_, err := tokenizers.FromFile("./non-existent.json")
	require.Error(t, err)
}

func TestEmbeddingConfig(t *testing.T) {
	tk, err := tokenizers.FromBytes(embeddedBytes)
	require.NoError(t, err)
	defer tk.Close()

	tests := []struct {
		name       string
		str        string
		addSpecial bool
		want       []uint32
	}{
		{
			name:       "without special tokens",
			str:        "brown fox jumps over the lazy dog",
			addSpecial: false,
			want:       []uint32{0xca3f, 0x2f304, 0x5185b, 0x3c54, 0x3a89, 0x35fc3, 0x57b4},
		},
		{
			name:       "with special tokens",
			str:        "brown fox jumps over the lazy dog",
			addSpecial: true,
			want:       []uint32{0x65, 0xca3f, 0x2f304, 0x5185b, 0x3c54, 0x3a89, 0x35fc3, 0x57b4, 0x66},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tk.Encode(tt.str, tt.addSpecial)
			got := tk.Encode(tt.str, tt.addSpecial)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestEncode(t *testing.T) {
	tk, err := tokenizers.FromFile("./test/data/bert-base-uncased.json")
	require.NoError(t, err)
	defer tk.Close()
	tests := []struct {
		name       string
		str        string
		addSpecial bool
		want       []uint32
	}{
		{
			name:       "without special tokens",
			str:        "brown fox jumps over the lazy dog",
			addSpecial: false,
			want:       []uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899},
		},
		{
			name:       "with special tokens",
			str:        "brown fox jumps over the lazy dog",
			addSpecial: true,
			want:       []uint32{101, 2829, 4419, 14523, 2058, 1996, 13971, 3899, 102},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tk.Encode(tt.str, tt.addSpecial)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestDecode(t *testing.T) {
	tk, err := tokenizers.FromFile("./test/data/bert-base-uncased.json")
	require.NoError(t, err)
	defer tk.Close()
	tests := []struct {
		name        string
		tokens      []uint32
		skipSpecial bool
		want        string
	}{
		{
			name:        "without special tokens, skip special tokens",
			tokens:      []uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899},
			skipSpecial: true,
			want:        "brown fox jumps over the lazy dog",
		},
		{
			name:        "with special tokens, skip special tokens",
			tokens:      []uint32{101, 2829, 4419, 14523, 2058, 1996, 13971, 3899, 102},
			skipSpecial: true,
			want:        "brown fox jumps over the lazy dog",
		},
		{
			name:        "without special tokens, don't skip special tokens",
			tokens:      []uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899},
			skipSpecial: false,
			want:        "brown fox jumps over the lazy dog",
		},
		{
			name:        "with special tokens, don't skip special tokens",
			tokens:      []uint32{101, 2829, 4419, 14523, 2058, 1996, 13971, 3899, 102},
			skipSpecial: false,
			want:        "[CLS] brown fox jumps over the lazy dog [SEP]",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tk.Decode(tt.tokens, tt.skipSpecial)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestVocabSize(t *testing.T) {
	tk, err := tokenizers.FromFile("./test/data/bert-base-uncased.json")
	require.NoError(t, err)
	defer tk.Close()
	assert.Equal(t, uint32(30522), tk.VocabSize())
}

func BenchmarkEncodeNTimes(b *testing.B) {
	tk, err := tokenizers.FromFile("./test/data/bert-base-uncased.json")
	require.NoError(b, err)
	defer tk.Close()
	expected := []uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokens := tk.Encode("brown fox jumps over the lazy dog", false)
		assert.Equal(b, expected, tokens)
	}
}

func BenchmarkEncodeNChars(b *testing.B) {
	tk, err := tokenizers.FromFile("./test/data/bert-base-uncased.json")
	require.NoError(b, err)
	defer tk.Close()
	input := make([]rune, 0, b.N)
	for i := 0; i < b.N; i++ {
		input = append(input, rune(rand.Uint32()%tk.VocabSize()))
	}
	str := string(input)
	b.ResetTimer()
	tokens := tk.Encode(str, false)
	assert.Greater(b, len(tokens), 0)
}

func BenchmarkDecodeNTimes(b *testing.B) {
	tk, err := tokenizers.FromFile("./test/data/bert-base-uncased.json")
	require.NoError(b, err)
	defer tk.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		str := tk.Decode([]uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899}, true)
		assert.Equal(b, "brown fox jumps over the lazy dog", str)
	}
}

func BenchmarkDecodeNTokens(b *testing.B) {
	tk, err := tokenizers.FromFile("./test/data/bert-base-uncased.json")
	require.NoError(b, err)
	defer tk.Close()
	input := make([]uint32, 0, b.N)
	for i := 0; i < b.N; i++ {
		input = append(input, rand.Uint32()%tk.VocabSize())
	}
	b.ResetTimer()
	text := tk.Decode(input, true)
	// a token is one or more characters
	assert.Greater(b, len(text), b.N)
}
