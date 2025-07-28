package tokenizers_test

import (
	_ "embed"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	// "github.com/daulet/tokenizers"
	tokenizers "gitlab.alibaba-inc.com/eas/tokenizers_go"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

//go:embed test/data/sentence-transformers-labse.json
var embeddedBytes []byte

// TODO test for leaks
// TODO fuzz tests

func TestInvalidConfigPath(t *testing.T) {
	_, err := tokenizers.FromFile("./non-existent.json")
	require.Error(t, err)
}

func TestEmbeddingConfig(t *testing.T) {
	tk, err := tokenizers.FromBytes(embeddedBytes)
	require.NoError(t, err)
	defer tk.Close()

	tests := []struct {
		name                  string
		str                   string
		addSpecial            bool
		wantIDs               []uint32
		wantTypeIDs           []uint32
		wantTokens            []string
		wantSpecialTokensMask []uint32
		wantAttentionMask     []uint32
		wantOffsets           []tokenizers.Offset
	}{
		{
			name:                  "without special tokens",
			str:                   "brown fox jumps over the lazy dog",
			addSpecial:            false,
			wantIDs:               []uint32{0xca3f, 0x2f304, 0x5185b, 0x3c54, 0x3a89, 0x35fc3, 0x57b4},
			wantTypeIDs:           []uint32{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
			wantTokens:            []string{"brown", "fox", "jumps", "over", "the", "lazy", "dog"},
			wantSpecialTokensMask: []uint32{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
			wantAttentionMask:     []uint32{0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1},
			wantOffsets:           []tokenizers.Offset{{0x0, 0x5}, {0x6, 0x9}, {0xa, 0xf}, {0x10, 0x14}, {0x15, 0x18}, {0x19, 0x1d}, {0x1e, 0x21}},
		},
		{
			name:                  "with special tokens",
			str:                   "brown fox jumps over the lazy dog",
			addSpecial:            true,
			wantIDs:               []uint32{0x65, 0xca3f, 0x2f304, 0x5185b, 0x3c54, 0x3a89, 0x35fc3, 0x57b4, 0x66},
			wantTypeIDs:           []uint32{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
			wantTokens:            []string{"[CLS]", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "[SEP]"},
			wantSpecialTokensMask: []uint32{0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1},
			wantAttentionMask:     []uint32{0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1},
			wantOffsets:           []tokenizers.Offset{{0x0, 0x0}, {0x0, 0x5}, {0x6, 0x9}, {0xa, 0xf}, {0x10, 0x14}, {0x15, 0x18}, {0x19, 0x1d}, {0x1e, 0x21}, {0x0, 0x0}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoding := tk.EncodeWithOptions(tt.str, tt.addSpecial, tokenizers.WithReturnAllAttributes())
			assert.Equal(t, tt.wantIDs, encoding.IDs, "wrong ids")
			assert.Equal(t, tt.wantTypeIDs, encoding.TypeIDs, "wrong type ids")
			assert.Equal(t, tt.wantTokens, encoding.Tokens, "wrong tokens")
			assert.Equal(t, tt.wantSpecialTokensMask, encoding.SpecialTokensMask, "wrong special tokens mask")
			assert.Equal(t, tt.wantAttentionMask, encoding.AttentionMask, "wrong attention mask")
			assert.Equal(t, tt.wantOffsets, encoding.Offsets, "wrong offsets")

			ids, tokens := tk.Encode(tt.str, tt.addSpecial)
			assert.Equal(t, tt.wantIDs, ids, "wrong ids")
			assert.Equal(t, tt.wantTokens, tokens, "wrong tokens")
		})
	}
}

func TestEncodeWithAndWithoutOptions(t *testing.T) {
	tk, err := tokenizers.FromFile("./test/data/bert-base-uncased.json")
	require.NoError(t, err)
	defer tk.Close()
	tests := []struct {
		name                  string
		str                   string
		addSpecial            bool
		wantIDs               []uint32
		wantTypeIDs           []uint32
		wantTokens            []string
		wantSpecialTokensMask []uint32
		wantAttentionMask     []uint32
		wantOffsets           []tokenizers.Offset
	}{
		{
			name:                  "without special tokens",
			str:                   "brown fox jumps over the lazy dog",
			addSpecial:            false,
			wantIDs:               []uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899},
			wantTypeIDs:           []uint32{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
			wantTokens:            []string{"brown", "fox", "jumps", "over", "the", "lazy", "dog"},
			wantSpecialTokensMask: []uint32{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
			wantAttentionMask:     []uint32{0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1},
			wantOffsets:           []tokenizers.Offset{{0x0, 0x5}, {0x6, 0x9}, {0xa, 0xf}, {0x10, 0x14}, {0x15, 0x18}, {0x19, 0x1d}, {0x1e, 0x21}},
		},
		{
			name:                  "with special tokens",
			str:                   "brown fox jumps over the lazy dog",
			addSpecial:            true,
			wantIDs:               []uint32{101, 2829, 4419, 14523, 2058, 1996, 13971, 3899, 102},
			wantTypeIDs:           []uint32{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
			wantTokens:            []string{"[CLS]", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "[SEP]"},
			wantSpecialTokensMask: []uint32{0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1},
			wantAttentionMask:     []uint32{0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1},
			wantOffsets:           []tokenizers.Offset{{0x0, 0x0}, {0x0, 0x5}, {0x6, 0x9}, {0xa, 0xf}, {0x10, 0x14}, {0x15, 0x18}, {0x19, 0x1d}, {0x1e, 0x21}, {0x0, 0x0}},
		},
		{
			name:       "empty string",
			str:        "",
			addSpecial: false,
		},
		{
			name:                  "empty string with special tokens",
			str:                   "",
			addSpecial:            true,
			wantTypeIDs:           []uint32{0x0, 0x0},
			wantSpecialTokensMask: []uint32{0x1, 0x1},
			wantAttentionMask:     []uint32{0x1, 0x1},
			wantIDs:               []uint32{101, 102},
			wantTokens:            []string{"[CLS]", "[SEP]"},
			wantOffsets:           []tokenizers.Offset{{0x0, 0x0}, {0x0, 0x0}},
		},
		{
			name:                  "invalid utf8 string",
			str:                   "\x91D",
			wantIDs:               []uint32{1040},
			wantTypeIDs:           []uint32{0x0},
			wantTokens:            []string{"d"}, // should be \x91D but tokenizer doesn't handle non-utf8 well
			wantOffsets:           []tokenizers.Offset{{0x3, 0x4}},
			addSpecial:            false,
			wantSpecialTokensMask: []uint32{0x0},
			wantAttentionMask:     []uint32{0x1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoding := tk.EncodeWithOptions(tt.str, tt.addSpecial, tokenizers.WithReturnAllAttributes())
			assert.Equal(t, tt.wantIDs, encoding.IDs, "wrong ids")
			assert.Equal(t, tt.wantTypeIDs, encoding.TypeIDs, "wrong type ids")
			assert.Equal(t, tt.wantTokens, encoding.Tokens, "wrong tokens")
			assert.Equal(t, tt.wantSpecialTokensMask, encoding.SpecialTokensMask, "wrong special tokens mask")
			assert.Equal(t, tt.wantAttentionMask, encoding.AttentionMask, "wrong attention mask")
			assert.Equal(t, tt.wantOffsets, encoding.Offsets, "wrong offsets mask")

			ids, tokens := tk.Encode(tt.str, tt.addSpecial)
			assert.Equal(t, tt.wantIDs, ids, "wrong ids")
			assert.Equal(t, tt.wantTokens, tokens, "wrong tokens")
		})
	}
}

func TestEncodeSpecialTokens(t *testing.T) {
	tk, err := tokenizers.FromBytes(embeddedBytes)
	require.NoError(t, err)
	// special tokens are not encoded by default,
	// meaning if input matches a special token, encoding will include the special token
	ids, _ := tk.Encode("[CLS]fox[SEP]", false)
	assert.Equal(t, []uint32{101, 193284, 102}, ids)
	tk.Close()

	tk, err = tokenizers.FromBytes(embeddedBytes, tokenizers.WithEncodeSpecialTokens())
	require.NoError(t, err)
	ids, _ = tk.Encode("[CLS]fox[SEP]", false)
	// assert that special tokens 101 and 102 are not present
	assert.Equal(t, []uint32{164, 304910, 166, 193284, 164, 211703, 166}, ids)
	tk.Close()
}

func TestEncodeOptions(t *testing.T) {
	tk, err := tokenizers.FromFile("./test/data/bert-base-uncased.json")
	require.NoError(t, err)
	defer tk.Close()
	tests := []struct {
		name                  string
		str                   string
		addSpecial            bool
		wantIDs               []uint32
		wantTypeIDs           []uint32
		wantTokens            []string
		wantSpecialTokensMask []uint32
		wantAttentionMask     []uint32
		wantOffsets           []tokenizers.Offset
	}{
		{
			name:                  "without special tokens",
			str:                   "brown fox jumps over the lazy dog",
			addSpecial:            false,
			wantIDs:               []uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899},
			wantTypeIDs:           []uint32{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
			wantTokens:            []string{"brown", "fox", "jumps", "over", "the", "lazy", "dog"},
			wantSpecialTokensMask: []uint32{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
			wantAttentionMask:     []uint32{0x1, 0x1, 0x1, 0x1, 0x1, 0x1, 0x1},
			wantOffsets:           []tokenizers.Offset{{0x0, 0x5}, {0x6, 0x9}, {0xa, 0xf}, {0x10, 0x14}, {0x15, 0x18}, {0x19, 0x1d}, {0x1e, 0x21}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoding := tk.EncodeWithOptions(tt.str, tt.addSpecial)
			assert.Equal(t, tt.wantIDs, encoding.IDs, "wrong ids")
			assert.Equal(t, []uint32(nil), encoding.TypeIDs, "wrong type ids")
			assert.Equal(t, []string(nil), encoding.Tokens, "wrong tokens")
			assert.Equal(t, []uint32(nil), encoding.SpecialTokensMask, "wrong special tokens mask")
			assert.Equal(t, []uint32(nil), encoding.AttentionMask, "wrong attention mask")
			assert.Equal(t, []tokenizers.Offset(nil), encoding.Offsets, "wrong offsets")

			encoding = tk.EncodeWithOptions(tt.str, tt.addSpecial, tokenizers.WithReturnTokens())
			assert.Equal(t, tt.wantIDs, encoding.IDs, "wrong ids")
			assert.Equal(t, []uint32(nil), encoding.TypeIDs, "wrong type ids")
			assert.Equal(t, tt.wantTokens, encoding.Tokens, "wrong tokens")
			assert.Equal(t, []uint32(nil), encoding.SpecialTokensMask, "wrong special tokens mask")
			assert.Equal(t, []uint32(nil), encoding.AttentionMask, "wrong attention mask")
			assert.Equal(t, []tokenizers.Offset(nil), encoding.Offsets, "wrong offsets")

			encoding = tk.EncodeWithOptions(tt.str, tt.addSpecial, tokenizers.WithReturnTypeIDs())
			assert.Equal(t, tt.wantIDs, encoding.IDs, "wrong ids")
			assert.Equal(t, tt.wantTypeIDs, encoding.TypeIDs, "wrong type ids")
			assert.Equal(t, []string(nil), encoding.Tokens, "wrong tokens")
			assert.Equal(t, []uint32(nil), encoding.SpecialTokensMask, "wrong special tokens mask")
			assert.Equal(t, []uint32(nil), encoding.AttentionMask, "wrong attention mask")
			assert.Equal(t, []tokenizers.Offset(nil), encoding.Offsets, "wrong offsets")

			encoding = tk.EncodeWithOptions(tt.str, tt.addSpecial, tokenizers.WithReturnSpecialTokensMask())
			assert.Equal(t, tt.wantIDs, encoding.IDs, "wrong ids")
			assert.Equal(t, []uint32(nil), encoding.TypeIDs, "wrong type ids")
			assert.Equal(t, []string(nil), encoding.Tokens, "wrong tokens")
			assert.Equal(t, tt.wantSpecialTokensMask, encoding.SpecialTokensMask, "wrong special tokens mask")
			assert.Equal(t, []uint32(nil), encoding.AttentionMask, "wrong attention mask")
			assert.Equal(t, []tokenizers.Offset(nil), encoding.Offsets, "wrong offsets")

			encoding = tk.EncodeWithOptions(tt.str, tt.addSpecial, tokenizers.WithReturnAttentionMask())
			assert.Equal(t, tt.wantIDs, encoding.IDs, "wrong ids")
			assert.Equal(t, []uint32(nil), encoding.TypeIDs, "wrong type ids")
			assert.Equal(t, []string(nil), encoding.Tokens, "wrong tokens")
			assert.Equal(t, []uint32(nil), encoding.SpecialTokensMask, "wrong special tokens mask")
			assert.Equal(t, tt.wantAttentionMask, encoding.AttentionMask, "wrong attention mask")
			assert.Equal(t, []tokenizers.Offset(nil), encoding.Offsets, "wrong offsets")

			encoding = tk.EncodeWithOptions(tt.str, tt.addSpecial, tokenizers.WithReturnOffsets())
			assert.Equal(t, tt.wantIDs, encoding.IDs, "wrong ids")
			assert.Equal(t, []uint32(nil), encoding.TypeIDs, "wrong type ids")
			assert.Equal(t, []string(nil), encoding.Tokens, "wrong tokens")
			assert.Equal(t, []uint32(nil), encoding.SpecialTokensMask, "wrong special tokens mask")
			assert.Equal(t, []uint32(nil), encoding.AttentionMask, "wrong attention mask")
			assert.Equal(t, tt.wantOffsets, encoding.Offsets, "wrong offsets")
		})
	}
}

func TestEncodeWithTruncation(t *testing.T) {
	tests := []struct {
		name       string
		str        string
		addSpecial bool
		maxLen     int
		dir        tokenizers.TruncationDirection
		wantIDs    []uint32
		wantTokens []string
	}{
		{
			name:       "without special tokens, left truncation",
			str:        "brown fox jumps over the lazy dog",
			addSpecial: false,
			maxLen:     5,
			dir:        tokenizers.TruncationDirectionLeft,
			wantIDs:    []uint32{0x5185b, 0x3c54, 0x3a89, 0x35fc3, 0x57b4},
			wantTokens: []string{"jumps", "over", "the", "lazy", "dog"},
		},
		{
			name:       "without special tokens, right truncation",
			str:        "brown fox jumps over the lazy dog",
			addSpecial: false,
			maxLen:     5,
			dir:        tokenizers.TruncationDirectionRight,
			wantIDs:    []uint32{0xca3f, 0x2f304, 0x5185b, 0x3c54, 0x3a89},
			wantTokens: []string{"brown", "fox", "jumps", "over", "the"},
		},
		{
			name:       "with special tokens, left truncation",
			str:        "brown fox jumps over the lazy dog",
			addSpecial: true,
			maxLen:     5,
			dir:        tokenizers.TruncationDirectionLeft,
			wantIDs:    []uint32{0x65, 0x3a89, 0x35fc3, 0x57b4, 0x66},
			wantTokens: []string{"[CLS]", "the", "lazy", "dog", "[SEP]"},
		},
		{
			name:       "with special tokens, right truncation",
			str:        "brown fox jumps over the lazy dog",
			addSpecial: true,
			maxLen:     5,
			dir:        tokenizers.TruncationDirectionRight,
			wantIDs:    []uint32{0x65, 0xca3f, 0x2f304, 0x5185b, 0x66},
			wantTokens: []string{"[CLS]", "brown", "fox", "jumps", "[SEP]"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tk, err := tokenizers.FromBytesWithTruncation(embeddedBytes, uint32(tt.maxLen), tt.dir)
			require.NoError(t, err)
			defer tk.Close()

			ids, tokens := tk.Encode(tt.str, tt.addSpecial)
			assert.Equal(t, tt.wantIDs, ids, "wrong ids")
			assert.Equal(t, tt.wantTokens, tokens, "wrong tokens")
		})
	}
}

func TestEncodeWithPadding(t *testing.T) {
	tk, err := tokenizers.FromFile("./test/data/all-minilm-l6-v2.json")
	require.NoError(t, err)
	defer tk.Close()

	tests := []struct {
		name                  string
		str                   string
		addSpecial            bool
		wantIDs               []uint32
		wantTypeIDs           []uint32
		wantTokens            []string
		wantSpecialTokensMask []uint32
		wantAttentionMask     []uint32
		wantOffsets           []tokenizers.Offset
	}{
		{
			name:                  "sentence with padding",
			str:                   "this short sentence",
			addSpecial:            false,
			wantIDs:               []uint32{0x7e7, 0x99c, 0x186b, 0x0, 0x0, 0x0, 0x0, 0x0},
			wantTypeIDs:           []uint32{0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
			wantTokens:            []string{"this", "short", "sentence", "[PAD]", "[PAD]", "[PAD]", "[PAD]", "[PAD]"},
			wantSpecialTokensMask: []uint32{0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x1},
			wantAttentionMask:     []uint32{0x1, 0x1, 0x1, 0x0, 0x0, 0x0, 0x0, 0x0},
			wantOffsets:           []tokenizers.Offset{{0x0, 0x4}, {0x5, 0xa}, {0xb, 0x13}, {0x0, 0x0}, {0x0, 0x0}, {0x0, 0x0}, {0x0, 0x0}, {0x0, 0x0}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoding := tk.EncodeWithOptions(tt.str, tt.addSpecial, tokenizers.WithReturnAllAttributes())
			assert.Equal(t, tt.wantIDs, encoding.IDs, "wrong ids")
			assert.Equal(t, tt.wantTypeIDs, encoding.TypeIDs, "wrong type ids")
			assert.Equal(t, tt.wantTokens, encoding.Tokens, "wrong tokens")
			assert.Equal(t, tt.wantSpecialTokensMask, encoding.SpecialTokensMask, "wrong special tokens mask")
			assert.Equal(t, tt.wantAttentionMask, encoding.AttentionMask, "wrong attention mask")
			assert.Equal(t, tt.wantOffsets, encoding.Offsets, "wrong offsets")

			ids, tokens := tk.Encode(tt.str, tt.addSpecial)
			assert.Equal(t, tt.wantIDs, ids, "wrong ids")
			assert.Equal(t, tt.wantTokens, tokens, "wrong tokens")
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
		{
			name:        "no tokens",
			tokens:      []uint32{},
			skipSpecial: false,
			want:        "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tk.Decode(tt.tokens, tt.skipSpecial)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestDecodeInvalidString(t *testing.T) {
	tk, err := tokenizers.FromFile("test/data/cohere-tokenizer.json")
	require.NoError(t, err)
	defer tk.Close()

	str := tk.Decode([]uint32{196}, true)
	assert.Empty(t, str)
}

func TestVocabSize(t *testing.T) {
	tk, err := tokenizers.FromFile("./test/data/bert-base-uncased.json")
	require.NoError(t, err)
	defer tk.Close()
	assert.Equal(t, uint32(30522), tk.VocabSize())
}

func BenchmarkEncodeNTimes(b *testing.B) {
	hfTk, err := tokenizers.FromFile("./test/data/meta-llama-3-8b-instruct.json")
	require.NoError(b, err)
	defer hfTk.Close()

	// Source: https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py
	ttTk, err := tokenizers.FromTiktoken(
		"./test/data/meta-llama-3-8b-instruct/tiktoken.model",
		"./test/data/meta-llama-3-8b-instruct/tokenizer_config.json",
		`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
	)
	require.NoError(b, err)
	defer ttTk.Close()

	tests := []struct {
		name string
		tk   *tokenizers.Tokenizer
	}{
		{name: "huggingface", tk: hfTk},
		{name: "tiktoken", tk: ttTk},
	}

	for _, tt := range tests {
		expected := []uint32{0x10019, 0x9bff, 0x89ec, 0x39f, 0x117, 0x3eb5, 0x162f}
		b.Run(tt.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ids, _ := tt.tk.Encode("brown fox jumps over the lazy dog", false)
				assert.Equal(b, expected, ids)
			}
		})
	}
}

func BenchmarkEncodeNChars(b *testing.B) {
	hfTk, err := tokenizers.FromFile("./test/data/meta-llama-3-8b-instruct.json")
	require.NoError(b, err)
	defer hfTk.Close()

	// Source: https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py
	ttTk, err := tokenizers.FromTiktoken(
		"./test/data/meta-llama-3-8b-instruct/tiktoken.model",
		"./test/data/meta-llama-3-8b-instruct/tokenizer_config.json",
		`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
	)
	require.NoError(b, err)
	defer ttTk.Close()

	tests := []struct {
		name string
		tk   *tokenizers.Tokenizer
	}{
		{name: "huggingface", tk: hfTk},
		{name: "tiktoken", tk: ttTk},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			vocabSize := tt.tk.VocabSize()
			input := make([]rune, 0, b.N)
			for i := 0; i < b.N; i++ {
				input = append(input, rune(rand.Uint32()%vocabSize))
			}
			str := string(input)
			b.ResetTimer()
			ids, _ := tt.tk.Encode(str, false)
			assert.Greater(b, len(ids), 0, "input (len: %d): %v", len(input), str)
		})
	}
}

func BenchmarkDecodeNTimes(b *testing.B) {
	hfTk, err := tokenizers.FromFile("./test/data/meta-llama-3-8b-instruct.json")
	require.NoError(b, err)
	defer hfTk.Close()

	// Source: https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py
	ttTk, err := tokenizers.FromTiktoken(
		"./test/data/meta-llama-3-8b-instruct/tiktoken.model",
		"./test/data/meta-llama-3-8b-instruct/tokenizer_config.json",
		`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
	)
	require.NoError(b, err)
	defer ttTk.Close()

	tests := []struct {
		name string
		tk   *tokenizers.Tokenizer
	}{
		{name: "huggingface", tk: hfTk},
		{name: "tiktoken", tk: ttTk},
	}

	// Token IDs for "brown fox jumps over the lazy dog" in Llama tokenizer
	tokenIDs := []uint32{0x10019, 0x9bff, 0x89ec, 0x39f, 0x117, 0x3eb5, 0x162f}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				str := tt.tk.Decode(tokenIDs, true)
				assert.Equal(b, "brown fox jumps over the lazy dog", str)
			}
		})
	}
}

func BenchmarkDecodeNTokens(b *testing.B) {
	hfTk, err := tokenizers.FromFile("./test/data/meta-llama-3-8b-instruct.json")
	require.NoError(b, err)
	defer hfTk.Close()

	// Source: https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py
	ttTk, err := tokenizers.FromTiktoken(
		"./test/data/meta-llama-3-8b-instruct/tiktoken.model",
		"./test/data/meta-llama-3-8b-instruct/tokenizer_config.json",
		`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
	)
	require.NoError(b, err)
	defer ttTk.Close()

	tests := []struct {
		name string
		tk   *tokenizers.Tokenizer
	}{
		{name: "huggingface", tk: hfTk},
		{name: "tiktoken", tk: ttTk},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			vocabSize := tt.tk.VocabSize()
			// Generate some valid tokens by encoding text first
			sampleText := "The quick brown fox jumps over the lazy dog. This is a sample text for benchmarking."
			validTokens, _ := tt.tk.Encode(sampleText, false)
			if len(validTokens) == 0 {
				b.Fatal("Failed to generate valid tokens from sample text")
			}

			input := make([]uint32, 0, b.N)
			for i := 0; i < b.N; i++ {
				// Use valid tokens from our sample, cycling through them
				input = append(input, validTokens[i%len(validTokens)])
			}
			b.ResetTimer()
			text := tt.tk.Decode(input, true)
			// a token is one or more characters
			assert.GreaterOrEqual(b, len(text), b.N, "decoded text length: %d, expected at least: %d (vocab size: %d)", len(text), b.N, vocabSize)
		})
	}
}

func TestFromTiktoken(t *testing.T) {
	// Define the pattern for tiktoken tokenization
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

	tk, err := tokenizers.FromTiktoken(
		"./test/data/kimi-k2-instruct/tiktoken.model",
		"./test/data/kimi-k2-instruct/tokenizer_config.json",
		pattern,
	)
	require.NoError(t, err)
	defer tk.Close()

	{
		text := "Hello, world! 你好，世界！"
		ids, _ := tk.Encode(text, false)
		assert.Equal(t, []uint32{19180, 11, 2695, 0, 220, 33845, 378, 2243, 856}, ids)
		ids, _ = tk.Encode(text, true)
		assert.Equal(t, []uint32{19180, 11, 2695, 0, 220, 33845, 378, 2243, 856}, ids)

		decoded := tk.Decode(ids, false)
		assert.Equal(t, text, decoded)
		decoded = tk.Decode(ids, true)
		assert.Equal(t, text, decoded)
	}

	{
		text := "<|im_middle|>Hello, world! 你好，世界！<|im_end|>"
		ids, _ := tk.Encode(text, true)
		assert.Equal(t, []uint32{163601, 19180, 11, 2695, 0, 220, 33845, 378, 2243, 856, 163586}, ids)

		decoded := tk.Decode(ids, false)
		assert.Equal(t, text, decoded)
		decoded = tk.Decode(ids, true)
		assert.Equal(t, "Hello, world! 你好，世界！", decoded)
	}

	vocabSize := tk.VocabSize()
	assert.Equal(t, uint32(163840), vocabSize)
}

func TestTiktokenReplacementCharacter(t *testing.T) {
	// Test tiktoken tokenizer with replacement character (U+FFFD)
	// Source: https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py
	tk, err := tokenizers.FromTiktoken(
		"./test/data/meta-llama-3-8b-instruct/tiktoken.model",
		"./test/data/meta-llama-3-8b-instruct/tokenizer_config.json",
		`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
	)
	require.NoError(t, err)
	defer tk.Close()

	tests := []struct {
		name       string
		input      string
		shouldWork bool
	}{
		{
			name:       "normal text",
			input:      "Hello world",
			shouldWork: true,
		},
		{
			name:       "text with replacement character",
			input:      "Hello �world",
			shouldWork: true, // This currently fails but should work
		},
		{
			name:       "multiple replacement characters",
			input:      "Test � multiple � chars",
			shouldWork: true,
		},
		{
			name:       "only replacement character",
			input:      "�",
			shouldWork: true,
		},
		{
			name:       "UTF-8 replacement character bytes",
			input:      "\xEF\xBF\xBD", // UTF-8 encoding of U+FFFD
			shouldWork: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ids, _ := tk.Encode(tt.input, false)
			if tt.shouldWork {
				assert.NotEmpty(t, ids, "Expected non-empty token IDs for input: %q", tt.input)
				// assert.NotEmpty(t, tokens, "Expected non-empty tokens for input: %q", tt.input)

				// Test decoding
				if len(ids) > 0 {
					decoded := tk.Decode(ids, false)
					assert.Equal(t, tt.input, decoded, "Decoded text should match input")
				}
			} else {
				assert.Empty(t, ids, "Expected empty token IDs for input: %q", tt.input)
			}
		})
	}
}

func TestFromPretrained(t *testing.T) {
	tests := []struct {
		name          string
		modelID       string
		setupOpts     func(t *testing.T) ([]tokenizers.TokenizerConfigOption, string)
		wantErr       bool
		expectedToken bool
	}{
		{
			name:          "valid public model with cache dir",
			modelID:       "bert-base-uncased",
			expectedToken: true,
			setupOpts: func(t *testing.T) ([]tokenizers.TokenizerConfigOption, string) {
				tmpDir := t.TempDir()
				return []tokenizers.TokenizerConfigOption{
					tokenizers.WithCacheDir(tmpDir),
				}, tmpDir
			},
		},
		{
			name:          "valid public model without cache dir",
			modelID:       "bert-base-uncased",
			expectedToken: true,
			setupOpts: func(t *testing.T) ([]tokenizers.TokenizerConfigOption, string) {
				return nil, ""
			},
		},
		{
			name:    "private model with invalid auth token",
			modelID: "private-model",
			wantErr: true,
			setupOpts: func(t *testing.T) ([]tokenizers.TokenizerConfigOption, string) {
				tmpDir := t.TempDir()
				return []tokenizers.TokenizerConfigOption{
					tokenizers.WithCacheDir(tmpDir),
					tokenizers.WithAuthToken("invalid-token"),
				}, tmpDir
			},
		},
		{
			name:    "empty model ID",
			modelID: "",
			wantErr: true,
			setupOpts: func(t *testing.T) ([]tokenizers.TokenizerConfigOption, string) {
				return nil, ""
			},
		},
		{
			name:    "nonexistent model",
			modelID: "nonexistent/model",
			wantErr: true,
			setupOpts: func(t *testing.T) ([]tokenizers.TokenizerConfigOption, string) {
				tmpDir := t.TempDir()
				return []tokenizers.TokenizerConfigOption{
					tokenizers.WithCacheDir(tmpDir),
				}, tmpDir
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts, cacheDir := tt.setupOpts(t)
			tokenizer, err := tokenizers.FromPretrained(tt.modelID, opts...)

			if gotErr := err != nil; gotErr != tt.wantErr {
				t.Fatalf("expected error: %v, got error: %v", tt.wantErr, err)
			}
			if tt.wantErr {
				return
			}
			if cacheDir != "" {
				validateCache(t, cacheDir, tt.modelID)
			}
			if err := tokenizer.Close(); err != nil {
				t.Fatalf("error closing tokenizer: %v", err)
			}
		})
	}
}

func validateCache(t *testing.T, dir string, modelID string) {
	t.Helper()
	files := []string{"tokenizer.json", "vocab.txt"}
	for _, file := range files {
		path := filepath.Join(dir, modelID, file)
		if _, err := os.Stat(path); err != nil {
			t.Errorf("expected file %s to exist in cache for model %s", file, modelID)
		}
	}
}

func TestChatTemplateDeepSeek(t *testing.T) {
	// template := `{% for message in messages %}{% if message.role == 'user' %}{{ 'User: ' + message.content }}{% else %}{{ 'Assistant: ' + message.content }}{% endif %}{% endfor %}`
	template := "test/data/deepseek-ai/DeepSeek-R1/tokenizer_config.json"
	ct, err := tokenizers.NewChatTemplate(template)
	if err != nil {
		t.Fatalf("Failed to create chat template: %v", err)
	}

	messages_str := `[{"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "你好吗"
        },
		{
			"role": "assistant",
			"content": "你好，有什么我可以帮助你的吗？"
		},
		{
			"role": "user",
			"content": "你能做什么？"
		}
    ]`

	result, err := ct.ApplyChatTemplate(messages_str, "", "")
	if err != nil {
		t.Fatalf("Failed to apply chat template: %v", err)
	}
	fmt.Println(result)
}

func TestChatTemplateQwen3(t *testing.T) {
	// template := `{% for message in messages %}{% if message.role == 'user' %}{{ 'User: ' + message.content }}{% else %}{{ 'Assistant: ' + message.content }}{% endif %}{% endfor %}`
	template := "test/data/Qwen/Qwen3-235B-A22B/tokenizer_config.json"
	ct, err := tokenizers.NewChatTemplate(template)
	if err != nil {
		t.Fatalf("Failed to create chat template: %v", err)
	}

	messages_str := `[{"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "你好吗"
        },
		{
			"role": "assistant",
			"content": "你好，有什么我可以帮助你的吗？"
		},
		{
			"role": "user",
			"content": "你能做什么？"
		}
    ]`

	result, err := ct.ApplyChatTemplate(messages_str, "", "")
	if err != nil {
		t.Fatalf("Failed to apply chat template: %v", err)
	}
	fmt.Println(result)
}
