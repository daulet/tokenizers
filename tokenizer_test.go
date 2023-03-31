package tokenizer_test

import (
	"testing"

	"github.com/daulet/tokenizer"

	"github.com/stretchr/testify/assert"
)

func TestEncode(t *testing.T) {
	tokens := tokenizer.Encode("brown fox jumps over the lazy dog")
	assert.Equal(t, []int64{2829, 4419, 14523, 2058, 1996, 13971, 3899}, tokens)
}
