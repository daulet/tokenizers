package tokenizer_test

import (
	"testing"

	"github.com/daulet/tokenizer"

	"github.com/stretchr/testify/assert"
)

// TODO test for leaks

func TestEncode(t *testing.T) {
	tokens := tokenizer.Encode("brown fox jumps over the lazy dog")
	assert.Equal(t, []uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899}, tokens)
}

func TestDecode(t *testing.T) {
	str := tokenizer.Decode([]uint32{2829, 4419, 14523, 2058, 1996, 13971, 3899})
	assert.Equal(t, "brown fox jumps over the lazy dog", str)
}
