package tokenizers

// TODO packaging: how do we build the rust lib for distribution?

/*
#cgo LDFLAGS: ${SRCDIR}/libtokenizers.a -ldl -lm -lstdc++
#include <stdlib.h>
#include "tokenizers.h"
*/
import "C"

// NOTE: There should be NO space between the comments and the `import "C"` line.
import (
	"io"
	"unsafe"
)

type Offset struct {
	Start uint32
	End   uint32
}

type TokenizerResult struct {
	TokenIds []uint32
	Tokens   []string
	Offsets  []Offset
}

type Tokenizer struct {
	tokenizer unsafe.Pointer
}

type TruncationDirection int

const (
	TruncationDirectionLeft TruncationDirection = iota
	TruncationDirectionRight
)

var _ io.Closer = (*Tokenizer)(nil)

func FromBytes(data []byte) (*Tokenizer, error) {
	tokenizer := C.from_bytes((*C.uchar)(unsafe.Pointer(&data[0])), C.uint(len(data)))

	return &Tokenizer{tokenizer: tokenizer}, nil
}

func FromBytesWithTruncation(data []byte, maxLen uint32, dir TruncationDirection) (*Tokenizer, error) {
	tokenizer := C.from_bytes_with_truncation((*C.uchar)(unsafe.Pointer(&data[0])), C.uint(len(data)), C.uint(maxLen), C.uchar(dir))

	return &Tokenizer{tokenizer: tokenizer}, nil
}

func FromFile(path string) (*Tokenizer, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	tokenizer, err := C.from_file(cPath)
	if err != nil {
		return nil, err
	}

	return &Tokenizer{tokenizer: tokenizer}, nil
}

func (t *Tokenizer) Close() error {
	C.free_tokenizer(t.tokenizer)
	t.tokenizer = nil

	return nil
}

func (t *Tokenizer) Encode(str string, addSpecialTokens, returnOffsets, withCharMode bool) *TokenizerResult {
	cStr := C.CString(str)
	defer C.free(unsafe.Pointer(cStr))

	res := C.encode(t.tokenizer, cStr, C.bool(addSpecialTokens), C.bool(returnOffsets), C.bool(withCharMode))
	resLen := int(res.len)
	if resLen == 0 {
		return new(TokenizerResult)
	}
	defer C.free_buffer(res)

	encodeResult := &TokenizerResult{
		TokenIds: make([]uint32, resLen),
		Tokens:   make([]string, resLen),
	}
	// process token ids
	for i, v := range unsafe.Slice(res.ids, resLen) {
		encodeResult.TokenIds[i] = uint32(v)
	}
	// process tokens
	for i, s := range (*[1 << 30]*C.char)(unsafe.Pointer(res.tokens))[:resLen:resLen] {
		encodeResult.Tokens[i] = C.GoString(s)
	}
	// process offsets
	if returnOffsets {
		encodeResult.Offsets = make([]Offset, resLen)
		cOffsets := (*[1 << 30]C.struct_Offset)(unsafe.Pointer(res.offsets))
		for i := 0; i < resLen; i++ {
			encodeResult.Offsets[i] = Offset{
				Start: uint32(uintptr(unsafe.Pointer(cOffsets[i].start))), // Convert C.uintptr_t to uintptr
				End:   uint32(uintptr(unsafe.Pointer(cOffsets[i].end))),   // Convert C.uintptr_t to uintptr
			}
		}
	}

	return encodeResult
}

func (t *Tokenizer) EncodeBatch(strArr []string, addSpecialTokens, returnOffsets, withCharMode bool) []*TokenizerResult {
	batchLen := len(strArr)

	cStrings := make([]*C.char, len(strArr))
	for i, s := range strArr {
		cStrings[i] = C.CString(s)
		defer C.free(unsafe.Pointer(cStrings[i])) // Remember to free C strings
	}
	batchRes := C.encode_batch(
		t.tokenizer,
		(**C.char)(unsafe.Pointer(&cStrings[0])),
		C.bool(addSpecialTokens),
		C.bool(returnOffsets),
		C.bool(withCharMode))

	batchResult := make([]*TokenizerResult, batchLen)

	for i, encodeResult := range (*[1 << 30]C.struct_Buffer)(unsafe.Pointer(batchRes))[:batchLen:batchLen] {
		subResLen := int(encodeResult.len)
		subTokenizerResult := &TokenizerResult{
			TokenIds: make([]uint32, subResLen),
			Tokens:   make([]string, subResLen),
		}
		// process token ids
		for j, v := range unsafe.Slice(encodeResult.ids, subResLen) {
			subTokenizerResult.TokenIds[j] = uint32(v)
		}
		// process tokens
		for j, s := range (*[1 << 30]*C.char)(unsafe.Pointer(encodeResult.tokens))[:subResLen:subResLen] {
			subTokenizerResult.Tokens[j] = C.GoString(s)
		}

		batchResult[i] = subTokenizerResult
	}
	defer C.free_batch_buffer(batchRes)

	return batchResult
}

func (t *Tokenizer) Decode(tokenIDs []uint32, skipSpecialTokens bool) string {
	if len(tokenIDs) == 0 {
		return ""
	}
	res := C.decode(t.tokenizer, (*C.uint)(unsafe.Pointer(&tokenIDs[0])), C.uint(len(tokenIDs)), C.bool(skipSpecialTokens))
	defer C.free_string(res)

	return C.GoString(res)
}

func (t *Tokenizer) VocabSize() uint32 {
	return uint32(C.vocab_size(t.tokenizer))
}
