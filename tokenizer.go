package tokenizers

// TODO packaging: how do we build the rust lib for distribution?

/*
#cgo LDFLAGS: -ltokenizers -ldl -lm -lstdc++
#include <stdlib.h>
#include "tokenizers.h"

// Link-time version check: this will fail to link if the library version doesn't match
// Using a global variable that references the function ensures the linker must resolve it
extern void tokenizers_version_1_23_0(void);
void (*tokenizers_version_check)(void) = &tokenizers_version_1_23_0;
*/
import "C"

// NOTE: There should be NO space between the comments and the `import "C"` line.
import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
	"unsafe"
)

const baseURL = "https://huggingface.co"
const defaultHTTPTimeout = 30 * time.Second

var hfHTTPClient = &http.Client{Timeout: defaultHTTPTimeout}
var ErrTokenizerClosed = errors.New("tokenizer is nil or closed")

// List of necessary tokenizer files and their mandatory status.
// True means mandatory, false means optional.
var tokenizerFiles = map[string]bool{
	"tokenizer.json":          true,
	"vocab.txt":               false,
	"merges.txt":              false,
	"special_tokens_map.json": false,
	"added_tokens.json":       false,
}

type Tokenizer struct {
	tokenizer unsafe.Pointer
}

type tokenizerOpts struct {
	encodeSpecialTokens C.bool
}

type TokenizerOption func(to *tokenizerOpts)

func WithEncodeSpecialTokens() TokenizerOption {
	return func(to *tokenizerOpts) {
		to.encodeSpecialTokens = C.bool(true)
	}
}

type TruncationDirection int

const (
	TruncationDirectionLeft TruncationDirection = iota
	TruncationDirectionRight
)

var _ io.Closer = (*Tokenizer)(nil)

func FromBytes(data []byte, opts ...TokenizerOption) (*Tokenizer, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("tokenizer data cannot be empty")
	}

	allOpts := &tokenizerOpts{
		// by default, we do not encode special tokens
		encodeSpecialTokens: C.bool(false),
	}
	for _, opt := range opts {
		opt(allOpts)
	}

	var errPtr *C.char
	tokenizer := C.tokenizers_from_bytes((*C.uchar)(unsafe.Pointer(&data[0])), C.uint(len(data)), (*C.struct_tokenizers_options)(unsafe.Pointer(allOpts)), &errPtr)
	if tokenizer == nil {
		if errPtr != nil {
			errStr := C.GoString(errPtr)
			C.tokenizers_free_string(errPtr)
			return nil, fmt.Errorf("%s", errStr)
		}
		return nil, fmt.Errorf("failed to create tokenizer from bytes")
	}

	return &Tokenizer{tokenizer: tokenizer}, nil
}

func FromBytesWithTruncation(data []byte, maxLen uint32, dir TruncationDirection) (*Tokenizer, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("tokenizer data cannot be empty")
	}

	var errPtr *C.char
	tokenizer := C.tokenizers_from_bytes_with_truncation((*C.uchar)(unsafe.Pointer(&data[0])), C.uint(len(data)), C.size_t(maxLen), C.uchar(dir), &errPtr)
	if tokenizer == nil {
		if errPtr != nil {
			errStr := C.GoString(errPtr)
			C.tokenizers_free_string(errPtr)
			return nil, fmt.Errorf("%s", errStr)
		}
		return nil, fmt.Errorf("failed to create tokenizer with truncation")
	}

	return &Tokenizer{tokenizer: tokenizer}, nil
}

func FromFile(path string) (*Tokenizer, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var errPtr *C.char
	tokenizer := C.tokenizers_from_file(cPath, &errPtr)
	if tokenizer == nil {
		if errPtr != nil {
			errStr := C.GoString(errPtr)
			C.tokenizers_free_string(errPtr)
			return nil, fmt.Errorf("%s", errStr)
		}
		return nil, fmt.Errorf("failed to create tokenizer from file")
	}

	return &Tokenizer{tokenizer: tokenizer}, nil
}

// FromTiktoken creates a tokenizer from tiktoken model and config files
func FromTiktoken(modelPath, configPath, pattern string) (*Tokenizer, error) {
	cModelPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelPath))

	cConfigPath := C.CString(configPath)
	defer C.free(unsafe.Pointer(cConfigPath))

	cPattern := C.CString(pattern)
	defer C.free(unsafe.Pointer(cPattern))

	var errPtr *C.char
	tokenizer := C.tokenizers_from_tiktoken(cModelPath, cConfigPath, cPattern, &errPtr)

	if tokenizer == nil {
		if errPtr != nil {
			errStr := C.GoString(errPtr)
			C.tokenizers_free_string(errPtr)
			return nil, fmt.Errorf("%s", errStr)
		}
		return nil, fmt.Errorf("failed to create tiktoken tokenizer")
	}

	return &Tokenizer{tokenizer: tokenizer}, nil
}

type tokenizerConfig struct {
	cacheDir  *string
	authToken *string
}

type TokenizerConfigOption func(cfg *tokenizerConfig)

func WithCacheDir(path string) TokenizerConfigOption {
	return func(cfg *tokenizerConfig) {
		cfg.cacheDir = &path
	}
}

func WithAuthToken(token string) TokenizerConfigOption {
	return func(cfg *tokenizerConfig) {
		cfg.authToken = &token
	}
}

func normalizeModelID(modelID string) (string, error) {
	modelID = strings.TrimSpace(modelID)
	if modelID == "" {
		return "", fmt.Errorf("modelID cannot be empty")
	}
	if strings.ContainsRune(modelID, '\x00') {
		return "", fmt.Errorf("modelID contains an invalid null byte")
	}
	return modelID, nil
}

func safeJoinCacheDir(baseDir, modelID string) (string, error) {
	cleanModelID := filepath.Clean(modelID)
	if filepath.IsAbs(cleanModelID) {
		return "", fmt.Errorf("modelID must be relative")
	}
	if cleanModelID == ".." || strings.HasPrefix(cleanModelID, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("modelID must not escape cache directory")
	}

	baseClean := filepath.Clean(baseDir)
	target := filepath.Join(baseClean, cleanModelID)
	rel, err := filepath.Rel(baseClean, target)
	if err != nil {
		return "", fmt.Errorf("failed to validate cache path: %w", err)
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("modelID must not escape cache directory")
	}

	return target, nil
}

// FromPretrained downloads necessary files and initializes the tokenizer.
// Parameters:
//   - modelID: The Hugging Face model identifier (e.g., "bert-base-uncased").
//   - WithCacheDir(path): Optional. If provided, files will be downloaded to this folder.
//   - WithAuthToken(token): Optional. If provided, it will be used to authenticate requests.
func FromPretrained(modelID string, opts ...TokenizerConfigOption) (*Tokenizer, error) {
	cfg := &tokenizerConfig{}
	for _, opt := range opts {
		opt(cfg)
	}
	normalizedModelID, err := normalizeModelID(modelID)
	if err != nil {
		return nil, err
	}

	// Construct the model URL
	modelURL := fmt.Sprintf("%s/%s/resolve/main", baseURL, normalizedModelID)

	// Determine the download directory
	var downloadDir string
	isTempDir := false
	if cfg.cacheDir != nil {
		downloadDir, err = safeJoinCacheDir(*cfg.cacheDir, normalizedModelID)
		if err != nil {
			return nil, err
		}
		// Create the destination directory if it doesn't exist
		err = os.MkdirAll(downloadDir, os.ModePerm)
		if err != nil {
			return nil, fmt.Errorf("failed to create destination directory %s: %w", downloadDir, err)
		}
	} else {
		// Create a temporary directory
		tmpDir, err := os.MkdirTemp("", "huggingface-tokenizer-*")
		if err != nil {
			return nil, fmt.Errorf("error creating temporary directory: %w", err)
		}
		downloadDir = tmpDir
		isTempDir = true
	}
	if isTempDir {
		defer func() {
			_ = os.RemoveAll(downloadDir)
		}()
	}

	var wg sync.WaitGroup
	errCh := make(chan error)

	// Download each tokenizer file concurrently
	for filename, isMandatory := range tokenizerFiles {
		wg.Add(1)
		go func(fn string, mandatory bool) {
			defer wg.Done()
			fileURL := fmt.Sprintf("%s/%s", modelURL, fn)
			destPath := filepath.Join(downloadDir, fn)
			err := downloadFile(fileURL, destPath, cfg.authToken)
			if err != nil && mandatory {
				// If the file is mandatory, report an error
				errCh <- fmt.Errorf("failed to download mandatory file %s: %w", fn, err)
			}
		}(filename, isMandatory)
	}

	go func() {
		wg.Wait()
		close(errCh)
	}()

	var errs []error
	for err := range errCh {
		errs = append(errs, err)
	}

	if len(errs) > 0 {
		return nil, errs[0]
	}

	return FromFile(filepath.Join(downloadDir, "tokenizer.json"))
}

// downloadFile downloads a file from the given URL and saves it to the specified destination.
// If authToken is provided (non-nil), it will be used for authorization.
// Returns an error if the download fails.
func downloadFile(url, destination string, authToken *string) error {
	// Check if the file already exists
	if _, err := os.Stat(destination); err == nil {
		return nil
	}

	// Create a new HTTP request
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request for %s: %w", url, err)
	}

	// If authToken is provided, set the Authorization header
	if authToken != nil {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", *authToken))
	}

	resp, err := hfHTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to download from %s: %w", url, err)
	}
	defer resp.Body.Close()

	// Check for successful response
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download from %s: status code %d", url, resp.StatusCode)
	}

	// Create the destination file
	out, err := os.Create(destination)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", destination, err)
	}
	defer out.Close()

	// Write the response body to the file
	_, err = io.Copy(out, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to write to file %s: %w", destination, err)
	}

	return nil
}

func (t *Tokenizer) Close() error {
	C.tokenizers_free_tokenizer(t.tokenizer)
	t.tokenizer = nil
	return nil
}

type Offset [2]uint

type Encoding struct {
	IDs               []uint32
	TypeIDs           []uint32
	SpecialTokensMask []uint32
	AttentionMask     []uint32
	Tokens            []string
	Offsets           []Offset
}

type encodeOpts struct {
	AddSpecialTokens C.bool

	ReturnTypeIDs           C.bool
	ReturnTokens            C.bool
	ReturnSpecialTokensMask C.bool
	ReturnAttentionMask     C.bool
	ReturnOffsets           C.bool
}

type EncodeOption func(eo *encodeOpts)

func uintVecToSlice(arrPtr *C.uint, len int) []uint32 {
	arr := unsafe.Slice(arrPtr, len)
	slice := make([]uint32, len)
	for i, v := range arr {
		slice[i] = uint32(v)
	}
	return slice
}

func offsetVecToSlice(arrPtr *C.size_t, tokenLength int) []Offset {
	arr := unsafe.Slice(arrPtr, tokenLength*2)
	slice := make([]Offset, tokenLength)
	counter := 0
	for i := 0; i < tokenLength; i++ {
		offset := Offset{uint(arr[counter]), uint(arr[counter+1])}
		slice[i] = offset
		counter = counter + 2
	}
	return slice
}

func (t *Tokenizer) EncodeErr(str string, addSpecialTokens bool) ([]uint32, []string, error) {
	if t == nil || t.tokenizer == nil {
		return nil, nil, ErrTokenizerClosed
	}
	cStr := C.CString(str)
	defer C.free(unsafe.Pointer(cStr))
	options := encodeOpts{
		AddSpecialTokens: C.bool(addSpecialTokens),
		ReturnTokens:     C.bool(true),
	}
	res := C.tokenizers_encode(t.tokenizer, cStr, (*C.struct_tokenizers_encode_options)(unsafe.Pointer(&options)))
	len := int(res.len)
	if len == 0 {
		if str == "" {
			return nil, nil, nil
		}
		if res.ids == nil {
			return nil, nil, fmt.Errorf("failed to encode input")
		}
		return nil, nil, nil
	}
	defer C.tokenizers_free_buffer(res)

	ids := uintVecToSlice(res.ids, len)

	var tokens []string
	if res.tokens != nil {
		tokens = make([]string, len)
		for i, s := range (*[1 << 30]*C.char)(unsafe.Pointer(res.tokens))[:len:len] {
			tokens[i] = C.GoString(s)
		}
	}
	return ids, tokens, nil
}

func (t *Tokenizer) Encode(str string, addSpecialTokens bool) ([]uint32, []string) {
	if t == nil || t.tokenizer == nil {
		return nil, nil
	}
	cStr := C.CString(str)
	defer C.free(unsafe.Pointer(cStr))
	options := encodeOpts{
		AddSpecialTokens: C.bool(addSpecialTokens),
		ReturnTokens:     C.bool(true),
	}
	res := C.tokenizers_encode(t.tokenizer, cStr, (*C.struct_tokenizers_encode_options)(unsafe.Pointer(&options)))
	len := int(res.len)
	if len == 0 {
		return nil, nil
	}
	defer C.tokenizers_free_buffer(res)

	ids := uintVecToSlice(res.ids, len)

	var tokens []string
	if res.tokens != nil {
		tokens = make([]string, len)
		for i, s := range (*[1 << 30]*C.char)(unsafe.Pointer(res.tokens))[:len:len] {
			tokens[i] = C.GoString(s)
		}
	}

	return ids, tokens
}

func WithReturnAllAttributes() EncodeOption {
	return func(eo *encodeOpts) {
		eo.ReturnTypeIDs = C.bool(true)
		eo.ReturnSpecialTokensMask = C.bool(true)
		eo.ReturnAttentionMask = C.bool(true)
		eo.ReturnTokens = C.bool(true)
		eo.ReturnOffsets = C.bool(true)
	}
}

func WithReturnTypeIDs() EncodeOption {
	return func(eo *encodeOpts) {
		eo.ReturnTypeIDs = C.bool(true)
	}
}

func WithReturnSpecialTokensMask() EncodeOption {
	return func(eo *encodeOpts) {
		eo.ReturnSpecialTokensMask = C.bool(true)
	}
}

func WithReturnTokens() EncodeOption {
	return func(eo *encodeOpts) {
		eo.ReturnTokens = C.bool(true)
	}
}

func WithReturnAttentionMask() EncodeOption {
	return func(eo *encodeOpts) {
		eo.ReturnAttentionMask = C.bool(true)
	}
}

func WithReturnOffsets() EncodeOption {
	return func(eo *encodeOpts) {
		eo.ReturnOffsets = C.bool(true)
	}
}

func (t *Tokenizer) EncodeWithOptionsErr(str string, addSpecialTokens bool, opts ...EncodeOption) (Encoding, error) {
	if t == nil || t.tokenizer == nil {
		return Encoding{}, ErrTokenizerClosed
	}
	cStr := C.CString(str)
	defer C.free(unsafe.Pointer(cStr))

	encOptions := encodeOpts{
		AddSpecialTokens: C.bool(addSpecialTokens),
	}
	for _, opt := range opts {
		opt(&encOptions)
	}

	res := C.tokenizers_encode(t.tokenizer, cStr, (*C.struct_tokenizers_encode_options)(unsafe.Pointer(&encOptions)))
	len := int(res.len)
	if len == 0 {
		if str == "" {
			return Encoding{}, nil
		}
		if res.ids == nil {
			return Encoding{}, fmt.Errorf("failed to encode input")
		}
		return Encoding{}, nil
	}
	defer C.tokenizers_free_buffer(res)

	encoding := Encoding{}
	encoding.IDs = uintVecToSlice(res.ids, len)

	if encOptions.ReturnTypeIDs && res.type_ids != nil {
		encoding.TypeIDs = uintVecToSlice(res.type_ids, len)
	}

	if encOptions.ReturnTokens && res.tokens != nil {
		tokens := make([]string, len)
		for i, s := range (*[1 << 30]*C.char)(unsafe.Pointer(res.tokens))[:len:len] {
			tokens[i] = C.GoString(s)
		}
		encoding.Tokens = tokens
	}

	if encOptions.ReturnSpecialTokensMask && res.special_tokens_mask != nil {
		encoding.SpecialTokensMask = uintVecToSlice(res.special_tokens_mask, len)
	}

	if encOptions.ReturnAttentionMask && res.attention_mask != nil {
		encoding.AttentionMask = uintVecToSlice(res.attention_mask, len)
	}

	if encOptions.ReturnOffsets && res.offsets != nil {
		encoding.Offsets = offsetVecToSlice(res.offsets, len)
	}

	return encoding, nil
}

func (t *Tokenizer) EncodeWithOptions(str string, addSpecialTokens bool, opts ...EncodeOption) Encoding {
	if t == nil || t.tokenizer == nil {
		return Encoding{}
	}
	cStr := C.CString(str)
	defer C.free(unsafe.Pointer(cStr))

	encOptions := encodeOpts{
		AddSpecialTokens: C.bool(addSpecialTokens),
	}
	for _, opt := range opts {
		opt(&encOptions)
	}

	res := C.tokenizers_encode(t.tokenizer, cStr, (*C.struct_tokenizers_encode_options)(unsafe.Pointer(&encOptions)))
	len := int(res.len)
	if len == 0 {
		return Encoding{}
	}
	defer C.tokenizers_free_buffer(res)

	encoding := Encoding{}
	encoding.IDs = uintVecToSlice(res.ids, len)

	if encOptions.ReturnTypeIDs && res.type_ids != nil {
		encoding.TypeIDs = uintVecToSlice(res.type_ids, len)
	}

	if encOptions.ReturnTokens && res.tokens != nil {
		tokens := make([]string, len)
		for i, s := range (*[1 << 30]*C.char)(unsafe.Pointer(res.tokens))[:len:len] {
			tokens[i] = C.GoString(s)
		}
		encoding.Tokens = tokens
	}

	if encOptions.ReturnSpecialTokensMask && res.special_tokens_mask != nil {
		encoding.SpecialTokensMask = uintVecToSlice(res.special_tokens_mask, len)
	}

	if encOptions.ReturnAttentionMask && res.attention_mask != nil {
		encoding.AttentionMask = uintVecToSlice(res.attention_mask, len)
	}

	if encOptions.ReturnOffsets && res.offsets != nil {
		encoding.Offsets = offsetVecToSlice(res.offsets, len)
	}

	return encoding
}

func (t *Tokenizer) DecodeErr(tokenIDs []uint32, skipSpecialTokens bool) (string, error) {
	if t == nil || t.tokenizer == nil {
		return "", ErrTokenizerClosed
	}
	if len(tokenIDs) == 0 {
		return "", nil
	}
	len := C.uint(len(tokenIDs))
	res := C.tokenizers_decode(t.tokenizer, (*C.uint)(unsafe.Pointer(&tokenIDs[0])), len, C.bool(skipSpecialTokens))
	if res == nil {
		return "", fmt.Errorf("failed to decode token IDs")
	}
	defer C.tokenizers_free_string(res)
	return C.GoString(res), nil
}

func (t *Tokenizer) Decode(tokenIDs []uint32, skipSpecialTokens bool) string {
	if t == nil || t.tokenizer == nil {
		return ""
	}
	if len(tokenIDs) == 0 {
		return ""
	}
	len := C.uint(len(tokenIDs))
	res := C.tokenizers_decode(t.tokenizer, (*C.uint)(unsafe.Pointer(&tokenIDs[0])), len, C.bool(skipSpecialTokens))
	if res == nil {
		return ""
	}
	defer C.tokenizers_free_string(res)
	return C.GoString(res)
}

func (t *Tokenizer) VocabSize() uint32 {
	return uint32(C.tokenizers_vocab_size(t.tokenizer))
}
