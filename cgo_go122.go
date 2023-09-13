//go:build go1.22

package tokenizers

/*
// This is a performance optimization.
// The following noescape and nocallback directives are used to
// prevent the Go compiler from allocating function parameters on the heap.

#cgo noescape from_bytes
#cgo nocallback from_bytes
#cgo noescape from_bytes_with_truncation
#cgo nocallback from_bytes_with_truncation
#cgo noescape free_tokenizer
#cgo nocallback free_tokenizer
#cgo noescape encode
#cgo nocallback encode
#cgo noescape free_buffer
#cgo nocallback free_buffer
#cgo noescape encode_batch
#cgo nocallback encode_batch
#cgo noescape decode
#cgo nocallback decode
#cgo noescape free_string
#cgo nocallback free_string
#cgo noescape vocab_size
#cgo nocallback vocab_size

*/
import "C"
