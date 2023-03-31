package tokenizer

// TODO packaging: how do we build the rust lib for distribution?

/*
#cgo LDFLAGS: ./lib/libtokenizer.a -ldl -lstdc++
#include <stdlib.h>
#include "./lib/tokenizer.h"
*/
import "C"

// NOTE: There should be NO space between the comments and the `import "C"` line.
import (
	"unsafe"
)

func Encode(str string) []uint32 {
	config := C.CString("./lib/tokenizer/data/bert-base-uncased.json")
	defer C.free(unsafe.Pointer(config))
	tokenizer := C.from_file(config)
	defer C.free_tokenizer(tokenizer)

	cStr := C.CString(str)
	defer C.free(unsafe.Pointer(cStr))
	var len C.uint
	res := C.encode(tokenizer, cStr, &len)
	defer C.free(unsafe.Pointer(res))
	slice := unsafe.Slice(res, len)

	tokenIDs := make([]uint32, len)
	for i, v := range slice {
		tokenIDs[i] = uint32(v)
	}
	return tokenIDs
}

func Decode(tokenIDs []uint32) string {
	config := C.CString("./lib/tokenizer/data/bert-base-uncased.json")
	defer C.free(unsafe.Pointer(config))
	tokenizer := C.from_file(config)
	defer C.free_tokenizer(tokenizer)

	len := C.uint(len(tokenIDs))
	res := C.decode(tokenizer, (*C.uint)(unsafe.Pointer(&tokenIDs[0])), len)
	defer C.free(unsafe.Pointer(res))
	return C.GoString(res)
}
