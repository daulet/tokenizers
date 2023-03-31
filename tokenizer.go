package tokenizer

// NOTE: There should be NO space between the comments and the `import "C"` line.

/*
#cgo LDFLAGS: ./lib/libtokenizer.a -ldl -lstdc++
#include "./lib/tokenizer.h"
*/
import "C"
import (
	"unsafe"
)

func Encode(str string) []int64 {
	var len C.uint
	var x *C.uint = C.encode(C.CString(str), &len)
	slice := unsafe.Slice(x, len)

	tokenIDs := make([]int64, len)
	for i, v := range slice {
		// TODO what's the native type in rust library, maybe uint32 is wrong
		tokenIDs[i] = int64(v)
	}
	return tokenIDs
	// TODO free memory
}
