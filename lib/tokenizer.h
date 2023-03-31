// NOTE: You could use https://michael-f-bryan.github.io/rust-ffi-guide/cbindgen.html to generate
// this header automatically from your Rust code.  But for now, we'll just write it by hand.
#include <stdint.h>

uint32_t *encode(char *message, uint32_t *len);
