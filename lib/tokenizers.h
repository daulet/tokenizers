#include <stdbool.h>
#include <stdint.h>

void *from_file(const char *config);

void free_tokenizer(void *ptr);

uint32_t *encode(void *ptr, const char *message, uint32_t *len, bool add_special_tokens);

char *decode(void *ptr, const uint32_t *ids, uint32_t len, bool skip_special_tokens);

uint32_t vocab_size(void *ptr);
