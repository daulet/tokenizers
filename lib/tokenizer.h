#include <stdint.h>

void *from_file(const char *config);

void free_tokenizer(void *ptr);

uint32_t *encode(void *ptr, const char *message, uint32_t *len);

char *decode(void *ptr, const uint32_t *ids, uint32_t len);
