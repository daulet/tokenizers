#include <stdbool.h>
#include <stdint.h>

struct Buffer {
  uint32_t *ids;
  uint32_t *type_ids;
  char *tokens;
  uint32_t *special_tokens_mask;
  uint32_t *attention_mask;
  uint32_t len;
};

void *from_bytes(const uint8_t *config, uint32_t len);

void *from_bytes_with_truncation(const uint8_t *config, uint32_t len, uint32_t max_len, uint8_t direction);

void *from_file(const char *config);

struct Buffer encode(void *ptr, const char *message, bool add_special_tokens);

char *decode(void *ptr, const uint32_t *ids, uint32_t len, bool skip_special_tokens);

uint32_t vocab_size(void *ptr);

void free_tokenizer(void *ptr);

void free_buffer(struct Buffer buffer);

void free_string(char *string);
