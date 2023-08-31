#include <stdbool.h>
#include <stdint.h>

struct Offset {
  uint32_t *start;
  uint32_t *end;
};

struct Buffer {
  uint32_t *ids;
  uint32_t *type_ids;
  uint32_t *special_tokens_mask;
  uint32_t *attention_mask;
  char *tokens;
  struct Offset **offsets;
  uint32_t len;
};

struct EncodeOptions {
  bool add_special_tokens;
  bool return_type_ids;
  bool return_special_tokens_mask;
  bool return_attention_mask;
  bool return_offsets;
  bool with_offsets_char_mode;
};

void *from_bytes(const uint8_t *config, uint32_t len);

void *from_bytes_with_truncation(const uint8_t *config, uint32_t len, uint32_t max_len, uint8_t direction);

void *from_file(const char *config);

struct Buffer encode(void *ptr, const char *message, const struct EncodeOptions *options);

struct Buffer* encode_batch(void *ptr, const char **messages, const struct EncodeOptions *options);

char *decode(void *ptr, const uint32_t *ids, uint32_t len, bool skip_special_tokens);

uint32_t vocab_size(void *ptr);

void free_tokenizer(void *ptr);

void free_buffer(struct Buffer buffer);

void free_batch_buffer(struct Buffer* buffers);

void free_string(char *string);
