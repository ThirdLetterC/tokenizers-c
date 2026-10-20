/*
 * tokenizers.h — Embeddable C FFI Header for HuggingFace Tokenizers (Rust Core)
 *
 * Compatible with C99, C11, and C++11+.
 * Wraps the official Rust `tokenizers` crate (https://github.com/huggingface/tokenizers)
 * into a static library (`libtokenizers.a` / `tokenizers.lib`) with a single
 * embeddable C header file.
 *
 * License: Apache-2.0
 */

#ifndef HUGGINGFACE_TOKENIZERS_C_H
#define HUGGINGFACE_TOKENIZERS_C_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== *
 * Version & Export Macros
 * ========================================================================== */

#define TK_VERSION_MAJOR 0
#define TK_VERSION_MINOR 23
#define TK_VERSION_PATCH 2
#define TK_VERSION_STRING "0.23.2"

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(TK_BUILD_SHARED)
    #define TK_API __declspec(dllexport)
  #elif defined(TK_USE_SHARED)
    #define TK_API __declspec(dllimport)
  #else
    #define TK_API
  #endif
#elif defined(__GNUC__) && __GNUC__ >= 4
  #define TK_API __attribute__((visibility("default")))
#else
  #define TK_API
#endif

/* ========================================================================== *
 * Status & Error Codes
 * ========================================================================== */

typedef enum TkStatus {
    TK_OK = 0,
    TK_ERR_NULL_PTR = 1,
    TK_ERR_INVALID_UTF8 = 2,
    TK_ERR_IO = 3,
    TK_ERR_JSON = 4,
    TK_ERR_MODEL = 5,
    TK_ERR_TRUNCATION = 6,
    TK_ERR_PADDING = 7,
    TK_ERR_OUT_OF_RANGE = 8,
    TK_ERR_UNSUPPORTED = 9,
    TK_ERR_HTTP = 10,
    TK_ERR_PANIC = 11,
    TK_ERR_INTERNAL = 255
} TkStatus;

/* ========================================================================== *
 * Opaque Handles (Backed by Rust `tokenizers` Crate Structs)
 * ========================================================================== */

/**
 * Opaque handle to a Rust `tokenizers::Tokenizer` instance.
 *
 * Thread safety:
 * - `tokenizers::Tokenizer` is `Send + Sync` in Rust. Concurrent read-only calls
 *   (`tk_encode*`, `tk_decode*`, `tk_token_to_id`, `tk_id_to_token`, `tk_vocab_size`)
 *   on a shared `const TkTokenizer*` are thread-safe.
 * - Mutating configuration calls (`tk_enable_padding`, `tk_enable_truncation`,
 *   `tk_add_tokens`, etc.) require exclusive access.
 */
typedef struct TkTokenizer TkTokenizer;

/**
 * Opaque handle to a Rust `tokenizers::Encoding` result produced by `tk_encode*`.
 */
typedef struct TkEncoding TkEncoding;

/**
 * Opaque handle to a stateful token-by-token streaming decoder.
 */
typedef struct TkDecodeStream TkDecodeStream;

/* ========================================================================== *
 * Configuration Enums & Structs
 * ========================================================================== */

/** Direction for padding or truncation (`PaddingDirection` / `TruncationDirection`). */
typedef enum TkDirection {
    TK_DIRECTION_RIGHT = 0,
    TK_DIRECTION_LEFT = 1
} TkDirection;

/** Strategy for sequence/batch padding (`PaddingStrategy`). */
typedef enum TkPaddingStrategyTag {
    /** Pad each sequence to the longest sequence in the batch (`PaddingStrategy::BatchLongest`). */
    TK_PADDING_BATCH_LONGEST = 0,
    /** Pad each sequence to a fixed length (`PaddingStrategy::Fixed(fixed_length)`). */
    TK_PADDING_FIXED = 1
} TkPaddingStrategyTag;

/** Strategy for sequence truncation (`TruncationStrategy`). */
typedef enum TkTruncationStrategy {
    /** Truncate the longest sequence token-by-token (`TruncationStrategy::LongestFirst`). */
    TK_TRUNCATION_LONGEST_FIRST = 0,
    /** Only truncate the first sequence (`TruncationStrategy::OnlyFirst`). */
    TK_TRUNCATION_ONLY_FIRST = 1,
    /** Only truncate the second sequence in a pair (`TruncationStrategy::OnlySecond`). */
    TK_TRUNCATION_ONLY_SECOND = 2
} TkTruncationStrategy;

/** Offset unit when computing token spans (`encode` vs `encode_char_offsets` vs `encode_fast`). */
typedef enum TkOffsetType {
    /** Byte offsets into the UTF-8 input string (`Tokenizer::encode`). */
    TK_OFFSET_BYTE = 0,
    /** Unicode scalar (`char`) offsets into the input string (`Tokenizer::encode_char_offsets`). */
    TK_OFFSET_CHAR = 1,
    /** Skip offset computation for faster encoding (`Tokenizer::encode_fast`). */
    TK_OFFSET_NONE = 2
} TkOffsetType;

/**
 * Padding configuration mirroring `tokenizers::PaddingParams`.
 * Use `tk_padding_params_default()` to initialize with defaults.
 */
typedef struct TkPaddingParams {
    TkPaddingStrategyTag strategy;
    size_t fixed_length;
    TkDirection direction;
    /** If > 0, pad target length to a multiple of this value (`Some(pad_to_multiple_of)`). */
    size_t pad_to_multiple_of;
    uint32_t pad_id;
    uint32_t pad_type_id;
    /** Null-terminated pad token string (e.g. "[PAD]"). Defaults to "[PAD]" if NULL. */
    const char *pad_token;
} TkPaddingParams;

/**
 * Truncation configuration mirroring `tokenizers::TruncationParams`.
 * Use `tk_truncation_params_default(max_length)` to initialize with defaults.
 */
typedef struct TkTruncationParams {
    size_t max_length;
    TkTruncationStrategy strategy;
    size_t stride;
    TkDirection direction;
} TkTruncationParams;

/**
 * Added token specification mirroring `tokenizers::AddedToken`.
 * Use `tk_added_token_default(content, special)` to initialize with defaults.
 */
typedef struct TkAddedToken {
    /** Null-terminated token content string (required). */
    const char *content;
    /** Whether this token is a special token (e.g., [CLS], <|endoftext|>). */
    bool special;
    /** Whether this token must match a whole single word. */
    bool single_word;
    /** Whether to strip whitespace on the left of this token. */
    bool lstrip;
    /** Whether to strip whitespace on the right of this token. */
    bool rstrip;
    /** Whether this token is matched against normalized text. */
    bool normalized;
} TkAddedToken;

/**
 * Parameters for `tk_from_pretrained_ex` mirroring `tokenizers::FromPretrainedParameters`.
 */
typedef struct TkPretrainedParams {
    /** Git branch, tag, or commit hash (NULL defaults to "main"). */
    const char *revision;
    /** Optional HuggingFace Hub bearer token for gated/private models (may be NULL). */
    const char *auth_token;
    /** Optional custom User-Agent string (may be NULL). */
    const char *user_agent;
} TkPretrainedParams;

/**
 * Start and end offset span `[start, end)` in bytes or Unicode scalar values.
 */
typedef struct TkOffset {
    size_t start;
    size_t end;
} TkOffset;

/* ========================================================================== *
 * Error Handling & Library Metadata
 * ========================================================================== */

/**
 * Return the library version string ("0.23.2").
 */
TK_API const char *tk_version(void);

/**
 * Return the backend identifier ("rust-huggingface-tokenizers").
 */
TK_API const char *tk_backend_name(void);

/**
 * Return the thread-local error code from the last failed `tk_*` call on the
 * calling thread, or `TK_OK` if no error is recorded.
 */
TK_API TkStatus tk_last_error_code(void);

/**
 * Return a borrowed pointer to a null-terminated error description for the
 * last failed `tk_*` call on the calling thread.
 *
 * Returns `""` if no error is set. Valid until the next `tk_*` call on the same thread.
 */
TK_API const char *tk_last_error_message(void);

/**
 * Clear the thread-local error state on the calling thread.
 */
TK_API void tk_clear_error(void);

/* ========================================================================== *
 * Default Struct Initializers
 * ========================================================================== */

TK_API TkPaddingParams tk_padding_params_default(void);
TK_API TkTruncationParams tk_truncation_params_default(size_t max_length);
TK_API TkAddedToken tk_added_token_default(const char *content, bool special);
TK_API TkPretrainedParams tk_pretrained_params_default(void);

/* ========================================================================== *
 * Tokenizer Lifecycle & Serialization (`tokenizers::Tokenizer`)
 * ========================================================================== */

/**
 * Load a Tokenizer from a local `tokenizer.json` file (`Tokenizer::from_file`).
 */
TK_API TkTokenizer *tk_from_file(const char *path);

/**
 * Load a Tokenizer from an in-memory UTF-8 JSON buffer (`Tokenizer::from_bytes`).
 */
TK_API TkTokenizer *tk_from_bytes(const uint8_t *data, size_t len);

/**
 * Load a Tokenizer from a null-terminated JSON C string (`Tokenizer::from_bytes`).
 */
TK_API TkTokenizer *tk_from_json(const char *json);

/**
 * Load a pretrained Tokenizer from the HuggingFace Hub (`Tokenizer::from_pretrained`).
 * Requires the `http` Cargo feature (enabled by default).
 */
TK_API TkTokenizer *tk_from_pretrained(const char *identifier);

/**
 * Load a pretrained Tokenizer from the HuggingFace Hub with custom revision/token.
 */
TK_API TkTokenizer *tk_from_pretrained_ex(
    const char *identifier,
    const TkPretrainedParams *params
);

/**
 * Deep-copy an existing `TkTokenizer` (`Clone::clone`).
 */
TK_API TkTokenizer *tk_clone(const TkTokenizer *tokenizer);

/**
 * Save a `TkTokenizer` to a JSON file on disk (`Tokenizer::save`).
 */
TK_API TkStatus tk_save(const TkTokenizer *tokenizer, const char *path, bool pretty);

/**
 * Serialize a `TkTokenizer` to a newly allocated JSON C string (`Tokenizer::to_string`).
 * Free the returned string with `tk_string_free()`.
 */
TK_API char *tk_to_json(const TkTokenizer *tokenizer, bool pretty);

/**
 * Destroy a `TkTokenizer` and release all associated Rust memory.
 * Passing `NULL` is a safe no-op.
 */
TK_API void tk_free(TkTokenizer *tokenizer);

/* ========================================================================== *
 * Configuration: Padding, Truncation & Added Tokens
 * ========================================================================== */

TK_API TkStatus tk_enable_padding(TkTokenizer *tokenizer, const TkPaddingParams *params);
TK_API void tk_disable_padding(TkTokenizer *tokenizer);
TK_API bool tk_get_padding(const TkTokenizer *tokenizer, TkPaddingParams *out_params);

TK_API TkStatus tk_enable_truncation(TkTokenizer *tokenizer, const TkTruncationParams *params);
TK_API void tk_disable_truncation(TkTokenizer *tokenizer);
TK_API bool tk_get_truncation(const TkTokenizer *tokenizer, TkTruncationParams *out_params);

TK_API TkStatus tk_add_tokens(
    TkTokenizer *tokenizer,
    const TkAddedToken *tokens,
    size_t num_tokens,
    size_t *out_added
);

TK_API TkStatus tk_add_special_tokens(
    TkTokenizer *tokenizer,
    const TkAddedToken *tokens,
    size_t num_tokens,
    size_t *out_added
);

TK_API void tk_set_encode_special_tokens(TkTokenizer *tokenizer, bool value);
TK_API bool tk_get_encode_special_tokens(const TkTokenizer *tokenizer);

/* ========================================================================== *
 * Vocabulary Inspection
 * ========================================================================== */

TK_API size_t tk_vocab_size(const TkTokenizer *tokenizer, bool with_added_tokens);
TK_API bool tk_token_to_id(const TkTokenizer *tokenizer, const char *token, uint32_t *out_id);
TK_API char *tk_id_to_token(const TkTokenizer *tokenizer, uint32_t id);

/* ========================================================================== *
 * Encoding (Single, Pair, Pre-tokenized, and Rayon Parallel Batch)
 * ========================================================================== */

/**
 * Encode a single null-terminated UTF-8 sequence with byte offsets (`Tokenizer::encode`).
 */
TK_API TkEncoding *tk_encode(
    const TkTokenizer *tokenizer,
    const char *text,
    bool add_special_tokens
);

/**
 * Encode a single UTF-8 buffer with explicit length and offset mode
 * (`TK_OFFSET_BYTE` -> `encode`, `TK_OFFSET_CHAR` -> `encode_char_offsets`,
 * `TK_OFFSET_NONE` -> `encode_fast`).
 */
TK_API TkEncoding *tk_encode_ex(
    const TkTokenizer *tokenizer,
    const char *text,
    size_t text_len,
    bool add_special_tokens,
    TkOffsetType offset_type
);

/**
 * Encode a pair of null-terminated UTF-8 sequences (`Tokenizer::encode((a, b), ...)`).
 */
TK_API TkEncoding *tk_encode_pair(
    const TkTokenizer *tokenizer,
    const char *text_a,
    const char *text_b,
    bool add_special_tokens
);

/**
 * Encode a pair of UTF-8 buffers with explicit lengths and offset mode.
 */
TK_API TkEncoding *tk_encode_pair_ex(
    const TkTokenizer *tokenizer,
    const char *text_a,
    size_t text_a_len,
    const char *text_b,
    size_t text_b_len,
    bool add_special_tokens,
    TkOffsetType offset_type
);

/**
 * Encode an already pre-tokenized array of words (`Tokenizer::encode(&words[..], ...)`).
 */
TK_API TkEncoding *tk_encode_pretokenized(
    const TkTokenizer *tokenizer,
    const char *const *words,
    size_t word_count,
    bool add_special_tokens
);

/**
 * Encode a batch of sequences in parallel via Rayon (`Tokenizer::encode_batch`).
 */
TK_API TkStatus tk_encode_batch(
    const TkTokenizer *tokenizer,
    const char *const *texts,
    size_t count,
    bool add_special_tokens,
    TkEncoding **out_encodings
);

/**
 * Encode a batch of sequences in parallel without computing offsets (`Tokenizer::encode_batch_fast`).
 */
TK_API TkStatus tk_encode_batch_fast(
    const TkTokenizer *tokenizer,
    const char *const *texts,
    size_t count,
    bool add_special_tokens,
    TkEncoding **out_encodings
);

/**
 * Encode a batch of sequence pairs in parallel (`Tokenizer::encode_batch`).
 */
TK_API TkStatus tk_encode_batch_pair(
    const TkTokenizer *tokenizer,
    const char *const *texts_a,
    const char *const *texts_b,
    size_t count,
    bool add_special_tokens,
    TkEncoding **out_encodings
);

/**
 * Free every `TkEncoding*` in an array of length `count`.
 */
TK_API void tk_encoding_batch_free(TkEncoding **encodings, size_t count);

/* ========================================================================== *
 * Encoding Inspection, Manipulation & Alignment Queries (`tokenizers::Encoding`)
 * ========================================================================== */

TK_API void tk_encoding_free(TkEncoding *encoding);
TK_API size_t tk_encoding_len(const TkEncoding *encoding);
TK_API bool tk_encoding_is_empty(const TkEncoding *encoding);
TK_API size_t tk_encoding_n_sequences(const TkEncoding *encoding);

TK_API const uint32_t *tk_encoding_get_ids(const TkEncoding *encoding, size_t *out_len);
TK_API const uint32_t *tk_encoding_get_type_ids(const TkEncoding *encoding, size_t *out_len);
TK_API const uint32_t *tk_encoding_get_attention_mask(const TkEncoding *encoding, size_t *out_len);
TK_API const uint32_t *tk_encoding_get_special_tokens_mask(const TkEncoding *encoding, size_t *out_len);

TK_API const char *tk_encoding_get_token(const TkEncoding *encoding, size_t index);
TK_API bool tk_encoding_get_offset(const TkEncoding *encoding, size_t index, TkOffset *out_offset);
TK_API size_t tk_encoding_get_offsets(const TkEncoding *encoding, TkOffset *out_offsets, size_t max_len);

TK_API bool tk_encoding_get_word_id(const TkEncoding *encoding, size_t index, uint32_t *out_word_id);
TK_API size_t tk_encoding_get_word_ids(const TkEncoding *encoding, int64_t *out_word_ids, size_t max_len);

TK_API bool tk_encoding_get_sequence_id(const TkEncoding *encoding, size_t index, size_t *out_seq_id);
TK_API size_t tk_encoding_get_sequence_ids(const TkEncoding *encoding, int64_t *out_seq_ids, size_t max_len);

TK_API size_t tk_encoding_get_overflowing_count(const TkEncoding *encoding);
TK_API TkEncoding *tk_encoding_get_overflowing(const TkEncoding *encoding, size_t index);

TK_API bool tk_encoding_word_to_tokens(
    const TkEncoding *encoding,
    uint32_t word_id,
    size_t sequence_id,
    size_t *out_start_token,
    size_t *out_end_token
);

TK_API bool tk_encoding_word_to_chars(
    const TkEncoding *encoding,
    uint32_t word_id,
    size_t sequence_id,
    TkOffset *out_offset
);

TK_API bool tk_encoding_token_to_chars(
    const TkEncoding *encoding,
    size_t token_index,
    size_t *out_sequence_id,
    TkOffset *out_offset
);

TK_API bool tk_encoding_token_to_word(
    const TkEncoding *encoding,
    size_t token_index,
    size_t *out_sequence_id,
    uint32_t *out_word_id
);

TK_API bool tk_encoding_char_to_token(
    const TkEncoding *encoding,
    size_t pos,
    size_t sequence_id,
    size_t *out_token_index
);

TK_API bool tk_encoding_char_to_word(
    const TkEncoding *encoding,
    size_t pos,
    size_t sequence_id,
    uint32_t *out_word_id
);

/**
 * Truncate an `Encoding` in place (`Encoding::truncate`).
 * Returns `TK_ERR_TRUNCATION` if `stride >= max_len` or `max_len == 0`.
 */
TK_API TkStatus tk_encoding_truncate(
    TkEncoding *encoding,
    size_t max_len,
    size_t stride,
    TkDirection direction
);

/**
 * Pad an `Encoding` in place (`Encoding::pad`).
 */
TK_API TkStatus tk_encoding_pad(
    TkEncoding *encoding,
    size_t target_length,
    uint32_t pad_id,
    uint32_t pad_type_id,
    const char *pad_token,
    TkDirection direction
);

/**
 * Merge an array of `TkEncoding*` handles into a single new `TkEncoding*` (`Encoding::merge`).
 */
TK_API TkEncoding *tk_encoding_merge(
    const TkEncoding *const *encodings,
    size_t count,
    bool growing_offsets
);

/* ========================================================================== *
 * Decoding (Single, Batch, and Streaming)
 * ========================================================================== */

TK_API char *tk_decode(
    const TkTokenizer *tokenizer,
    const uint32_t *ids,
    size_t len,
    bool skip_special_tokens
);

TK_API TkStatus tk_decode_batch(
    const TkTokenizer *tokenizer,
    const uint32_t *const *batch_ids,
    const size_t *batch_lengths,
    size_t batch_size,
    bool skip_special_tokens,
    char **out_strings
);

TK_API void tk_string_free(char *str);
TK_API void tk_string_batch_free(char **strings, size_t count);

TK_API TkDecodeStream *tk_decode_stream_new(bool skip_special_tokens);

TK_API TkStatus tk_decode_stream_step(
    TkDecodeStream *stream,
    const TkTokenizer *tokenizer,
    uint32_t token_id,
    char **out_chunk
);

TK_API void tk_decode_stream_free(TkDecodeStream *stream);

#ifdef __cplusplus
}
#endif

#endif /* HUGGINGFACE_TOKENIZERS_C_H */
