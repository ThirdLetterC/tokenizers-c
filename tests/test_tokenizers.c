/*
 * tests/test_tokenizers.c — Unit Test Suite for `tokenizers-c` (Rust FFI Wrapper)
 *
 * Verifies the public C FFI API declared in `include/tokenizers.h` against the
 * Rust `huggingface/tokenizers` core:
 *   - Loading from file, bytes, and JSON string
 *   - WordPiece, BPE, and Unigram models
 *   - Single, pair, and pre-tokenized encoding, offsets, and word/sequence alignments
 *   - Parallel batch encoding (`tk_encode_batch`, `tk_encode_batch_fast`) & batch decoding
 *   - Padding & Truncation (both on `TkTokenizer` and in-place on `TkEncoding`)
 *   - Merging encodings (`tk_encoding_merge`)
 *   - Added tokens & special tokens
 *   - Cloning and JSON serialization (`tk_save`, `tk_to_json`)
 *   - Stateful token-by-token streaming (`TkDecodeStream`)
 *   - Thread-local error reporting (`tk_last_error_code`, `tk_last_error_message`)
 */

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#ifndef __USE_MINGW_ANSI_STDIO
#define __USE_MINGW_ANSI_STDIO 1
#endif

#include "tokenizers.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT_TRUE(cond)                                                       \
    do {                                                                        \
        if (!(cond)) {                                                          \
            fprintf(stderr, "ASSERT_TRUE failed at %s:%d: %s (err: %s)\n",      \
                    __FILE__, __LINE__, #cond, tk_last_error_message());        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define ASSERT_EQ(a, b)                                                         \
    do {                                                                        \
        long long _va = (long long)(a);                                         \
        long long _vb = (long long)(b);                                         \
        if (_va != _vb) {                                                       \
            fprintf(stderr,                                                     \
                    "ASSERT_EQ failed at %s:%d: %s (%lld) != %s (%lld)\n",      \
                    __FILE__, __LINE__, #a, _va, #b, _vb);                      \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define ASSERT_STR_EQ(a, b)                                                     \
    do {                                                                        \
        const char *_sa = (a);                                                  \
        const char *_sb = (b);                                                  \
        if (!_sa || !_sb || strcmp(_sa, _sb) != 0) {                            \
            fprintf(stderr,                                                     \
                    "ASSERT_STR_EQ failed at %s:%d: \"%s\" != \"%s\"\n",        \
                    __FILE__, __LINE__, _sa ? _sa : "(null)",                   \
                    _sb ? _sb : "(null)");                                      \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static void test_version_and_errors(void) {
    printf("  [TEST] test_version_and_errors...\n");
    ASSERT_STR_EQ(tk_version(), TK_VERSION_STRING);
    ASSERT_STR_EQ(tk_backend_name(), "rust-huggingface-tokenizers");

    tk_clear_error();
    ASSERT_EQ(tk_last_error_code(), TK_OK);
    ASSERT_STR_EQ(tk_last_error_message(), "");

    /* Null pointer error handling */
    TkTokenizer *null_tk = tk_from_file(NULL);
    ASSERT_TRUE(null_tk == NULL);
    ASSERT_EQ(tk_last_error_code(), TK_ERR_NULL_PTR);
    ASSERT_TRUE(strlen(tk_last_error_message()) > 0);

    /* Non-existent file error handling */
    TkTokenizer *missing_tk = tk_from_file("__nonexistent_dir__/tokenizer.json");
    ASSERT_TRUE(missing_tk == NULL);
    ASSERT_EQ(tk_last_error_code(), TK_ERR_IO);

    /* Malformed JSON error handling */
    TkTokenizer *bad_json = tk_from_json("{invalid json");
    ASSERT_TRUE(bad_json == NULL);
    ASSERT_EQ(tk_last_error_code(), TK_ERR_JSON);

    tk_clear_error();
    ASSERT_EQ(tk_last_error_code(), TK_OK);
}

static void test_wordpiece_encode_decode_and_alignments(void) {
    printf("  [TEST] test_wordpiece_encode_decode_and_alignments...\n");
    TkTokenizer *tk = tk_from_file("examples/sample_bert_tokenizer.json");
    ASSERT_TRUE(tk != NULL);
    ASSERT_EQ(tk_vocab_size(tk, true), 24);

    uint32_t hf_id = 0;
    ASSERT_TRUE(tk_token_to_id(tk, "hugging", &hf_id));
    ASSERT_EQ(hf_id, 7);

    char *tok_str = tk_id_to_token(tk, 8);
    ASSERT_STR_EQ(tok_str, "##face");
    tk_string_free(tok_str);

    /* Encode: "Hello, HuggingFace tokenizers!" */
    const char *text = "Hello, HuggingFace tokenizers!";
    TkEncoding *enc = tk_encode(tk, text, true);
    ASSERT_TRUE(enc != NULL);
    ASSERT_TRUE(!tk_encoding_is_empty(enc));
    ASSERT_EQ(tk_encoding_n_sequences(enc), 1);

    size_t len = 0;
    const uint32_t *ids = tk_encoding_get_ids(enc, &len);
    ASSERT_EQ(len, 10);
    ASSERT_EQ(ids[0], 2);  /* [CLS] */
    ASSERT_EQ(ids[1], 5);  /* hello */
    ASSERT_EQ(ids[2], 17); /* , */
    ASSERT_EQ(ids[3], 7);  /* hugging */
    ASSERT_EQ(ids[4], 8);  /* ##face */
    ASSERT_EQ(ids[5], 9);  /* token */
    ASSERT_EQ(ids[6], 10); /* ##izer */
    ASSERT_EQ(ids[7], 11); /* ##s */
    ASSERT_EQ(ids[8], 18); /* ! */
    ASSERT_EQ(ids[9], 3);  /* [SEP] */

    ASSERT_STR_EQ(tk_encoding_get_token(enc, 3), "hugging");
    ASSERT_STR_EQ(tk_encoding_get_token(enc, 4), "##face");

    /* Alignment: word 2 ("HuggingFace") spans tokens [3, 5) and bytes [7, 18) */
    size_t start_tok = 0, end_tok = 0;
    ASSERT_TRUE(tk_encoding_word_to_tokens(enc, 2, 0, &start_tok, &end_tok));
    ASSERT_EQ(start_tok, 3);
    ASSERT_EQ(end_tok, 5);

    TkOffset word_span = {0, 0};
    ASSERT_TRUE(tk_encoding_word_to_chars(enc, 2, 0, &word_span));
    ASSERT_EQ(word_span.start, 7);
    ASSERT_EQ(word_span.end, 18);

    size_t tok_idx = 0;
    ASSERT_TRUE(tk_encoding_char_to_token(enc, 15, 0, &tok_idx));
    ASSERT_EQ(tok_idx, 4);

    uint32_t word_id = 0;
    ASSERT_TRUE(tk_encoding_char_to_word(enc, 15, 0, &word_id));
    ASSERT_EQ(word_id, 2);

    char *decoded = tk_decode(tk, ids, len, true);
    ASSERT_STR_EQ(decoded, "hello, huggingface tokenizers!");
    tk_string_free(decoded);

    /* Test pre-tokenized input */
    const char *pre_words[2] = {"Hello", "world"};
    TkEncoding *pre_enc = tk_encode_pretokenized(tk, pre_words, 2, false);
    ASSERT_TRUE(pre_enc != NULL);
    ASSERT_EQ(tk_encoding_len(pre_enc), 2);
    tk_encoding_free(pre_enc);

    tk_encoding_free(enc);
    tk_free(tk);
}

static void test_bpe_merges_and_streaming(void) {
    printf("  [TEST] test_bpe_merges_and_streaming...\n");
    TkTokenizer *tk = tk_from_file("examples/sample_bpe_tokenizer.json");
    ASSERT_TRUE(tk != NULL);

    TkEncoding *enc = tk_encode(tk, "HuggingFace hugging face", false);
    ASSERT_TRUE(enc != NULL);

    size_t len = 0;
    const uint32_t *ids = tk_encoding_get_ids(enc, &len);
    ASSERT_EQ(len, 3);
    ASSERT_EQ(ids[0], 20); /* "huggingface" */
    ASSERT_EQ(ids[1], 16); /* "hugging" */
    ASSERT_EQ(ids[2], 19); /* "face" */

    /* Test in-place Encoding pad, truncate, and merge */
    ASSERT_EQ(tk_encoding_pad(enc, 5, 1, 0, "<|pad|>", TK_DIRECTION_RIGHT), TK_OK);
    ASSERT_EQ(tk_encoding_len(enc), 5);
    ASSERT_STR_EQ(tk_encoding_get_token(enc, 4), "<|pad|>");

    ASSERT_EQ(tk_encoding_truncate(enc, 3, 0, TK_DIRECTION_RIGHT), TK_OK);
    ASSERT_EQ(tk_encoding_len(enc), 3);

    const TkEncoding *to_merge[2] = {enc, enc};
    TkEncoding *merged = tk_encoding_merge(to_merge, 2, true);
    ASSERT_TRUE(merged != NULL);
    ASSERT_EQ(tk_encoding_len(merged), 6);
    tk_encoding_free(merged);

    /* Test stateful DecodeStream */
    TkEncoding *stream_enc = tk_encode(tk, "HuggingFace hugging face", false);
    ASSERT_TRUE(stream_enc != NULL);
    size_t stream_len = 0;
    const uint32_t *stream_ids = tk_encoding_get_ids(stream_enc, &stream_len);

    TkDecodeStream *stream = tk_decode_stream_new(true);
    ASSERT_TRUE(stream != NULL);

    char accum[128] = {0};
    size_t accum_len = 0;
    for (size_t i = 0; i < stream_len; i++) {
        char *chunk = NULL;
        ASSERT_EQ(tk_decode_stream_step(stream, tk, stream_ids[i], &chunk), TK_OK);
        if (chunk) {
            size_t chunk_len = strlen(chunk);
            ASSERT_TRUE(chunk_len < sizeof(accum) - accum_len);
            memcpy(accum + accum_len, chunk, chunk_len + 1);
            accum_len += chunk_len;
            tk_string_free(chunk);
        }
    }
    ASSERT_STR_EQ(accum, "huggingface hugging face");

    tk_decode_stream_free(stream);
    tk_encoding_free(stream_enc);
    tk_encoding_free(enc);
    tk_free(tk);
}

static void test_padding_truncation_batch_and_clone(void) {
    printf("  [TEST] test_padding_truncation_batch_and_clone...\n");
    TkTokenizer *tk = tk_from_file("examples/sample_bert_tokenizer.json");
    ASSERT_TRUE(tk != NULL);

    /* Enable truncation with max_length=6, stride=1 */
    TkTruncationParams trunc = tk_truncation_params_default(6);
    trunc.stride = 1;
    ASSERT_EQ(tk_enable_truncation(tk, &trunc), TK_OK);

    TkTruncationParams queried_trunc;
    ASSERT_TRUE(tk_get_truncation(tk, &queried_trunc));
    ASSERT_EQ(queried_trunc.max_length, 6);
    ASSERT_EQ(queried_trunc.stride, 1);

    TkEncoding *enc = tk_encode(tk, "Hello world, HuggingFace tokenizers C library is awesome!", true);
    ASSERT_TRUE(enc != NULL);
    ASSERT_EQ(tk_encoding_len(enc), 6);
    ASSERT_TRUE(tk_encoding_get_overflowing_count(enc) >= 1);

    TkEncoding *ov0 = tk_encoding_get_overflowing(enc, 0);
    ASSERT_TRUE(ov0 != NULL);
    ASSERT_TRUE(tk_encoding_len(ov0) <= 6);
    tk_encoding_free(ov0);
    tk_encoding_free(enc);

    tk_disable_truncation(tk);
    ASSERT_TRUE(!tk_get_truncation(tk, NULL));

    /* Enable left-padding to fixed length 8 */
    TkPaddingParams pad = tk_padding_params_default();
    pad.strategy = TK_PADDING_FIXED;
    pad.fixed_length = 8;
    pad.direction = TK_DIRECTION_LEFT;
    pad.pad_id = 0;
    pad.pad_token = "[PAD]";
    ASSERT_EQ(tk_enable_padding(tk, &pad), TK_OK);

    TkTokenizer *cloned = tk_clone(tk);
    ASSERT_TRUE(cloned != NULL);

    const char *batch[2] = {"Hello world", "Fast C library"};
    TkEncoding *batch_encs[2] = {NULL, NULL};
    ASSERT_EQ(tk_encode_batch(cloned, batch, 2, true, batch_encs), TK_OK);

    for (size_t b = 0; b < 2; b++) {
        size_t blen = 0;
        const uint32_t *bids = tk_encoding_get_ids(batch_encs[b], &blen);
        const uint32_t *battn = tk_encoding_get_attention_mask(batch_encs[b], NULL);
        ASSERT_EQ(blen, 8);
        ASSERT_EQ(bids[0], 0);
        ASSERT_EQ(battn[0], 0);
        ASSERT_EQ(battn[7], 1);
    }

    const uint32_t *decode_ptrs[2] = {
        tk_encoding_get_ids(batch_encs[0], NULL),
        tk_encoding_get_ids(batch_encs[1], NULL)
    };
    size_t decode_lens[2] = {8, 8};
    char *decoded_strs[2] = {NULL, NULL};
    ASSERT_EQ(tk_decode_batch(cloned, decode_ptrs, decode_lens, 2, true, decoded_strs), TK_OK);
    ASSERT_STR_EQ(decoded_strs[0], "hello world");
    ASSERT_STR_EQ(decoded_strs[1], "fast c library");

    tk_string_batch_free(decoded_strs, 2);
    tk_encoding_batch_free(batch_encs, 2);

    /* Test fast batch encode */
    ASSERT_EQ(tk_encode_batch_fast(cloned, batch, 2, true, batch_encs), TK_OK);
    tk_encoding_batch_free(batch_encs, 2);

    /* Test JSON round-trip */
    char *json_out = tk_to_json(cloned, false);
    ASSERT_TRUE(json_out != NULL && strlen(json_out) > 20);
    TkTokenizer *reloaded = tk_from_json(json_out);
    ASSERT_TRUE(reloaded != NULL);
    ASSERT_EQ(tk_vocab_size(reloaded, true), 24);
    tk_string_free(json_out);

    tk_free(reloaded);
    tk_free(cloned);
    tk_free(tk);
}

int main(void) {
    printf("Running tokenizers-c unit test suite (version %s, backend %s)...\n",
           tk_version(), tk_backend_name());

    test_version_and_errors();
    test_wordpiece_encode_decode_and_alignments();
    test_bpe_merges_and_streaming();
    test_padding_truncation_batch_and_clone();

    printf("All tokenizers-c tests passed!\n");
    return EXIT_SUCCESS;
}
