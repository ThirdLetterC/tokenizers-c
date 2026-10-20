/*
 * examples/basic_usage.c — Minimal example of embedding `tokenizers.h` + `libtokenizers.a`
 *
 * Build & run (Linux / macOS / MinGW):
 *   make examples
 *   ./build/basic_usage
 *
 * Build & run (Windows CMD / PowerShell):
 *   build.bat examples
 *   build\basic_usage.exe
 */

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#ifndef __USE_MINGW_ANSI_STDIO
#define __USE_MINGW_ANSI_STDIO 1
#endif

#include "tokenizers.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    const char *tokenizer_path =
        (argc > 1) ? argv[1] : "examples/sample_bert_tokenizer.json";
    const char *input_text =
        (argc > 2) ? argv[2] : "Hello, HuggingFace tokenizers C library is awesome!";

    printf("tokenizers-c version: %s (backend: %s)\n",
           tk_version(), tk_backend_name());

    /* 1. Load Tokenizer from file */
    TkTokenizer *tk = tk_from_file(tokenizer_path);
    if (!tk) {
        fprintf(stderr, "Error loading tokenizer (%d): %s\n",
                (int)tk_last_error_code(), tk_last_error_message());
        return EXIT_FAILURE;
    }

    printf("Loaded '%s' (vocab_size = %zu)\n\n",
           tokenizer_path, tk_vocab_size(tk, true));

    /* 2. Encode single sequence with special tokens ([CLS] ... [SEP]) */
    TkEncoding *enc = tk_encode(tk, input_text, true);
    if (!enc) {
        fprintf(stderr, "Encode failed: %s\n", tk_last_error_message());
        tk_free(tk);
        return EXIT_FAILURE;
    }

    size_t len = 0;
    const uint32_t *ids = tk_encoding_get_ids(enc, &len);
    const uint32_t *type_ids = tk_encoding_get_type_ids(enc, NULL);
    const uint32_t *att_mask = tk_encoding_get_attention_mask(enc, NULL);
    const uint32_t *spec_mask = tk_encoding_get_special_tokens_mask(enc, NULL);

    printf("Input: \"%s\"\n", input_text);
    printf("Encoded %zu tokens:\n", len);
    printf("  %-4s %-6s %-14s %-6s %-6s %-8s %-8s %-12s\n",
           "Idx", "ID", "Token", "Type", "Attn", "Special", "WordID", "ByteOffsets");

    for (size_t i = 0; i < len; i++) {
        const char *tok = tk_encoding_get_token(enc, i);
        TkOffset off = {0, 0};
        tk_encoding_get_offset(enc, i, &off);
        uint32_t word_id = 0;
        bool has_word = tk_encoding_get_word_id(enc, i, &word_id);

        char word_buf[16];
        if (has_word) {
            snprintf(word_buf, sizeof(word_buf), "%" PRIu32, word_id);
        } else {
            snprintf(word_buf, sizeof(word_buf), "None");
        }

        printf("  %-4zu %-6" PRIu32 " %-14s %-6" PRIu32 " %-6" PRIu32 " %-8" PRIu32 " %-8s [%zu, %zu)\n",
               i, ids[i], tok ? tok : "", type_ids[i], att_mask[i],
               spec_mask[i], word_buf, off.start, off.end);
    }

    /* 3. Decode back to text (skipping special tokens) */
    char *decoded_clean = tk_decode(tk, ids, len, true);
    char *decoded_full = tk_decode(tk, ids, len, false);
    printf("\nDecoded (skip_special_tokens=true):  \"%s\"\n",
           decoded_clean ? decoded_clean : "");
    printf("Decoded (skip_special_tokens=false): \"%s\"\n",
           decoded_full ? decoded_full : "");

    tk_string_free(decoded_clean);
    tk_string_free(decoded_full);

    /* 4. Demonstrate incremental token-by-token DecodeStream */
    printf("\nStreaming decode token-by-token: ");
    TkDecodeStream *stream = tk_decode_stream_new(true);
    for (size_t i = 0; i < len; i++) {
        char *chunk = NULL;
        if (tk_decode_stream_step(stream, tk, ids[i], &chunk) == TK_OK && chunk) {
            printf("[%s]", chunk);
            tk_string_free(chunk);
        }
    }
    printf("\n");

    tk_decode_stream_free(stream);
    tk_encoding_free(enc);
    tk_free(tk);
    return EXIT_SUCCESS;
}
