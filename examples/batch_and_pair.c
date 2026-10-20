/*
 * examples/batch_and_pair.c — Batch encoding, padding, truncation, and sequence pairs
 *
 * Build & run (Linux / macOS / MinGW):
 *   make examples
 *   ./build/batch_and_pair
 *
 * Build & run (Windows CMD / PowerShell):
 *   build.bat examples
 *   build\batch_and_pair.exe
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

int main(void) {
    TkTokenizer *tk = tk_from_file("examples/sample_bert_tokenizer.json");
    if (!tk) {
        fprintf(stderr, "Failed to load tokenizer: %s\n", tk_last_error_message());
        return EXIT_FAILURE;
    }

    /* 1. Add custom domain tokens dynamically */
    TkAddedToken custom_tokens[2];
    custom_tokens[0] = tk_added_token_default("<|thought|>", true);
    custom_tokens[1] = tk_added_token_default("embeddable", false);

    size_t added_count = 0;
    tk_add_tokens(tk, custom_tokens, 2, &added_count);
    printf("Added %zu new tokens (new vocab size = %zu)\n",
           added_count, tk_vocab_size(tk, true));

    /* 2. Configure batch padding to multiple of 4 */
    TkPaddingParams pad = tk_padding_params_default();
    pad.strategy = TK_PADDING_BATCH_LONGEST;
    pad.pad_to_multiple_of = 4;
    pad.pad_id = 0;
    pad.pad_token = "[PAD]";
    tk_enable_padding(tk, &pad);

    /* 3. Encode a batch of sequences */
    const char *batch_inputs[3] = {
        "Hello world!",
        "<|thought|> HuggingFace tokenizers is an embeddable C library.",
        "Fast tokenizers"
    };
    TkEncoding *batch_out[3] = {NULL, NULL, NULL};

    if (tk_encode_batch(tk, batch_inputs, 3, true, batch_out) != TK_OK) {
        fprintf(stderr, "Batch encode failed: %s\n", tk_last_error_message());
        tk_free(tk);
        return EXIT_FAILURE;
    }

    printf("\nBatch Encoded (%zu sequences, padded to multiple of 4):\n", (size_t)3);
    for (size_t b = 0; b < 3; b++) {
        size_t len = 0;
        const uint32_t *ids = tk_encoding_get_ids(batch_out[b], &len);
        const uint32_t *attn = tk_encoding_get_attention_mask(batch_out[b], NULL);
        printf("  Batch[%zu] (len=%zu): IDs=[", b, len);
        for (size_t i = 0; i < len; i++) {
            printf("%s%" PRIu32, (i > 0) ? ", " : "", ids[i]);
        }
        printf("]  Attn=[");
        for (size_t i = 0; i < len; i++) {
            printf("%s%" PRIu32, (i > 0) ? ", " : "", attn[i]);
        }
        printf("]\n");
    }
    tk_encoding_batch_free(batch_out, 3);

    /* 4. Encode a sequence pair (Question + Context) with truncation */
    tk_disable_padding(tk);
    TkTruncationParams trunc = tk_truncation_params_default(8);
    /* Pair special tokens reduce the effective truncation length to 2 here. */
    trunc.stride = 1;
    if (tk_enable_truncation(tk, &trunc) != TK_OK) {
        fprintf(stderr, "Failed to enable truncation: %s\n", tk_last_error_message());
        tk_free(tk);
        return EXIT_FAILURE;
    }

    TkEncoding *pair_enc = tk_encode_pair(
        tk,
        "Hello world?",
        "HuggingFace tokenizers C library is fast!",
        true
    );
    if (pair_enc) {
        printf("\nPair Encoding (max_length=8, stride=1): len=%zu, overflowing_windows=%zu\n",
               tk_encoding_len(pair_enc),
               tk_encoding_get_overflowing_count(pair_enc));
        for (size_t i = 0; i < tk_encoding_len(pair_enc); i++) {
            printf("  [%zu] token=%-10s id=%-4" PRIu32 " type_id=%" PRIu32 "\n",
                   i,
                   tk_encoding_get_token(pair_enc, i),
                   tk_encoding_get_ids(pair_enc, NULL)[i],
                   tk_encoding_get_type_ids(pair_enc, NULL)[i]);
        }
        tk_encoding_free(pair_enc);
    }

    tk_free(tk);
    return EXIT_SUCCESS;
}
