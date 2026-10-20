/*
 * streaming_decode.c — Decode encoded token IDs one at a time.
 *
 * Run from the project root:
 *   build\streaming_decode.exe [tokenizer.json] [text]
 */

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#ifndef _WIN32
#define _POSIX_C_SOURCE 199309L
#endif

#include "tokenizers.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#define STREAM_DELAY_MS 70

static void sleep_between_tokens(void) {
#ifdef _WIN32
    Sleep(STREAM_DELAY_MS);
#else
    const struct timespec delay = {0, STREAM_DELAY_MS * 1000000L};
    nanosleep(&delay, NULL);
#endif
}

int main(int argc, char **argv) {
    const char *tokenizer_path =
        (argc > 1) ? argv[1] : "examples/sample_bert_tokenizer.json";
    const char *text = (argc > 2) ? argv[2] :
        "Hello world! HuggingFace tokenizers is fast! Hello world! "
        "HuggingFace tokenizers is awesome! Fast tokenizers is fast! "
        "Hello world! HuggingFace tokenizers is fast! Hello world!";

    TkTokenizer *tokenizer = tk_from_file(tokenizer_path);
    if (!tokenizer) {
        fprintf(stderr, "Failed to load tokenizer '%s': %s\n",
                tokenizer_path, tk_last_error_message());
        return EXIT_FAILURE;
    }

    TkEncoding *encoding = tk_encode(tokenizer, text, true);
    if (!encoding) {
        fprintf(stderr, "Failed to encode input: %s\n", tk_last_error_message());
        tk_free(tokenizer);
        return EXIT_FAILURE;
    }

    size_t count = 0;
    const uint32_t *ids = tk_encoding_get_ids(encoding, &count);
    if (count != 0 && !ids) {
        fprintf(stderr, "Failed to read encoded token IDs: %s\n",
                tk_last_error_message());
        tk_encoding_free(encoding);
        tk_free(tokenizer);
        return EXIT_FAILURE;
    }

    TkDecodeStream *stream = tk_decode_stream_new(true);
    if (!stream) {
        fprintf(stderr, "Failed to create decode stream: %s\n",
                tk_last_error_message());
        tk_encoding_free(encoding);
        tk_free(tokenizer);
        return EXIT_FAILURE;
    }

    printf("You: %s\n\nAssistant: ", text);
    fflush(stdout);
    for (size_t i = 0; i < count; ++i) {
        char *chunk = NULL;
        TkStatus status = tk_decode_stream_step(stream, tokenizer, ids[i], &chunk);
        if (status != TK_OK) {
            fprintf(stderr, "\nDecode failed at token %zu: %s\n",
                    i, tk_last_error_message());
            tk_decode_stream_free(stream);
            tk_encoding_free(encoding);
            tk_free(tokenizer);
            return EXIT_FAILURE;
        }

        if (chunk && chunk[0]) {
            fputs(chunk, stdout);
            fflush(stdout);
        }
        if (chunk) tk_string_free(chunk);
        if (i + 1 < count) sleep_between_tokens();
    }
    putchar('\n');

    tk_decode_stream_free(stream);
    tk_encoding_free(encoding);
    tk_free(tokenizer);
    return EXIT_SUCCESS;
}
