# `tokenizers-c` — Embeddable C FFI Wrapper for HuggingFace Tokenizers (Rust Core)

`tokenizers-c` is a thin, panic-safe Rust FFI `staticlib` wrapper around the official [HuggingFace `tokenizers` Rust crate (`v0.23`)](https://github.com/huggingface/tokenizers) that compiles into an embeddable C header (`dist/include/tokenizers.h`) and static library archive (`dist/lib/libtokenizers.a` / `dist/lib/tokenizers.lib`) for **Windows**, **Linux**, and **macOS**.

## Build Outputs

Building the project runs `cargo build --release` and populates `dist/`:

- **`dist/include/tokenizers.h`** — Single embeddable C header (compatible with C99, C11, and C++11+, zero external header dependencies).
- **`dist/lib/libtokenizers.a`** — Static library archive for GCC / Clang / MinGW-w64 / Unix linkers.
- **`dist/lib/tokenizers.lib`** — Static library archive copy for Windows MSVC (`cl.exe` / `link.exe`).
- **`dist/lib/pkgconfig/tokenizers.pc`** — `pkg-config` metadata with platform linker flags.

## Prerequisites

- **Rust & Cargo** (`rustup` / `cargo` >= 1.75) to compile the upstream `tokenizers` crate.
- **C Compiler** (`cl.exe` on Windows MSVC, `gcc` on MinGW-w64 / Linux, or `clang` on Windows / macOS / Linux) to compile and link your C application, examples, or unit tests.
- Note on Windows Regex Engine: `Cargo.toml` enables `fancy-regex` and `http` by default (`default-features = false` on `tokenizers`), which compiles 100% in Rust on both `x86_64-pc-windows-msvc` and `x86_64-pc-windows-gnu` without `onig_sys` C-header conflicts. If you also need Oniguruma or `esaxx_fast`, pass `--features onig,esaxx_fast` to Cargo.

---

## Building on Windows

### Option A: Windows Command Prompt (`build.bat`)

```bat
:: Build dist\include\tokenizers.h + dist\lib\libtokenizers.a & dist\lib\tokenizers.lib via Cargo
build.bat

:: Optional: cross-compile specifically for MinGW-w64 GNU ABI or MSVC ABI
build.bat gnu
build.bat msvc

:: Compile and run unit tests and examples (auto-selects MSVC or MinGW ABI)
build.bat test
build.bat examples

:: Optionally choose the matching ABI explicitly
build.bat test gnu
build.bat examples msvc
build\basic_usage.exe
build\batch_and_pair.exe
```

### Option B: Windows PowerShell (`build.ps1`)

```powershell
# Build static library & header via Cargo, then compile and run C tests and examples
.\build.ps1 -Test -Examples

# Or specify a specific Rust target triple (e.g. for MinGW gcc linking):
.\build.ps1 -Target x86_64-pc-windows-gnu -Test
```

### Option C: Visual Studio `nmake` (`NMakefile`)

From an **x64 Native Tools Command Prompt for VS**:

```bat
nmake /f NMakefile
nmake /f NMakefile test
nmake /f NMakefile examples
```

### Option D: CMake (Visual Studio, Ninja, or MinGW)

```powershell
cmake -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

### Option E: MSYS2 / MinGW-w64 (`mingw32-make` or `make`)

```bash
mingw32-make
mingw32-make test
mingw32-make examples
```

---

## Building on Linux & macOS

```bash
make
make test
make examples
./build/basic_usage
./build/batch_and_pair
```

---

## Embedding in Your C Codebase

Copy `dist/include/tokenizers.h` and `dist/lib/libtokenizers.a` (or `dist/lib/tokenizers.lib` on MSVC) into your C project:

```c
#include "tokenizers.h"
#include <stdio.h>

int main(void) {
    /* Load from local tokenizer.json or HuggingFace Hub (tk_from_pretrained("bert-base-uncased")) */
    TkTokenizer *tk = tk_from_file("tokenizer.json");
    if (!tk) {
        fprintf(stderr, "Error: %s\n", tk_last_error_message());
        return 1;
    }

    TkEncoding *enc = tk_encode(tk, "Hello, HuggingFace tokenizers!", true);
    size_t len = 0;
    const uint32_t *ids = tk_encoding_get_ids(enc, &len);

    for (size_t i = 0; i < len; i++) {
        printf("%u (%s) ", ids[i], tk_encoding_get_token(enc, i));
    }
    printf("\n");

    char *decoded = tk_decode(tk, ids, len, true);
    printf("Decoded: %s\n", decoded);

    tk_string_free(decoded);
    tk_encoding_free(enc);
    tk_free(tk);
    return 0;
}
```

### Linking Flags by Platform

Because `libtokenizers.a` / `tokenizers.lib` is a Rust `staticlib`, link the standard OS system libraries required by the Rust standard library and HTTP/TLS stack:

- **Windows (MSVC `cl.exe`, built with `x86_64-pc-windows-msvc`)**:
  ```bat
  cl /O2 /Idist\include my_app.c dist\lib\tokenizers.lib ws2_32.lib userenv.lib advapi32.lib bcrypt.lib ntdll.lib crypt32.lib secur32.lib
  ```
- **Windows (MinGW-w64 `gcc`, built with `x86_64-pc-windows-gnu`)**:
  ```bash
  gcc -O2 -Idist/include my_app.c dist/lib/libtokenizers.a -lws2_32 -luserenv -ladvapi32 -lbcrypt -lntdll -lcrypt32 -lsecur32 -lstdc++ -lm -lpthread -o my_app.exe
  ```
- **Linux**:
  ```bash
  cc -O2 -Idist/include my_app.c dist/lib/libtokenizers.a -Wl,--no-as-needed -ldl -lpthread -lm -lstdc++ -lgcc_s -lc -o my_app
  ```
- **macOS**:
  ```bash
  cc -O2 -Idist/include my_app.c dist/lib/libtokenizers.a -framework Security -framework CoreFoundation -liconv -lSystem -lc++ -lm -lpthread -o my_app
  ```
- **Via `pkg-config` (Linux / macOS / MSYS2)**:
  ```bash
  PKG_CONFIG_PATH=dist/lib/pkgconfig cc -O2 my_app.c $(pkg-config --cflags --libs tokenizers) -o my_app
  ```

## Project Layout

```text
tokenizers-c/
├── Cargo.toml                         # Rust staticlib/cdylib crate manifest (wraps `tokenizers = "0.23"`)
├── CMakeLists.txt                     # CMake build & FetchContent integration
├── Makefile                           # Make build system (Linux, macOS, MSYS2/MinGW)
├── NMakefile                          # Visual Studio nmake build file (Windows)
├── build.bat                          # Windows Command Prompt / VS Developer Prompt build script
├── build.ps1                          # Windows PowerShell build script
├── build.rs                           # Cargo build script
├── cbindgen.toml                      # cbindgen configuration for C header generation
├── include/
│   └── tokenizers.h                   # Embeddable C99/C11/C++ header file
├── dist/
│   └── include/
│       └── tokenizers.h               # Pre-copied embeddable header file
├── src/
│   └── lib.rs                         # Rust FFI bindings around huggingface/tokenizers
├── examples/
│   ├── basic_usage.c                  # Single-sequence encode/decode/streaming demo
│   ├── batch_and_pair.c               # Batch padding, truncation, and pair encoding demo
│   ├── sample_bert_tokenizer.json     # Sample WordPiece tokenizer.json
│   └── sample_bpe_tokenizer.json      # Sample BPE tokenizer.json
└── tests/
    └── test_tokenizers.c              # C unit test suite
```
