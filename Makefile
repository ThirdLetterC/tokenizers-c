# Makefile for tokenizers-c (C FFI Static Library Wrapper for HuggingFace Tokenizers Rust Crate)
#
# Works on Linux, macOS, and Windows (MSYS2 / MinGW-w64 / Git Bash / mingw32-make).
# For native Windows CMD / PowerShell / Visual Studio nmake, see `build.bat`, `build.ps1`, and `NMakefile`.
#
# Targets:
#   make            Build dist/include/tokenizers.h + dist/lib/libtokenizers.a via Cargo
#   make header     Regenerate include/tokenizers.h via cbindgen (optional)
#   make examples   Compile C example binaries linked against dist/lib/libtokenizers.a
#   make test       Compile and run the C unit test suite against dist/lib/libtokenizers.a
#   make install    Install header, libtokenizers.a, and pkg-config file to $(PREFIX)
#   make clean      Remove build and dist artifacts

PREFIX      ?= /usr/local
CC          ?= cc
CARGO       ?= cargo
CBINDGEN    ?= cbindgen
CARGO_FLAGS ?= --release

DIST_DIR    := dist
INC_DIR     := $(DIST_DIR)/include
LIB_DIR     := $(DIST_DIR)/lib
PKG_DIR     := $(LIB_DIR)/pkgconfig
BUILD_DIR   := build

STATIC_LIB  := $(LIB_DIR)/libtokenizers.a
MSVC_LIB    := $(LIB_DIR)/tokenizers.lib
HEADER_OUT  := $(INC_DIR)/tokenizers.h
PC_OUT      := $(PKG_DIR)/tokenizers.pc

ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    EXE_EXT     := .exe
    PIC_FLAG    :=
    SYS_LIBS    := -lws2_32 -luserenv -ladvapi32 -lbcrypt -lntdll -lcrypt32 -lsecur32 -lstdc++ -lm -lpthread
else
    DETECTED_OS := $(shell uname -s 2>/dev/null || echo Unknown)
    EXE_EXT     :=
    PIC_FLAG    := -fPIC
    ifeq ($(DETECTED_OS),Darwin)
        SYS_LIBS := -framework Security -framework CoreFoundation -liconv -lSystem -lc++ -lm -lpthread
    else
        SYS_LIBS := -Wl,--no-as-needed -ldl -lpthread -lm -lstdc++ -lgcc_s -lc
    endif
endif

CFLAGS      ?= -O3 -Wall -Wextra -pedantic -std=c11 $(PIC_FLAG)

.PHONY: all rust header examples test install clean

all: rust

$(INC_DIR) $(LIB_DIR) $(PKG_DIR) $(BUILD_DIR):
	@mkdir -p $@

# ---------------------------------------------------------------------------
# Build Official Rust `huggingface/tokenizers` staticlib via Cargo
# ---------------------------------------------------------------------------
rust: | $(INC_DIR) $(LIB_DIR) $(PKG_DIR)
	$(CARGO) build $(CARGO_FLAGS)
	cp include/tokenizers.h $(HEADER_OUT)
	@if [ -f target/release/libtokenizers.a ]; then \
		cp target/release/libtokenizers.a $(STATIC_LIB); \
		cp target/release/libtokenizers.a $(MSVC_LIB); \
	elif [ -f target/release/tokenizers.lib ]; then \
		cp target/release/tokenizers.lib $(STATIC_LIB); \
		cp target/release/tokenizers.lib $(MSVC_LIB); \
	else \
		echo "Error: Cargo static library not found in target/release/"; exit 1; \
	fi
	@$(MAKE) --no-print-directory $(PC_OUT)
	@echo "==> Built static archive: $(STATIC_LIB) (and $(MSVC_LIB))"
	@echo "==> Built C header:       $(HEADER_OUT)"

$(PC_OUT): | $(PKG_DIR)
	@printf "prefix=%s\nexec_prefix=\$${prefix}\nlibdir=\$${exec_prefix}/lib\nincludedir=\$${prefix}/include\n\nName: tokenizers\nDescription: Embeddable C FFI library for HuggingFace Tokenizers (Rust Core)\nVersion: 0.23.2\nLibs: -L\$${libdir} -ltokenizers %s\nCflags: -I\$${includedir}\n" \
		"$(PREFIX)" "$(SYS_LIBS)" > $(PC_OUT)

header:
	$(CBINDGEN) --config cbindgen.toml --crate tokenizers-c --output include/tokenizers.h

# ---------------------------------------------------------------------------
# Examples & Tests
# ---------------------------------------------------------------------------
examples: all | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INC_DIR) examples/basic_usage.c $(STATIC_LIB) $(SYS_LIBS) -o $(BUILD_DIR)/basic_usage$(EXE_EXT)
	$(CC) $(CFLAGS) -I$(INC_DIR) examples/batch_and_pair.c $(STATIC_LIB) $(SYS_LIBS) -o $(BUILD_DIR)/batch_and_pair$(EXE_EXT)
	@echo "==> Compiled examples: $(BUILD_DIR)/basic_usage$(EXE_EXT), $(BUILD_DIR)/batch_and_pair$(EXE_EXT)"

test: all | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Werror -I$(INC_DIR) tests/test_tokenizers.c $(STATIC_LIB) $(SYS_LIBS) -o $(BUILD_DIR)/test_tokenizers$(EXE_EXT)
	./$(BUILD_DIR)/test_tokenizers$(EXE_EXT)

install: all
	install -d $(DESTDIR)$(PREFIX)/include
	install -d $(DESTDIR)$(PREFIX)/lib/pkgconfig
	install -m 0644 $(HEADER_OUT) $(DESTDIR)$(PREFIX)/include/tokenizers.h
	install -m 0644 $(STATIC_LIB) $(DESTDIR)$(PREFIX)/lib/libtokenizers.a
	install -m 0644 $(PC_OUT) $(DESTDIR)$(PREFIX)/lib/pkgconfig/tokenizers.pc

clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR)
	$(CARGO) clean 2>/dev/null || true
