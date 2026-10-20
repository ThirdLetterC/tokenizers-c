cmake_minimum_required(VERSION 3.16)
project(tokenizers_c VERSION 0.23.2 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(TOKENIZERS_BUILD_TESTS "Build tokenizers-c unit tests" ON)
option(TOKENIZERS_BUILD_EXAMPLES "Build tokenizers-c example programs" ON)
set(TOKENIZERS_RUST_TARGET "" CACHE STRING "Optional Rust target triple (e.g. x86_64-pc-windows-gnu or x86_64-pc-windows-msvc)")

find_program(CARGO_EXECUTABLE cargo REQUIRED)

set(TK_DIST_INCLUDE_DIR "${CMAKE_BINARY_DIR}/dist/include")
set(TK_DIST_LIB_DIR "${CMAKE_BINARY_DIR}/dist/lib")
file(MAKE_DIRECTORY "${TK_DIST_INCLUDE_DIR}" "${TK_DIST_LIB_DIR}")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/tokenizers.h"
    "${TK_DIST_INCLUDE_DIR}/tokenizers.h"
    COPYONLY
)

if(TOKENIZERS_RUST_TARGET)
    set(CARGO_TARGET_ARGS --target "${TOKENIZERS_RUST_TARGET}")
    set(CARGO_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/target/${TOKENIZERS_RUST_TARGET}/release")
else()
    set(CARGO_TARGET_ARGS "")
    set(CARGO_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/target/release")
endif()

if(MSVC AND NOT TOKENIZERS_RUST_TARGET MATCHES "gnu")
    set(RUST_BUILT_LIB "${CARGO_OUT_DIR}/tokenizers.lib")
else()
    set(RUST_BUILT_LIB "${CARGO_OUT_DIR}/libtokenizers.a")
endif()

add_custom_command(
    OUTPUT "${TK_DIST_LIB_DIR}/libtokenizers.a" "${TK_DIST_LIB_DIR}/tokenizers.lib"
    COMMAND "${CARGO_EXECUTABLE}" build --release ${CARGO_TARGET_ARGS}
    COMMAND "${CMAKE_COMMAND}" -E copy "${RUST_BUILT_LIB}" "${TK_DIST_LIB_DIR}/libtokenizers.a"
    COMMAND "${CMAKE_COMMAND}" -E copy "${RUST_BUILT_LIB}" "${TK_DIST_LIB_DIR}/tokenizers.lib"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/lib.rs"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/tokenizers.h"
    COMMENT "Building Rust staticlib (huggingface/tokenizers) via Cargo"
)
add_custom_target(tokenizers_rust_build ALL
    DEPENDS "${TK_DIST_LIB_DIR}/libtokenizers.a" "${TK_DIST_LIB_DIR}/tokenizers.lib"
)

add_library(tokenizers_static STATIC IMPORTED GLOBAL)
add_dependencies(tokenizers_static tokenizers_rust_build)
set_target_properties(tokenizers_static PROPERTIES
    IMPORTED_LOCATION "${TK_DIST_LIB_DIR}/libtokenizers.a"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

if(WIN32)
    target_link_libraries(tokenizers_static INTERFACE
        ws2_32 userenv advapi32 bcrypt ntdll crypt32 secur32
    )
elseif(APPLE)
    target_link_libraries(tokenizers_static INTERFACE
        "-framework Security" "-framework CoreFoundation" iconv c++ m pthread
    )
elseif(UNIX)
    target_link_libraries(tokenizers_static INTERFACE dl pthread m stdc++ gcc_s c)
endif()

add_library(tokenizers::tokenizers ALIAS tokenizers_static)

if(TOKENIZERS_BUILD_EXAMPLES)
    add_executable(basic_usage examples/basic_usage.c)
    target_link_libraries(basic_usage PRIVATE tokenizers_static)

    add_executable(batch_and_pair examples/batch_and_pair.c)
    target_link_libraries(batch_and_pair PRIVATE tokenizers_static)

    add_executable(streaming_decode examples/streaming_decode.c)
    target_link_libraries(streaming_decode PRIVATE tokenizers_static)
endif()

if(TOKENIZERS_BUILD_TESTS)
    enable_testing()
    add_executable(test_tokenizers tests/test_tokenizers.c)
    target_link_libraries(test_tokenizers PRIVATE tokenizers_static)
    add_test(NAME test_tokenizers COMMAND test_tokenizers)
endif()

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/tokenizers.h" DESTINATION include)
install(FILES "${TK_DIST_LIB_DIR}/libtokenizers.a" "${TK_DIST_LIB_DIR}/tokenizers.lib" DESTINATION lib)
