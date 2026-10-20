@echo off
setlocal EnableDelayedExpansion

:: ============================================================================
:: build.bat — Native Windows Build Script for tokenizers-c (Rust Core Wrapper)
::
:: Builds the Rust `tokenizers` crate staticlib via Cargo and outputs:
::   dist\include\tokenizers.h
::   dist\lib\libtokenizers.a
::   dist\lib\tokenizers.lib
::
:: Usage:
::   build.bat              Build staticlib using default Rust toolchain
::   build.bat msvc         Build staticlib for x86_64-pc-windows-msvc
::   build.bat gnu          Build staticlib for x86_64-pc-windows-gnu (MinGW-w64)
::   build.bat test         Build staticlib, compile tests, and run test suite
::   build.bat examples     Build staticlib and compile C example programs
::   build.bat clean        Remove build\, dist\, and target\ directories
:: ============================================================================

set "CMD_ARG=%~1"
if "%CMD_ARG%"=="" set "CMD_ARG=build"
set "BUILD_TARGET=%CMD_ARG%"

:: Test/example builds must use a Rust ABI that matches the selected C linker.
if /I "%CMD_ARG%"=="test" goto :select_test_target
if /I "%CMD_ARG%"=="examples" goto :select_test_target
goto :target_selected

:select_test_target
set "BUILD_TARGET=%~2"
if "%BUILD_TARGET%"=="" (
    set "BUILD_TARGET=msvc"
)
if /I not "%BUILD_TARGET%"=="msvc" if /I not "%BUILD_TARGET%"=="gnu" (
    echo [error] Test/example target must be msvc or gnu.
    echo         Usage: build.bat test [msvc^|gnu]
    exit /b 1
)

:target_selected

:: Load the Visual Studio compiler environment when called from a plain shell.
if /I "%BUILD_TARGET%"=="msvc" if /I "%CMD_ARG%"=="test" goto :setup_msvc
if /I "%BUILD_TARGET%"=="msvc" if /I "%CMD_ARG%"=="examples" goto :setup_msvc
goto :msvc_ready

:setup_msvc
where cl >nul 2>nul
if !ERRORLEVEL! EQU 0 goto :msvc_ready
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [error] Could not find Visual Studio Installer vswhere.exe.
    echo         Run this command from an x64 Native Tools Command Prompt for VS.
    exit /b 1
)
set "VS_PATH="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_PATH=%%i"
if not defined VS_PATH (
    echo [error] Visual Studio C++ Build Tools are not installed.
    echo         Install the MSVC x64/x86 build tools or use an x64 Native Tools Command Prompt.
    exit /b 1
)
call "%VS_PATH%\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 (
    echo [error] Failed to initialize the Visual Studio C++ build environment.
    exit /b 1
)

:msvc_ready

if /I "%CMD_ARG%"=="clean" (
    echo [clean] Removing build, dist, and Cargo target directories...
    if exist build rmdir /s /q build
    if exist dist rmdir /s /q dist
    where cargo >nul 2>nul && cargo clean
    exit /b 0
)

where cargo >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [error] Rust `cargo` not found in PATH. Install Rust from https://rustup.rs/
    exit /b 1
)

if not exist dist\include mkdir dist\include
if not exist dist\lib\pkgconfig mkdir dist\lib\pkgconfig
if not exist build mkdir build

copy /Y include\tokenizers.h dist\include\tokenizers.h >nul

set "CARGO_TARGET_FLAG="
set "CARGO_OUT_DIR=target\release"

if /I "%BUILD_TARGET%"=="msvc" (
    set "CARGO_TARGET_FLAG=--target x86_64-pc-windows-msvc"
    set "CARGO_OUT_DIR=target\x86_64-pc-windows-msvc\release"
)
if /I "%BUILD_TARGET%"=="gnu" (
    set "CARGO_TARGET_FLAG=--target x86_64-pc-windows-gnu"
    set "CARGO_OUT_DIR=target\x86_64-pc-windows-gnu\release"
)

if /I "%CMD_ARG%"=="test" (
    call "%~f0" %BUILD_TARGET% || exit /b 1
    goto :build_test
)
if /I "%CMD_ARG%"=="examples" (
    call "%~f0" %BUILD_TARGET% || exit /b 1
    goto :build_examples
)

echo [cargo] Building huggingface/tokenizers FFI staticlib...
cargo build --release %CARGO_TARGET_FLAG%
if %ERRORLEVEL% NEQ 0 exit /b 1

if exist "%CARGO_OUT_DIR%\libtokenizers.a" (
    copy /Y "%CARGO_OUT_DIR%\libtokenizers.a" dist\lib\libtokenizers.a >nul
    copy /Y "%CARGO_OUT_DIR%\libtokenizers.a" dist\lib\tokenizers.lib >nul
) else if exist "%CARGO_OUT_DIR%\tokenizers.lib" (
    copy /Y "%CARGO_OUT_DIR%\tokenizers.lib" dist\lib\tokenizers.lib >nul
    copy /Y "%CARGO_OUT_DIR%\tokenizers.lib" dist\lib\libtokenizers.a >nul
) else (
    echo [error] Could not locate static archive in %CARGO_OUT_DIR%
    exit /b 1
)

echo ==^> Built C header:       dist\include\tokenizers.h
echo ==^> Built static archive: dist\lib\libtokenizers.a + dist\lib\tokenizers.lib
goto :eof

:build_test
echo [test] Compiling and running C unit tests against Rust staticlib...
if /I "%BUILD_TARGET%"=="msvc" (
    where cl >nul 2>nul || (echo [error] MSVC target requires cl.exe.&& exit /b 1)
) else (
    where gcc >nul 2>nul || (echo [error] GNU target requires gcc.exe.&& exit /b 1)
)
if /I "%BUILD_TARGET%"=="msvc" (
    cl /nologo /O2 /W4 /Idist\include tests\test_tokenizers.c dist\lib\tokenizers.lib ws2_32.lib userenv.lib advapi32.lib bcrypt.lib ntdll.lib crypt32.lib secur32.lib /Fe:build\test_tokenizers.exe || exit /b 1
) else (
    gcc -O3 -Wall -Wextra -std=c11 -Idist\include tests\test_tokenizers.c dist\lib\libtokenizers.a -lws2_32 -luserenv -ladvapi32 -lbcrypt -lntdll -lcrypt32 -lsecur32 -lstdc++ -lm -lpthread -o build\test_tokenizers.exe || exit /b 1
)
build\test_tokenizers.exe || exit /b 1
goto :eof

:build_examples
echo [examples] Compiling C examples against Rust staticlib...
if /I "%BUILD_TARGET%"=="msvc" (
    where cl >nul 2>nul || (echo [error] MSVC target requires cl.exe.&& exit /b 1)
) else (
    where gcc >nul 2>nul || (echo [error] GNU target requires gcc.exe.&& exit /b 1)
)
if /I "%BUILD_TARGET%"=="msvc" (
    cl /nologo /O2 /W4 /Idist\include examples\basic_usage.c dist\lib\tokenizers.lib ws2_32.lib userenv.lib advapi32.lib bcrypt.lib ntdll.lib crypt32.lib secur32.lib /Fe:build\basic_usage.exe || exit /b 1
    cl /nologo /O2 /W4 /Idist\include examples\batch_and_pair.c dist\lib\tokenizers.lib ws2_32.lib userenv.lib advapi32.lib bcrypt.lib ntdll.lib crypt32.lib secur32.lib /Fe:build\batch_and_pair.exe || exit /b 1
) else (
    gcc -O3 -Wall -Wextra -std=c11 -Idist\include examples\basic_usage.c dist\lib\libtokenizers.a -lws2_32 -luserenv -ladvapi32 -lbcrypt -lntdll -lcrypt32 -lsecur32 -lstdc++ -lm -lpthread -o build\basic_usage.exe || exit /b 1
    gcc -O3 -Wall -Wextra -std=c11 -Idist\include examples\batch_and_pair.c dist\lib\libtokenizers.a -lws2_32 -luserenv -ladvapi32 -lbcrypt -lntdll -lcrypt32 -lsecur32 -lstdc++ -lm -lpthread -o build\batch_and_pair.exe || exit /b 1
)
echo ==^> Built examples: build\basic_usage.exe, build\batch_and_pair.exe
goto :eof
