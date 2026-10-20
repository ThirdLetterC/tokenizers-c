<#
.SYNOPSIS
    PowerShell build script for tokenizers-c (Rust Core Wrapper) on Windows.
.DESCRIPTION
    Builds dist\include\tokenizers.h + dist\lib\libtokenizers.a (and dist\lib\tokenizers.lib)
    by compiling the Rust `tokenizers` crate via Cargo.
.EXAMPLE
    .\build.ps1
    .\build.ps1 -Test -Examples
    .\build.ps1 -Target x86_64-pc-windows-gnu -Test
#>
param(
    [string]$Target = "",
    [switch]$Test,
    [switch]$Examples,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

if ($Clean) {
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    if (Test-Path "dist")  { Remove-Item -Recurse -Force "dist" }
    if (Get-Command cargo -ErrorAction SilentlyContinue) { cargo clean }
    Write-Host "==> Cleaned build, dist, and Cargo target directories."
    exit 0
}

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw "Rust `cargo` not found in PATH. Install Rust from https://rustup.rs/"
}

New-Item -ItemType Directory -Force -Path "dist\include", "dist\lib\pkgconfig", "build" | Out-Null
Copy-Item -Force "include\tokenizers.h" "dist\include\tokenizers.h"

Write-Host "==> Building Rust staticlib (huggingface/tokenizers) via Cargo..."
if ($Target -ne "") {
    cargo build --release --target $Target
    $OutDir = "target\$Target\release"
} else {
    cargo build --release
    $OutDir = "target\release"
}

if (Test-Path "$OutDir\libtokenizers.a") {
    Copy-Item -Force "$OutDir\libtokenizers.a" "dist\lib\libtokenizers.a"
    Copy-Item -Force "$OutDir\libtokenizers.a" "dist\lib\tokenizers.lib"
} elseif (Test-Path "$OutDir\tokenizers.lib") {
    Copy-Item -Force "$OutDir\tokenizers.lib" "dist\lib\tokenizers.lib"
    Copy-Item -Force "$OutDir\tokenizers.lib" "dist\lib\libtokenizers.a"
} else {
    throw "Could not find static archive in $OutDir"
}

Write-Host "==> Header ready:         dist\include\tokenizers.h"
Write-Host "==> Static archive ready: dist\lib\libtokenizers.a (and dist\lib\tokenizers.lib)"

$WinSysLibs = @("ws2_32.lib", "userenv.lib", "advapi32.lib", "bcrypt.lib", "ntdll.lib", "crypt32.lib", "secur32.lib")
$GccSysLibs = @("-lws2_32", "-luserenv", "-ladvapi32", "-lbcrypt", "-lntdll", "-lcrypt32", "-lsecur32", "-lstdc++", "-lm", "-lpthread")

if ($Test) {
    Write-Host "==> Building and running C unit test suite..."
    if (Get-Command cl -ErrorAction SilentlyContinue) {
        & cl /nologo /O2 /W4 /Idist\include tests\test_tokenizers.c dist\lib\tokenizers.lib @WinSysLibs /Fe:build\test_tokenizers.exe
    } else {
        & gcc -O3 -Wall -Wextra -std=c11 -Idist\include tests\test_tokenizers.c dist\lib\libtokenizers.a @GccSysLibs -o build\test_tokenizers.exe
    }
    & .\build\test_tokenizers.exe
}

if ($Examples) {
    Write-Host "==> Building C example programs..."
    if (Get-Command cl -ErrorAction SilentlyContinue) {
        & cl /nologo /O2 /W4 /Idist\include examples\basic_usage.c dist\lib\tokenizers.lib @WinSysLibs /Fe:build\basic_usage.exe
        & cl /nologo /O2 /W4 /Idist\include examples\batch_and_pair.c dist\lib\tokenizers.lib @WinSysLibs /Fe:build\batch_and_pair.exe
    } else {
        & gcc -O3 -Wall -Wextra -std=c11 -Idist\include examples\basic_usage.c dist\lib\libtokenizers.a @GccSysLibs -o build\basic_usage.exe
        & gcc -O3 -Wall -Wextra -std=c11 -Idist\include examples\batch_and_pair.c dist\lib\libtokenizers.a @GccSysLibs -o build\batch_and_pair.exe
    }
    & .\build\basic_usage.exe
    & .\build\batch_and_pair.exe
}
