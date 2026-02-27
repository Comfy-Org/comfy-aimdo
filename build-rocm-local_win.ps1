# comfy-aimdo Windows ROCm local build script
# Requires: Visual Studio, Git, and ROCm SDK
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$root = $PSScriptRoot
function F([string]$p) { $p.Replace('\','/') }  # forward-slash paths for clang rsp

# ── ROCm ───────────────────────────────────────────────────────────────────────
$rocmBase = if ($env:VIRTUAL_ENV -and (Test-Path "$env:VIRTUAL_ENV\Lib\site-packages\_rocm_sdk_core")) {
    "$env:VIRTUAL_ENV\Lib\site-packages\_rocm_sdk_core"
} elseif ($env:HIP_PATH)  { $env:HIP_PATH  } elseif ($env:ROCM_PATH) { $env:ROCM_PATH }
if (-not $rocmBase -or -not (Test-Path "$rocmBase\include\hip")) {
    throw "ROCm not found. Set HIP_PATH/ROCM_PATH or activate a venv with _rocm_sdk_core."
}
$clang = "$rocmBase\lib\llvm\bin\clang.exe"
if (-not (Test-Path $clang)) { throw "clang.exe not found at: $clang" }
Write-Host "ROCm: $rocmBase"

# ── Detours ────────────────────────────────────────────────────────────────────
$detoursDir = "$root\detours"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git not found. Please install Git and ensure it is on PATH." }
if (Test-Path $detoursDir) { Remove-Item -Recurse -Force $detoursDir }
git clone -q --depth 1 https://github.com/microsoft/Detours.git $detoursDir

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere.exe not found. Is Visual Studio installed?" }
$vcvars  = "$(& $vswhere -latest -property installationPath)\VC\Auxiliary\Build\vcvars64.bat"
$bat = [IO.Path]::Combine([IO.Path]::GetTempPath(), [IO.Path]::GetRandomFileName() + ".bat")
"@echo off`ncall `"$vcvars`" || exit /b 1`ncd /d `"$detoursDir\src`"`nnmake" | Set-Content $bat -Encoding ASCII
cmd /c $bat
Remove-Item $bat
if ($LASTEXITCODE -ne 0) { throw "Detours build failed (vcvars or nmake error)" }

# ── Compile ────────────────────────────────────────────────────────────────────
$sources = @(Get-Item "$root\src\*.c" -ErrorAction SilentlyContinue) + @(Get-Item "$root\src-win\*.c" -ErrorAction SilentlyContinue) | ForEach-Object { "`"$(F $_.FullName)`"" }
if (-not $sources) { throw "No .c source files found in src\ or src-win\." }

$rspFile = "$root\build.rsp"
[IO.File]::WriteAllLines($rspFile, @(
    "--target=x86_64-pc-windows-msvc", "-shared", "-O3"
    "-D__HIP_PLATFORM_AMD__", "-Wno-unused-command-line-argument"
    "-x", "c"
    "-I`"$(F $rocmBase)/include`"", "-I`"$(F $root)/src`"", "-I`"$(F $detoursDir)/include`""
    "-L`"$(F $rocmBase)/lib`"", "-L`"$(F $detoursDir)/lib.X64`""
    "-lamdhip64", "-ldxgi", "-ldxguid", "-ldetours"
    "-o", "`"$(F $root)/comfy_aimdo/aimdo.dll`""
) + $sources)

& $clang "@$rspFile"
$exitCode = $LASTEXITCODE
Remove-Item $rspFile -ErrorAction SilentlyContinue
if ($exitCode -ne 0) { throw "Build failed (exit code $exitCode)" }
Remove-Item -Recurse -Force $detoursDir
Write-Host "Build successful: comfy_aimdo\aimdo.dll"
