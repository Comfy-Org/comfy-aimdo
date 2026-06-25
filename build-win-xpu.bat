@echo off
REM Native Windows build of the Intel XPU (Level Zero) aimdo backend.
REM
REM Mirrors the CUDA Windows build in .github/workflows/build-wheels.yml but
REM swaps the CUDA SDK + nvcuda detour for the Level Zero SDK + ze_loader detour.
REM
REM The MSVC environment (cl.exe, INCLUDE, LIB) must already be set up by the
REM caller (vcvars64.bat in CI, or the portable-MSVC setup_env.bat locally).
REM
REM Override these to point at your Level Zero SDK and Detours checkout:
if "%LEVEL_ZERO_SDK%"=="" set LEVEL_ZERO_SDK=C:\Users\kosin\aimdo-xpu-spike\sdk\level-zero
if "%DETOURS_DIR%"==""    set DETOURS_DIR=C:\Users\kosin\aimdo-xpu-spike\Detours

REM src-win\*.c minus cuda-detour.c: the CUDA detour (nvcuda) defines the same
REM aimdo_setup_hooks/aimdo_teardown_hooks as our Level Zero detour
REM (src-xpu\ze-detour.c), so including it would be a duplicate-symbol clash.
REM The remaining src-win files are platform helpers the upstream src\*.c now
REM depends on (hostbuf / xfer-file / thread / module-load).
cl.exe /LD /O2 ^
  src\*.c src-xpu\*.c ^
  src-win\model-mmap.c src-win\shmem-detect.c src-win\module-load.c ^
  src-win\hostbuf-plat.c src-win\thread-plat.c src-win\xfer-file-plat.c ^
  /DAIMDO_XPU /FIcompiler.h /Isrc-xpu /Isrc /Isrc-win ^
  /I"%LEVEL_ZERO_SDK%\include" /I"%DETOURS_DIR%\include" /Fe:comfy_aimdo\aimdo.dll ^
  /link /LIBPATH:"%LEVEL_ZERO_SDK%\lib" /LIBPATH:"%DETOURS_DIR%\lib.X64" ^
  ze_loader.lib detours.lib dxgi.lib dxguid.lib onecore.lib
