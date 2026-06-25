#!/bin/bash

set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BUILD_DIR="$ROOT_DIR/build"
CUDA_OUTPUT_PATH="$ROOT_DIR/comfy_aimdo/aimdo.so"
ROCM_OUTPUT_PATH="$ROOT_DIR/comfy_aimdo/aimdo_rocm.so"
FUNCHOOK_VERSION=1.1.3
FUNCHOOK_SRC="$BUILD_DIR/funchook-$FUNCHOOK_VERSION"
FUNCHOOK_TARBALL="$BUILD_DIR/funchook-$FUNCHOOK_VERSION.tar.gz"

ARCH=$(uname -m)

# Linkless builds no longer need CUDA stubs; only the funchook backend varies.
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    FUNCHOOK_DISASM=capstone
    FUNCHOOK_BUILD_DIR="$BUILD_DIR/funchook-$FUNCHOOK_VERSION-capstone"
else
    FUNCHOOK_DISASM=distorm
    FUNCHOOK_BUILD_DIR="$BUILD_DIR/funchook-$FUNCHOOK_VERSION-distorm"
fi

if [ ! -f "$FUNCHOOK_SRC/CMakeLists.txt" ]; then
    URL="https://github.com/kubo/funchook/releases/download/v$FUNCHOOK_VERSION/funchook-$FUNCHOOK_VERSION.tar.gz"

    mkdir -p "$BUILD_DIR"
    curl -fL "$URL" -o "$FUNCHOOK_TARBALL"
    rm -rf "$FUNCHOOK_SRC"
    tar -xzf "$FUNCHOOK_TARBALL" -C "$BUILD_DIR"
fi

if [ "$FUNCHOOK_DISASM" = "capstone" ] && ! grep -q "CAPSTONE_CMAKELISTS" "$FUNCHOOK_SRC/CMakeLists.txt"; then
    patch -d "$FUNCHOOK_SRC" -p1 --forward <<'PATCH'
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -86,6 +86,11 @@ if (DISASM_CAPSTONE)
   execute_process(COMMAND "${CMAKE_COMMAND}" --build .
       WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/capstone-download"
   )
 
+  # Capstone 4.0.2 still sets CMP0048 to OLD, which newer CMake rejects.
+  file(READ "${CMAKE_CURRENT_BINARY_DIR}/capstone-src/CMakeLists.txt" CAPSTONE_CMAKELISTS)
+  string(REPLACE "cmake_policy (SET CMP0048 OLD)" "cmake_policy (SET CMP0048 NEW)" CAPSTONE_CMAKELISTS "${CAPSTONE_CMAKELISTS}")
+  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/capstone-src/CMakeLists.txt" "${CAPSTONE_CMAKELISTS}")
+
   string(TOUPPER ${FUNCHOOK_CPU} FUNCHOOK_CPU_UPPER)
PATCH
fi

FUNCHOOK_READY=false
if [ "$FUNCHOOK_DISASM" = "capstone" ]; then
    [ -f "$FUNCHOOK_BUILD_DIR/libfunchook.a" ] && \
    [ -f "$FUNCHOOK_BUILD_DIR/capstone-build/libcapstone.a" ] && \
    FUNCHOOK_READY=true
else
    [ -f "$FUNCHOOK_BUILD_DIR/libfunchook.a" ] && \
    [ -f "$FUNCHOOK_BUILD_DIR/libdistorm.a" ] && \
    FUNCHOOK_READY=true
fi

if [ "$FUNCHOOK_READY" = "false" ]; then
    if ! command -v cmake >/dev/null 2>&1; then
        echo "cmake is required to build funchook" >&2
        exit 1
    fi

    mkdir -p "$FUNCHOOK_BUILD_DIR"

    cmake -S "$FUNCHOOK_SRC" -B "$FUNCHOOK_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DFUNCHOOK_BUILD_SHARED=OFF \
        -DFUNCHOOK_BUILD_STATIC=ON \
        -DFUNCHOOK_BUILD_TESTS=OFF \
        -DFUNCHOOK_DISASM="$FUNCHOOK_DISASM" \
        -DFUNCHOOK_INSTALL=OFF

    cmake --build "$FUNCHOOK_BUILD_DIR" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
fi

mkdir -p "$(dirname -- "$CUDA_OUTPUT_PATH")"

# Collect funchook + disassembler static libraries
FUNCHOOK_LIBS="$FUNCHOOK_BUILD_DIR/libfunchook.a"
if [ "$FUNCHOOK_DISASM" = "capstone" ]; then
    FUNCHOOK_LIBS="$FUNCHOOK_LIBS $FUNCHOOK_BUILD_DIR/capstone-build/libcapstone.a"
else
    FUNCHOOK_LIBS="$FUNCHOOK_LIBS $FUNCHOOK_BUILD_DIR/libdistorm.a"
fi

# Shared POSIX platform helpers (everything in src-posix EXCEPT the vendor
# funchook installers, which are mutually exclusive: cuda-funchooks.c for
# CUDA/HIP, ze-funchooks.c for XPU - each defines aimdo_setup_hooks).
POSIX_PLAT_SRCS="$ROOT_DIR/src-posix/model-mmap.c \
$ROOT_DIR/src-posix/module-load.c \
$ROOT_DIR/src-posix/hostbuf-plat.c \
$ROOT_DIR/src-posix/thread-plat.c \
$ROOT_DIR/src-posix/xfer-file-plat.c"

# shellcheck disable=SC2086
gcc -shared -o "$CUDA_OUTPUT_PATH" -fPIC -O2 -g -pthread \
    ${AIMDO_EXTRA_CFLAGS:-} \
    "$ROOT_DIR"/src/*.c "$ROOT_DIR"/src-cuda/dispatch.c \
    $POSIX_PLAT_SRCS "$ROOT_DIR"/src-posix/cuda-funchooks.c \
    -I"$ROOT_DIR/src" -I"$FUNCHOOK_SRC/include" \
    $FUNCHOOK_LIBS \
    -ldl

# shellcheck disable=SC2086
gcc -shared -o "$ROCM_OUTPUT_PATH" -fPIC -O2 -g -pthread \
    -D__HIP_PLATFORM_AMD__ \
    ${AIMDO_EXTRA_CFLAGS:-} \
    "$ROOT_DIR"/src/*.c "$ROOT_DIR"/src-hip/dispatch.c \
    $POSIX_PLAT_SRCS "$ROOT_DIR"/src-posix/cuda-funchooks.c \
    -I"$ROOT_DIR/src" -I"$FUNCHOOK_SRC/include" \
    $FUNCHOOK_LIBS \
    -ldl

# ---- Intel XPU (Level Zero) backend -----------------------------------------
# Opt-in: requires a Level Zero SDK (headers + libze_loader) pointed to by
# LEVEL_ZERO_DIR (containing include/ and lib/). Mirrors the Windows XPU build
# in build-win-xpu.bat: src-xpu (Level Zero shim + ze_loader hooks) + the shared
# POSIX platform helpers + the funchook installer, linked against ze_loader.
XPU_OUTPUT_PATH="$ROOT_DIR/comfy_aimdo/aimdo_xpu.so"
# Level Zero loader version to build when LEVEL_ZERO_DIR is not supplied. Must
# export zeDriverGetDefaultContext and zeDeviceSynchronize (newer L0 loaders).
LEVEL_ZERO_VERSION="${LEVEL_ZERO_VERSION:-1.30.0}"
# Intel XPU only exists on x86_64.
if [ "${AIMDO_BUILD_XPU:-0}" = "1" ] && [ "$ARCH" = "x86_64" ]; then
    # If no SDK was supplied, build the Level Zero loader from source (headers +
    # libze_loader.so) using the same cmake we use for funchook. The loader is
    # NOT shipped in the wheel (auditwheel excludes it); it is resolved from the
    # user's Intel driver at runtime.
    if [ -z "${LEVEL_ZERO_DIR:-}" ]; then
        LEVEL_ZERO_DIR="$BUILD_DIR/level-zero-$LEVEL_ZERO_VERSION/prefix"
        if [ ! -f "$LEVEL_ZERO_DIR/lib/libze_loader.so" ]; then
            if ! command -v cmake >/dev/null 2>&1; then
                echo "cmake is required to build the Level Zero loader" >&2
                exit 1
            fi
            L0_SRC="$BUILD_DIR/level-zero-$LEVEL_ZERO_VERSION/src"
            L0_TARBALL="$BUILD_DIR/level-zero-$LEVEL_ZERO_VERSION.tar.gz"
            if [ ! -f "$L0_SRC/CMakeLists.txt" ]; then
                mkdir -p "$BUILD_DIR"
                curl --retry 3 --retry-all-errors -fL "https://github.com/oneapi-src/level-zero/archive/refs/tags/v$LEVEL_ZERO_VERSION.tar.gz" \
                    -o "$L0_TARBALL"
                rm -rf "$L0_SRC"
                mkdir -p "$L0_SRC"
                tar -xzf "$L0_TARBALL" -C "$L0_SRC" --strip-components=1
            fi
            cmake -S "$L0_SRC" -B "$L0_SRC/build" \
                -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_INSTALL_PREFIX="$LEVEL_ZERO_DIR" \
                -DCMAKE_INSTALL_LIBDIR=lib
            cmake --build "$L0_SRC/build" --target install \
                -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
        fi
    fi
    : "${LEVEL_ZERO_DIR:?set LEVEL_ZERO_DIR to a Level Zero SDK (with include/ and lib/) for the XPU build}"
    # shellcheck disable=SC2086
    gcc -shared -o "$XPU_OUTPUT_PATH" -fPIC -O2 -g -pthread \
        -DAIMDO_XPU \
        ${AIMDO_EXTRA_CFLAGS:-} \
        "$ROOT_DIR"/src/*.c \
        "$ROOT_DIR"/src-xpu/dispatch.c "$ROOT_DIR"/src-xpu/ze-shim.c \
        $POSIX_PLAT_SRCS "$ROOT_DIR"/src-posix/ze-funchooks.c \
        -I"$ROOT_DIR/src" -I"$ROOT_DIR/src-xpu" \
        -I"$LEVEL_ZERO_DIR/include" -I"$FUNCHOOK_SRC/include" \
        $FUNCHOOK_LIBS \
        -L"$LEVEL_ZERO_DIR/lib" -lze_loader -ldl \
        -Wl,--no-undefined

    # Sanity: the artifact must dynamically depend on the exact loader SONAME
    # that auditwheel excludes (libze_loader.so.1). It is resolved from the
    # user's Intel driver at runtime, not bundled into the wheel. If the
    # DT_NEEDED ever diverges from the exclude in build-wheels.yml, auditwheel
    # would try to graft the loader and fail, so assert the SONAME here.
    if ! command -v readelf >/dev/null 2>&1; then
        echo "readelf is required to validate aimdo_xpu.so dependencies" >&2
        exit 1
    fi
    readelf -d "$XPU_OUTPUT_PATH" | grep -Fq 'Shared library: [libze_loader.so.1]' || {
        readelf -d "$XPU_OUTPUT_PATH" >&2
        echo "aimdo_xpu.so must declare DT_NEEDED libze_loader.so.1" >&2
        exit 1
    }
fi
