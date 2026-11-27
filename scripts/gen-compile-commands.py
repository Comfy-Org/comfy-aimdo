"""Generate compile_commands.json for clangd.

Mirrors the gcc compile line in scripts/build-linux-aimdo.sh so clangd gets
the same flags and include paths without needing bear/compiledb.
"""

import argparse
import json
import os
import re
import shlex
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_SCRIPT = os.path.join(ROOT, "scripts", "build-linux-aimdo.sh")

# Keep in sync with scripts/build-linux-aimdo.sh
BACKENDS = {
    "cuda": {
        "define": "-DAIMDO_CUDA",
        "dispatch": os.path.join("src-cuda", "dispatch.c"),
    },
    "rocm": {
        "define": "-D__HIP_PLATFORM_AMD__",
        "dispatch": os.path.join("src-hip", "dispatch.c"),
    },
}


def funchook_version() -> str:
    with open(BUILD_SCRIPT) as f:
        m = re.search(r"^FUNCHOOK_VERSION=(.+)$", f.read(), re.MULTILINE)
    if not m:
        print(f"FUNCHOOK_VERSION not found in {BUILD_SCRIPT}", file=sys.stderr)
        sys.exit(1)
    return m.group(1).strip().strip('"')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "backend",
        nargs="?",
        choices=sorted(BACKENDS),
        default="cuda",
        help="which build flavor to model (default: cuda)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=os.path.join(ROOT, "compile_commands.json"),
        help="output path (default: <repo>/compile_commands.json)",
    )
    args = parser.parse_args()

    backend = BACKENDS[args.backend]
    funchook_src = os.path.join(ROOT, "build", f"funchook-{funchook_version()}")

    cflags = [
        "-c",
        "-fPIC",
        "-O2",
        "-g",
        "-pthread",
        backend["define"],
        # Same position as ${AIMDO_EXTRA_CFLAGS:-} in the build script
        *shlex.split(os.environ.get("AIMDO_EXTRA_CFLAGS", "")),
        f"-I{ROOT}/src",
        f"-I{funchook_src}/include",
    ]

    def c_sources(subdir):
        d = os.path.join(ROOT, subdir)
        return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".c")]

    # The build line is: src/*.c <dispatch-dir>/dispatch.c src-posix/*.c
    sources = c_sources("src") + [os.path.join(ROOT, backend["dispatch"])] + c_sources("src-posix")

    compiler = os.environ.get("CC", "gcc")
    entries = []
    for src in sources:
        cmd = [compiler] + cflags + [src]
        entries.append(
            {
                "directory": ROOT,
                "arguments": cmd,
                "file": src,
            }
        )

    with open(args.output, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"wrote {len(entries)} {args.backend} entries to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
