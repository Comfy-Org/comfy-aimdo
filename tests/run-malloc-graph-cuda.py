import os
from pathlib import Path
import subprocess
import sys


TESTS = (
    "malloc-graph-cuda.py",
    "malloc-graph-nested-cuda.py",
    "malloc-graph-empty-cuda.py",
    "malloc-graph-empty-subgraph-cuda.py",
    "malloc-graph-different-subgraph-cuda.py",
    "malloc-graph-subgraph-order-cuda.py",
    "malloc-graph-optional-subgraph-cuda.py",
    "malloc-graph-extra-subgraph-cuda.py",
    "malloc-graph-deep-nesting-cuda.py",
    "malloc-graph-different-stream-cuda.py",
    "malloc-graph-off-stream-cuda.py",
    "malloc-graph-odd-sizes-cuda.py",
    "malloc-graph-small-cuda.py",
    "malloc-graph-fragmentation-cuda.py",
    "malloc-graph-aliasing-cuda.py",
    "malloc-graph-long-run-cuda.py",
    "malloc-graph-multiple-graphs-cuda.py",
    "malloc-graph-destructor-cuda.py",
    "malloc-graph-poisoned-reuse-cuda.py",
    "malloc-graph-nested-stats-cuda.py",
    "malloc-graph-variable-subgraph-replay-cuda.py",
    "malloc-graph-changed-size-cuda.py",
    "malloc-graph-extra-allocation-cuda.py",
    "malloc-graph-missing-allocation-cuda.py",
    "malloc-graph-reordered-free-cuda.py",
    "malloc-graph-subgraph-phases-cuda.py",
    "malloc-graph-free-external-cuda.py",
    "malloc-graph-leak-cuda.py",
    "malloc-graph-free-external-subgraph-cuda.py",
    "malloc-graph-free-outer-subgraph-cuda.py",
    "malloc-graph-leak-subgraph-cuda.py",
)

directory = Path(__file__).parent
env = os.environ | {"PYTORCH_ALLOC_CONF": "backend:cudaMallocAsync"}
for test in TESTS:
    subprocess.run([sys.executable, directory / test], env=env, check=True)

print(f"All {len(TESTS)} CUDA malloc graph tests passed")
