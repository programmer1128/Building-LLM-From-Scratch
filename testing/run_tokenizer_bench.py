#!/usr/bin/env python3
"""Run optional benchmarks after building the CMake test targets."""

import argparse
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

TESTING = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTING / "python"))

from support.benchmark import benchmark, official_worker
from support.corpus import fixture
from support.reference import export_ranks, load_encoding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank-file", type=Path)
    parser.add_argument(
        "--jsonl", action="store_true", help="Print raw machine-readable samples"
    )
    parser.add_argument("--case", action="append")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--whole-gb", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--passes", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--operation", choices=["encode", "decode"], default="encode")
    args = parser.parse_args()
    if args.repeats < 1 or args.passes < 1:
        parser.error("Iteration counts must be positive")
    if args.worker:
        official_worker(args)
        return

    binary = TESTING / "build/bin/tokenizer_adapter"
    profile = TESTING / "build/lib/tokenizer_profile.so"
    if not binary.is_file():
        parser.error("Build testing/CMakeLists.txt before running benchmarks")

    with TemporaryDirectory(prefix="tokenizer-bench-", dir=TESTING) as directory:
        work = Path(directory)
        if args.rank_file:
            os.environ["TIKTOKEN_CACHE_DIR"] = str(work / "cache")
        encoding = load_encoding(args.rank_file)
        ranks = work / "ranks.bin"
        export_ranks(encoding, ranks)
        documents, _ = fixture()
        benchmark(
            binary, ranks, work, documents, args, profile if profile.is_file() else None
        )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        sys.exit(1)
