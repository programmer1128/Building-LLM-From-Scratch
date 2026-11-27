"""Optional throughput and hardware-counter measurements."""

import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from statistics import median

from support.corpus import exact_size, long_spans
from support.reference import load_encoding


def emit(kind, *, json_lines=False, **fields):
    """Show concise timing rows by default and retain optional raw JSON output."""
    if json_lines:
        print(json.dumps({"kind": kind, **fields}), flush=True)
        return
    if kind == "benchmark_config":
        print(f"Timing on CPU {fields['cpu']}, {fields['repeats']} measured passes")
        print(
            f"{'Case':<26} {'Operation':<10} {'Engine':<10} {'MB/s':>10} {'Peak MiB':>10}"
        )
        return
    print(
        f"{fields['case']:<26} {fields['operation']:<10} {fields['backend']:<10} "
        f"{fields['mb_s']:>10.2f} {fields['peak_kib'] / 1024:>10.1f}",
        flush=True,
    )
    if kind == "profile":
        print(
            f"  Allocation calls and perf counters: {fields['samples'][0]['profile']}"
        )


def official_worker(args):
    encoding = load_encoding()
    data = args.worker.read_bytes().decode("utf-8")
    prepared = encoding.encode_ordinary(data) if args.operation == "decode" else None
    empty = set()
    encode_buffer = encoding._core_bpe.encode_to_tiktoken_buffer
    size = args.worker.stat().st_size

    def run():
        if prepared is not None:
            result = encoding.decode_bytes(prepared)
            if len(result) != size:
                raise ValueError("incorrect decoded size")
            return len(prepared)
        result = memoryview(encode_buffer(data, empty))
        return result.nbytes // 4

    profile = ctypes.CDLL(None) if os.getenv("TOKENIZER_PROFILE") == "1" else None
    for sample in range(-1, args.repeats):
        if sample < 0 and size >= 100_000_000:
            continue
        counters = (ctypes.c_uint64 * 8)()
        if profile:
            profile.token_profile_start()
        then = time.perf_counter()
        tokens = 0
        iterations = 1 if sample < 0 and size > 1_000_000 else args.passes
        for _ in range(iterations):
            tokens += run()
        elapsed = time.perf_counter() - then
        if profile:
            profile.token_profile_stop(counters)
        if sample >= 0:
            status = Path("/proc/self/status").read_text().splitlines()
            peak = next(
                int(line.split()[1]) for line in status if line.startswith("VmHWM:")
            )
            fields = {"seconds": elapsed, "tokens": tokens, "peak_kib": peak}
            if profile:
                fields["profile"] = list(counters)
            print(json.dumps(fields), flush=True)


def benchmark(binary, ranks, work, docs, args, profile):
    seed = "".join(s for _, s in docs).encode()
    short = "Let's test বাংলা, हिन्दी, 中文, and 🙂.\n".encode()
    tests = [
        ("short", short, 25_000, ("encode", "decode")),
        ("1 MB", exact_size(seed, 1_000_000), 1, ("encode", "decode")),
        ("10 MB", exact_size(seed, 10_000_000), 1, ("encode", "decode")),
        ("100 MB", None, 1, ("encode", "decode")),
        ("1 GB streamed", exact_size(seed, 10_000_000), 100, ("encode", "decode")),
        ("letter run 4 KiB", b"a" * 4_096, 1, ("encode",)),
        ("letter run 16 KiB", b"a" * 16_384, 1, ("encode",)),
    ] + [(label, payload, 1, ("encode",)) for label, payload in long_spans()]
    if args.whole_gb:
        tests.append(("1 GB whole", None, 1, ("encode",)))
    cpu = min(os.sched_getaffinity(0))
    unknown = set(args.case or ()) - {label for label, *_ in tests}
    if unknown:
        raise ValueError(f"unknown benchmark cases: {sorted(unknown)}")
    emit(
        "benchmark_config",
        json_lines=args.jsonl,
        cpu=cpu,
        repeats=args.repeats,
        seed_sha256=sha256(seed).hexdigest(),
    )
    for label, data, passes, operations in tests:
        if args.case and label not in args.case:
            continue
        path = work / "benchmark.txt"
        if data is None:
            size = 100_000_000 if label == "100 MB" else 1_000_000_000
            # Write exactly-sized input with low parent memory usage.
            block = exact_size(seed, 10_000_000)
            with path.open("wb") as stream:
                for _ in range(size // len(block)):
                    stream.write(block)
            import hashlib

            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
        else:
            path.write_bytes(data)
            size = len(data)
            digest = sha256(data).hexdigest()
        for op in operations:
            modes = [(False, args.repeats)]
            if label == "10 MB" and profile and op == "encode":
                modes.append((True, 1))
            for profiling, repeat in modes:
                for backend in ("cpp", "tiktoken"):
                    command = [
                        str(binary),
                        "bench",
                        str(ranks),
                        str(path),
                        str(passes),
                        str(repeat),
                        op,
                    ]
                    if backend == "tiktoken":
                        command = [
                            sys.executable,
                            str(
                                Path(__file__).resolve().parents[2]
                                / "run_tokenizer_bench.py"
                            ),
                            "--worker",
                            str(path),
                            "--passes",
                            str(passes),
                            "--repeats",
                            str(repeat),
                            "--operation",
                            op,
                        ]
                    env = os.environ.copy()
                    if profiling:
                        env["LD_PRELOAD"] = str(profile)
                        env["TOKENIZER_PROFILE"] = "1"
                    result = subprocess.run(
                        ["taskset", "-c", str(cpu), *command],
                        env=env,
                        capture_output=True,
                        timeout=900,
                    )
                    if result.returncode != 0:
                        raise RuntimeError(
                            f"{backend} {op} failed for {label}: "
                            f"{result.stderr.decode(errors='replace')}"
                        )
                    samples = [json.loads(line) for line in result.stdout.splitlines()]
                    elapsed = median(x["seconds"] for x in samples)
                    counts = {x["tokens"] for x in samples}
                    if len(counts) != 1:
                        raise ValueError("token count changed between benchmark passes")
                    emit(
                        "profile" if profiling else "benchmark",
                        json_lines=args.jsonl,
                        case=label,
                        operation=op,
                        backend=backend,
                        input_sha256=digest,
                        bytes_per_request=size,
                        passes=passes,
                        median_seconds=elapsed,
                        mb_s=size * passes / elapsed / 1e6,
                        peak_kib=max(x["peak_kib"] for x in samples),
                        tokens=counts.pop(),
                        samples=samples,
                    )
