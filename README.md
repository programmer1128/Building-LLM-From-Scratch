# Building GPT from scratch in C++

This repository implements a language model from scratch in C++.

### Exact IDs and byte preservation

| Input | Full sequences matching tiktoken | Exact byte round trips, both engines |
| --- | ---: | ---: |
| English, Bengali, Hindi, Chinese, C++, Java, Python, math/JSON, and emoji/mixed text (600 each) | 5,400 / 5,400 | 5,400 / 5,400 |
| Seeded Unicode fuzzing | 5,997 / 6,000 | 6,000 / 6,000 |
| Targeted edge cases | 19 / 19 | 19 / 19 |
| **Total** | **11,416 / 11,419 (99.97%)** | **11,419 / 11,419** |

The harness compared every token ID in order, after translating C++ base-byte IDs to official IDs. All 256 singleton bytes passed the rank and C++ raw-byte round-trip check separately. Full IDs also matched on the 1 MB timing fixture and on all four 64 KiB stress inputs. The larger timing fixtures were checked for equal token counts.

A minimal failing input is `U+32A0C U+09BE U+09B2`. The C++ output in official IDs is `[172, 110, 101, 234, 50228, 110]`. Tiktoken emits `[172, 110, 101, 234, 42412, 11372, 110]`. An isolated regex probe found the same classification difference for U+32A0C, U+32434 and U+32F26. The C++ splitter treats them as letters, while the tested Rust regex groups them with nonletters. Applying the official Rust BPE to the C++ spans reproduces the C++ IDs for all three failures.

### Single-thread throughput

Median wall time over three measured passes per engine, converted to decimal MB/s. Each engine used the same inputs and one thread pinned to logical CPU 0.

| Mixed text | C++ encode (MB/s) | tiktoken encode (MB/s) | C++ decode (MB/s) | tiktoken decode (MB/s) |
| --- | ---: | ---: | ---: | ---: |
| 1 MB | 60.00 | 11.66 | 501.22 | 133.09 |
| 10 MB | 59.34 | 11.45 | 479.28 | 118.26 |
| 100 MB | 57.28 | 9.61 | 381.98 | 113.44 |
| 1 GB across 100 × 10 MB requests | 46.26 | 7.90 | 422.27 | 119.97 |

The 1 GB aggregate reuses one in-memory 10 MB block for 100 independent calls per sample. The recorded C++ run warmed up with a full batch, while the tiktoken run warmed up with one request.

For a 66-byte, 24-token multilingual request, median sample time divided by 25,000 requests was **1.15 vs 7.00 µs** to encode and **0.14 vs 0.79 µs** to decode, C++ first. These are warmed loop averages, with no request transport.

Both engines also encoded a **single 1 GB request** in a separate one-pass capacity test pinned to logical CPU 2. C++ took **30.96 s** (32.29 MB/s, 6.96 GiB peak RSS) and tiktoken took **129.85 s** (7.70 MB/s, 4.84 GiB peak RSS). Each produced **393,010,000 IDs**. The 1 GB decode test uses 10 MB requests to keep the Python-side token buffers within available memory.

Both implementations returned a contiguous native ID buffer for timed encoding. C++ uses `encode(text, false)`, and the official Python wheel exposes its Rust output via `encode_to_tiktoken_buffer` with an empty set of allowed special tokens. That Rust entry point still runs its special-token regex. Decode timing includes the Python binding’s ID conversion and byte copy for tiktoken. C++ decodes a native vector. Rank loading, file I/O, input preparation, and the C++ ID translation are outside the timing. The ratios compare these entry points on this corpus. [tiktoken 0.14.0 Rust core](https://github.com/openai/tiktoken/blob/0.14.0/src/lib.rs), [Python binding](https://github.com/openai/tiktoken/blob/0.14.0/src/py.rs)

### Stress inputs

Each row is one UTF-8 request with no span breaks. Times are medians over three runs, pinned to logical CPU 2 for both implementations.

| Input | C++ encode | tiktoken encode |
| --- | ---: | ---: |
| 4 KiB of `a` | 10.81 ms | 0.74 ms |
| 16 KiB of `a` | 159.02 ms | 3.54 ms |
| 64 KiB of `a` | 2,431.66 ms | 15.42 ms |
| 64 KiB of spaces | 2,568.47 ms | 20.60 ms |
| 64 KiB of `!` | 2,788.87 ms | 16.36 ms |
| 64 KiB of `🙂` | 1,859.83 ms | 4.04 ms |

Increasing the repeated letter input fourfold increased C++ wall time by about 15× at each step. The C++ merge loop repeatedly scans and compacts a vector. Tiktoken 0.14.0 uses a heap to schedule merges on long pieces. [C++ merge loop](exp/tokenizers/BPE_Tokenizer.cpp#L334), [tiktoken merge implementation](https://github.com/openai/tiktoken/blob/0.14.0/src/lib.rs)

### Memory and hardware counters

Peak process RSS includes input, vocabulary, allocator retention, and Python runtime where applicable. The decode process also holds IDs prepared by an encode before timing. Values are Linux `VmHWM` from a separate three-pass run pinned to logical CPU 2.

| Request | C++ encode peak | tiktoken encode peak | C++ decode peak | tiktoken decode peak |
| --- | ---: | ---: | ---: | ---: |
| 1 MB | 24.2 MiB | 69.5 MiB | 23.9 MiB | 78.4 MiB |
| 10 MB | 90.8 MiB | 111.6 MiB | 90.6 MiB | 239.6 MiB |
| 100 MB | 730.3 MiB | 554.2 MiB | 729.9 MiB | 1,853.3 MiB |

At 100 MB, C++ used more peak memory for encoding. Its splitter materializes all spans in a vector before BPE. The Rust encoder iterates over regex matches as it encodes. [C++ splitter](exp/tokenizers/SplitStateMachine.cpp#L513), [Rust encoder](https://github.com/openai/tiktoken/blob/0.14.0/src/lib.rs)

One separate warmed 10 MB encoding pass on CPU 0 produced these counts. The allocation probe counts calls into the interposed glibc allocation functions on the encoding thread. The CPU counters cover user space and were collected via Linux `perf_event_open`.

| Counter | C++ | tiktoken |
| --- | ---: | ---: |
| Allocation and reallocation calls | 3 | 1,512,080 |
| CPU cycles | 578,436,546 | 3,041,968,100 |
| Hardware cache misses | 475,652 | 526,314 |
| Branch misses | 557,240 | 4,539,788 |

### Concurrency and vocabulary artifacts

- A shared, immutable C++ tokenizer completed **137,028 encode/decode requests per worker configuration** with **0 output discrepancies** at 2, 4, and 8 threads. Its merge scratch buffer is thread local. The GCC ThreadSanitizer build could not link because the installed toolchain lacks `libtsan.so.2.0.0`.
- A newly trained native vocabulary serialized to a 288-byte ABPE v1 file. Reloading preserved ordinary IDs, allowed-special IDs and byte decoding. Saving again reproduced the exact file. A truncated artifact, bad magic, and unsupported version were rejected. The official rank table was imported separately for comparison.
- With the five official special tokens imported, C++ matched tiktoken with all specials allowed and with markers encoded as ordinary text. Their defaults differ. C++ recognizes registered marker strings by default, while tiktoken rejects them unless allowed or explicitly treated as ordinary text.


### Maintained test suite

The tests now use GoogleTest through CMake/CTest and pytest, with one test case per source file. The adapter imports ranks through the public tokenizer interface. C++ compilation enforces normal access control. The performance tables above record the earlier benchmark run.

```bash
cmake -S testing -B testing/build -DCMAKE_BUILD_TYPE=Release
cmake --build testing/build -j 4
ctest --test-dir testing/build --output-on-failure
testing/.venv/bin/python -m pytest testing/python
```

The Unicode parity test reports the three known input mismatches as a failure. Benchmark measurements have their own runner, which prints a table by default and offers `--jsonl` for raw samples. The test layout, dependency installation, offline rank option, and formatting instructions are documented in the testing guide. [Testing guide](testing/README.md)
