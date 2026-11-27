# Tokenizer tests

The C++ suite uses GoogleTest 1.17 through CMake and CTest. The Python suite uses pytest 9.0.2 and the official tiktoken wheel. Each test case has its own source file.

## Build and run

Run these commands from the repository root. Install GoogleTest development headers, Boost development headers, CMake, and a C++20 compiler through your system package manager.

```bash
testing/.venv/bin/python -m pip install -r testing/requirements.txt
cmake -S testing -B testing/build -DCMAKE_BUILD_TYPE=Release
cmake --build testing/build -j 4
ctest --test-dir testing/build --output-on-failure
testing/.venv/bin/python -m pytest testing/python
```

For an offline run, append `--rank-file /path/to/cl100k_base.tiktoken` to pytest. The loader verifies its SHA-256. Each test uses temporary artifacts. Source-relative includes and CMake target dependencies handle header and source locations. The compiler command needs no manually supplied include paths or access-control overrides.

The refactored suite was validated on 25 November 2026. All nine GoogleTest cases passed. Pytest passed eight cases and failed the Unicode parity case on the same three known inputs. A one-pass 1 MB benchmark smoke run completed for both engines.

Pytest prints individual pass/fail results and a short failure report. The Unicode parity test currently fails on three known inputs because the splitter and the installed tiktoken Rust regex classify some CJK characters differently. The test retains that failure as a compatibility requirement.

## Files and responsibilities

```text
testing/
├── CMakeLists.txt                   # Engine, GoogleTest suite, adapter, profiling targets
├── .clang-format                   # Four spaces, Allman braces, explicit control-flow braces
├── pytest.ini                      # Python discovery and concise failure output
├── requirements.txt                # Pinned Python test and formatting dependencies
├── conftest.py                     # Shared pytest fixtures and rank-file option
├── cpp/
│   ├── tests/                      # One GoogleTest case per file
│   ├── support/test_support.h/.cpp  # Binary fixture I/O and public rank import
│   └── adapter/tokenizer_adapter.cpp
├── python/
│   ├── test_*.py                    # One pytest test per file
│   └── support/
│       ├── reference.py            # Official rank loading and fixture export
│       ├── client.py               # C++ adapter process transport
│       ├── corpus.py               # Seeded multilingual and stress inputs
│       └── benchmark.py            # Optional timing workers
├── run_tokenizer_bench.py           # Benchmark CLI using prebuilt CMake targets
└── tokenizer_profile.c              # Optional Linux allocation/perf probe
```

The adapter translates native byte IDs to official IDs after encoding and checks byte recovery. Tests call `importRankedVocabulary` and `addSpecialTokens` through the tokenizer's public interface. The rank importer validates a replacement before installing it. Imported multi-split rank tables cannot be represented by ABPE v1, so saving one reports an explicit error. Native trained vocabulary save/load is covered independently.

## Optional performance runs

```bash
testing/.venv/bin/python testing/run_tokenizer_bench.py --case '1 MB'
testing/.venv/bin/python testing/run_tokenizer_bench.py --case '10 MB' --jsonl
```

The benchmark prints a readable table by default. `--jsonl` includes raw samples, hashes, and counters for reports. Use `--whole-gb --case '1 GB whole' --repeats 1` for the large capacity test. Timings run on the first CPU allowed by the process affinity. Pytest assertions run independently of performance measurements.

For native race detection, configure a separate build with `-DTOKENIZER_ENABLE_TSAN=ON` and run CTest there. The compiler's ThreadSanitizer runtime must be installed.

Format Python with `testing/.venv/bin/python -m black --workers 1 testing/conftest.py testing/run_tokenizer_bench.py testing/python`. Format the C/C++ test files with clang-format, which reads `testing/.clang-format`.
