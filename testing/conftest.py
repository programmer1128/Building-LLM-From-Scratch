"""Session fixtures keep build setup and oracle loading out of the test bodies."""

import os
from pathlib import Path

import pytest

from support.client import NativeTokenizer
from support.corpus import fixture
from support.reference import export_ranks, load_encoding

TESTING = Path(__file__).resolve().parent
BUILD = TESTING / "build"


def pytest_addoption(parser):
    parser.addoption(
        "--rank-file", type=Path, help="Verified local cl100k_base rank file"
    )


@pytest.fixture(scope="session")
def encoding(request, tmp_path_factory):
    cache = tmp_path_factory.mktemp("tiktoken-cache")
    previous = os.environ.get("TIKTOKEN_CACHE_DIR")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache)
    try:
        yield load_encoding(request.config.getoption("--rank-file"))
    finally:
        if previous is None:
            os.environ.pop("TIKTOKEN_CACHE_DIR", None)
        else:
            os.environ["TIKTOKEN_CACHE_DIR"] = previous


@pytest.fixture(scope="session")
def native(encoding, tmp_path_factory):
    executable = BUILD / "bin" / "tokenizer_adapter"
    if not executable.is_file():
        pytest.fail("Build testing/CMakeLists.txt before running pytest", pytrace=False)
    ranks = tmp_path_factory.mktemp("rank-fixture") / "ranks.bin"
    export_ranks(encoding, ranks)
    return NativeTokenizer(executable, ranks)


@pytest.fixture(scope="session")
def corpus():
    return fixture()


@pytest.fixture(scope="session")
def differential(native, encoding, corpus):
    documents, random_and_edges = corpus
    inputs = documents + random_and_edges
    actual = native.encode([text.encode() for _, text in inputs])
    return [
        (category, text, ids, encoding.encode_ordinary(text))
        for (category, text), ids in zip(inputs, actual, strict=True)
    ]
