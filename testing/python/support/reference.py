"""Load the official rank file and export a length-delimited test artifact."""

from hashlib import sha1, sha256
import os
from pathlib import Path
import struct

import tiktoken

RANK_SHA = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
RANK_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"


def load_encoding(rank_file=None):
    """Verify a supplied rank file before adding it to tiktoken's cache."""
    if rank_file is not None:
        data = Path(rank_file).read_bytes()
        if sha256(data).hexdigest() != RANK_SHA:
            raise ValueError("cl100k_base rank file SHA-256 mismatch")
        cache = Path(os.environ["TIKTOKEN_CACHE_DIR"])
        cache.mkdir(exist_ok=True, parents=True)
        (cache / sha1(RANK_URL.encode()).hexdigest()).write_bytes(data)
    return tiktoken.get_encoding("cl100k_base")


def frame(data):
    return struct.pack("<I", len(data)) + data


def export_ranks(encoding, path):
    """Keep official rank order. The C++ public importer validates the data."""
    ordered = sorted(encoding._mergeable_ranks.items(), key=lambda item: item[1])
    if [rank for _, rank in ordered] != list(range(len(ordered))):
        raise ValueError("Merge IDs must be consecutive")
    with path.open("wb") as output:
        output.write(struct.pack("<I", len(ordered)))
        for piece, _ in ordered:
            output.write(frame(piece))
        output.write(struct.pack("<I", len(encoding._special_tokens)))
        for piece, rank in sorted(encoding._special_tokens.items()):
            output.write(struct.pack("<I", rank))
            output.write(frame(piece.encode()))
