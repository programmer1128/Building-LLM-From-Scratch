"""Subprocess transport for the C++ test adapter."""

import struct
import subprocess

from support.reference import frame


class NativeTokenizer:
    def __init__(self, executable, rank_file):
        self.executable = executable
        self.rank_file = rank_file

    def encode(self, inputs, allow_special=False):
        """Return one official-ID sequence per byte string, or report stderr."""
        mode = "specials" if allow_special else "diff"
        command = [str(self.executable), mode, str(self.rank_file)]
        result = subprocess.run(
            command,
            input=b"".join(frame(data) for data in inputs),
            capture_output=True,
            timeout=240,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"C++ adapter exited with {result.returncode}: "
                f"{result.stderr.decode(errors='replace')}"
            )

        output = result.stdout
        offset = 0
        sequences = []
        while offset < len(output):
            if offset + 4 > len(output):
                raise ValueError("Truncated output record header")
            count = struct.unpack_from("<I", output, offset)[0]
            offset += 4
            end = offset + 4 * count
            if end > len(output):
                raise ValueError("Truncated output token sequence")
            sequences.append(list(struct.unpack_from(f"<{count}I", output, offset)))
            offset = end
        if len(sequences) != len(inputs):
            raise ValueError("C++ adapter returned an incorrect number of records")
        return sequences
