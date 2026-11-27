import pytest


def test_random_unicode_ids_match_official_tiktoken(differential):
    """Report the first token difference across all 6,000 seeded Unicode inputs."""
    failures = [
        (text, actual, expected)
        for name, text, actual, expected in differential
        if name == "Random Unicode" and actual != expected
    ]
    if failures:
        text, actual, expected = failures[0]
        first_difference = next(
            (
                index
                for index, pair in enumerate(zip(actual, expected))
                if pair[0] != pair[1]
            ),
            min(len(actual), len(expected)),
        )
        pytest.fail(
            f"{len(failures)} / 6000 Unicode inputs differ. "
            f"First input starts with {ascii(text[:32])}. "
            f"Token offset {first_difference}: "
            f"C++ {actual[first_difference:first_difference + 4]}, "
            f"tiktoken {expected[first_difference:first_difference + 4]}.",
            pytrace=False,
        )
