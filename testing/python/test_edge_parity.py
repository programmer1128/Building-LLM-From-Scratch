def test_edge_case_ids_match_official_tiktoken(differential):
    """Exercise NUL, combining marks, whitespace, marker literals, and emojis."""
    failures = [
        (ascii(text), actual, expected)
        for name, text, actual, expected in differential
        if name == "Edge cases" and actual != expected
    ]
    assert not failures, f"Edge case mismatches: {failures[:1]}"
