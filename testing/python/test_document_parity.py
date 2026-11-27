def test_document_ids_match_official_tiktoken(differential):
    """Compare complete token sequences for all 5,400 templated documents."""
    failures = [
        (name, text, actual, expected)
        for name, text, actual, expected in differential
        if name not in {"Random Unicode", "Edge cases"} and actual != expected
    ]
    assert not failures, f"{len(failures)} document mismatches; first: {failures[:1]}"
