def test_explicit_special_token_modes_match(native, encoding):
    """Compare allowed markers and ordinary marker text using explicit modes."""
    text = "a" + "🙂".join(encoding._special_tokens) + "b"
    assert native.encode([text.encode()], allow_special=True) == [
        encoding.encode(text, allowed_special="all")
    ]
    assert native.encode([text.encode()]) == [encoding.encode_ordinary(text)]
