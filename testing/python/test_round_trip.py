def test_both_engines_preserve_every_input_byte(differential, encoding):
    """Decode each engine's official IDs and compare the complete byte string."""
    for category, text, actual, expected in differential:
        assert encoding.decode_bytes(actual) == text.encode(), (category, ascii(text))
        assert encoding.decode_bytes(expected) == text.encode(), (category, ascii(text))
