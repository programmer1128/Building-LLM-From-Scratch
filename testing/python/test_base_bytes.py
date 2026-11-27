def test_all_256_base_byte_ids_match(native, encoding):
    """Raw-byte coverage includes NUL and bytes that are invalid UTF-8 alone."""
    inputs = [bytes([value]) for value in range(256)]
    actual = native.encode(inputs)
    expected = [[encoding._mergeable_ranks[data]] for data in inputs]
    assert actual == expected
