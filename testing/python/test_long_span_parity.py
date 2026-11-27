from support.corpus import long_spans


def test_unbroken_span_ids_match(native, encoding):
    """Check correctness independently of timing on four pathological spans."""
    cases = long_spans()
    actual = native.encode([data for _, data in cases])
    for (name, data), ids in zip(cases, actual, strict=True):
        assert ids == encoding.encode_ordinary(data.decode()), name
