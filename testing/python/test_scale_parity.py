from support.corpus import exact_size


def test_one_megabyte_ids_match(native, encoding, corpus):
    """Compare full output sequences on the smallest scaling workload."""
    documents, _ = corpus
    seed = "".join(text for _, text in documents).encode()
    data = exact_size(seed, 1_000_000)
    assert native.encode([data]) == [encoding.encode_ordinary(data.decode())]
