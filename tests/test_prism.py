import numpy as np
import pytest
import prism_ann


def test_basic_search():
    index = prism_ann.Index(dim=8, num_attributes=2)
    rng = np.random.default_rng(42)
    vectors = rng.random((100, 8), dtype=np.float32)
    attrs = rng.integers(0, 5, (100, 2)).astype(np.uint32)
    index.add(vectors, attrs)
    index.build()
    ids, dists = index.search(vectors[0], k=5, ef=50)
    assert ids.shape == (5,)
    assert dists.shape == (5,)
    assert ids[0] == 0  # nearest to itself
    assert dists[0] < 1e-6


def test_filtered_search():
    index = prism_ann.Index(dim=8, num_attributes=2)
    rng = np.random.default_rng(42)
    vectors = rng.random((200, 8), dtype=np.float32)
    attrs = np.tile(
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint32),
        (50, 1),
    )
    index.add(vectors, attrs)
    index.build()
    ids, dists = index.search(vectors[0], k=5, ef=50, filter={"attr_0": [0]})
    for i in ids:
        assert attrs[i, 0] == 0


def test_batch_search():
    index = prism_ann.Index(dim=8, num_attributes=1)
    rng = np.random.default_rng(42)
    vectors = rng.random((500, 8), dtype=np.float32)
    attrs = rng.integers(0, 3, (500, 1)).astype(np.uint32)
    index.add(vectors, attrs)
    index.build()
    queries = vectors[:10]
    ids, dists = index.batch_search(queries, k=5, ef=50)
    assert ids.shape == (10, 5)
    assert dists.shape == (10, 5)


def test_multiple_add():
    index = prism_ann.Index(dim=4, num_attributes=1)
    rng = np.random.default_rng(42)
    v1 = rng.random((50, 4), dtype=np.float32)
    a1 = np.zeros((50, 1), dtype=np.uint32)
    v2 = rng.random((50, 4), dtype=np.float32)
    a2 = np.ones((50, 1), dtype=np.uint32)
    index.add(v1, a1)
    index.add(v2, a2)
    assert index.num_points == 100
    index.build()
    assert index.is_built


def test_search_before_build_errors():
    index = prism_ann.Index(dim=4, num_attributes=1)
    query = np.zeros(4, dtype=np.float32)
    with pytest.raises(ValueError, match="not built"):
        index.search(query)


def test_add_after_build_errors():
    index = prism_ann.Index(dim=4, num_attributes=1)
    rng = np.random.default_rng(42)
    v = rng.random((10, 4), dtype=np.float32)
    a = np.zeros((10, 1), dtype=np.uint32)
    index.add(v, a)
    index.build()
    with pytest.raises(ValueError, match="cannot add after build"):
        index.add(v, a)


def test_wrong_dim_errors():
    index = prism_ann.Index(dim=8, num_attributes=2)
    v = np.zeros((10, 4), dtype=np.float32)  # wrong dim
    a = np.zeros((10, 2), dtype=np.uint32)
    with pytest.raises(ValueError, match="dim"):
        index.add(v, a)


def test_original_ids():
    """Search returns insertion-order IDs, not internal IDs."""
    index = prism_ann.Index(dim=4, num_attributes=2)
    rng = np.random.default_rng(42)
    vectors = rng.random((100, 4), dtype=np.float32)
    vectors[42] = [0.0, 0.0, 0.0, 0.0]
    attrs = rng.integers(0, 3, (100, 2)).astype(np.uint32)
    index.add(vectors, attrs)
    index.build()
    query = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ids, dists = index.search(query, k=1, ef=50)
    assert ids[0] == 42
