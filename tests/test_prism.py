import inspect
import sys
import threading

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


def test_batch_padding_and_per_query_predicates():
    vectors = np.array(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]],
        dtype=np.float32,
    )
    attrs = np.array([[0], [0], [1], [1]], dtype=np.uint32)
    index = prism_ann.Index(dim=2, num_attributes=1)
    index.add(vectors, attrs)
    index.build()

    ids, dists = index.batch_search(
        vectors[:2],
        k=5,
        ef=0,
        filters=[{"attr_0": [0]}, {"attr_0": [99]}],
    )
    padding = np.iinfo(np.uint32).max
    assert set(ids[0, :2].tolist()) == {0, 1}
    assert np.all(ids[0, 2:] == padding)
    assert np.all(np.isinf(dists[0, 2:]))
    assert np.all(ids[1] == padding)
    assert np.all(np.isinf(dists[1]))


def test_batch_broadcast_filter_applies_to_every_query():
    vectors = np.array(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]],
        dtype=np.float32,
    )
    attrs = np.array([[0], [0], [1], [1]], dtype=np.uint32)
    index = prism_ann.Index(dim=2, num_attributes=1)
    index.add(vectors, attrs)
    index.build()

    ids, _ = index.batch_search(
        vectors[:2], k=2, ef=0, filter={"attr_0": [1]}
    )
    assert ids.shape == (2, 2)
    assert np.all(attrs[ids, 0] == 1)


def test_batch_outputs_reject_unrepresentable_shapes():
    vectors = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    attrs = np.zeros((2, 1), dtype=np.uint32)
    index = prism_ann.Index(dim=2, num_attributes=1)
    index.add(vectors, attrs)
    index.build()

    with pytest.raises(ValueError, match="output"):
        index.batch_search(vectors[:1], k=sys.maxsize, ef=1)

    ivf, _ = _build_small_ivf(np.float32)
    with pytest.raises(ValueError, match="output"):
        ivf.search(
            vectors[:1],
            np.array([0, 0], dtype=np.int64),
            np.array([], dtype=np.int32),
            k=sys.maxsize,
            ef=1,
            nprobe=1,
        )


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


def test_failed_build_preflight_preserves_staged_vectors():
    index = prism_ann.Index(dim=2, num_attributes=1)
    extreme = np.full((1, 2), np.finfo(np.float32).max, dtype=np.float32)
    attrs = np.zeros((1, 1), dtype=np.uint32)
    index.add(extreme, attrs)

    with pytest.raises(ValueError, match="absolute magnitude"):
        index.build()
    assert index.num_points == 1
    assert not index.is_built
    with pytest.raises(ValueError, match="absolute magnitude"):
        index.build()
    assert index.num_points == 1

    valid_index = prism_ann.Index(dim=2, num_attributes=1)
    valid = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    valid_attrs = np.zeros((2, 1), dtype=np.uint32)
    valid_index.add(valid, valid_attrs)
    valid_index.build()
    ids, _ = valid_index.search(valid[0], k=1, ef=1)
    assert ids.tolist() == [0]


def test_failed_schema_preflight_preserves_staged_vectors():
    index = prism_ann.Index(
        dim=1,
        num_attributes=64,
        t=32,
        m_greedy=1,
        m_random=0,
        scan_threshold=0,
    )
    vectors = np.array([[0.0], [1.0]], dtype=np.float32)
    attrs = np.zeros((2, 64), dtype=np.uint32)
    attrs[1, :] = 1
    index.add(vectors, attrs)

    with pytest.raises(ValueError, match="more than 4096 subsets"):
        index.build()
    assert index.num_points == 2
    assert not index.is_built


def test_wrong_dim_errors():
    index = prism_ann.Index(dim=8, num_attributes=2)
    v = np.zeros((10, 4), dtype=np.float32)  # wrong dim
    a = np.zeros((10, 2), dtype=np.uint32)
    with pytest.raises(ValueError, match="dim"):
        index.add(v, a)


def test_graph_expansion_is_explicit_and_validated():
    with pytest.raises(ValueError, match="graph_expansion"):
        prism_ann.Index(dim=4, num_attributes=1, graph_expansion=0)

    index = prism_ann.Index(
        dim=4, num_attributes=1, graph_expansion=1, build_seed=123
    )
    vectors = np.arange(80, dtype=np.float32).reshape(20, 4)
    attrs = np.zeros((20, 1), dtype=np.uint32)
    index.add(vectors, attrs)
    index.build()
    ids, _ = index.search(vectors[0], k=3, ef=1)
    assert len(ids) == 3


def test_scan_threshold_is_exposed_and_searchable():
    with pytest.raises(OverflowError):
        prism_ann.Index(dim=2, num_attributes=1, scan_threshold=-1)

    vectors = np.array(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]],
        dtype=np.float32,
    )
    attrs = np.array([[0], [0], [1], [1]], dtype=np.uint32)
    for threshold in (0, 20_000):
        index = prism_ann.Index(
            dim=2, num_attributes=1, scan_threshold=threshold
        )
        index.add(vectors, attrs)
        index.build()
        ids, distances = index.search(
            vectors[0], k=2, ef=2, filter={"attr_0": [0]}
        )
        assert ids.tolist() == [0, 1]
        assert np.all(np.isfinite(distances))


def test_multi_cell_scan_threshold_is_exposed_and_searchable():
    with pytest.raises(OverflowError):
        prism_ann.Index(
            dim=2,
            num_attributes=1,
            multi_cell_scan_threshold=-1,
        )

    vectors = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
            [9.0, 9.0],
            [9.1, 9.0],
        ],
        dtype=np.float32,
    )
    attrs = np.array([[0], [0], [1], [1], [2], [2]], dtype=np.uint32)
    index = prism_ann.Index(
        dim=2,
        num_attributes=1,
        scan_threshold=0,
        multi_cell_scan_threshold=4,
    )
    index.add(vectors, attrs)
    index.build()
    predicate = {"attr_0": [0, 2]}
    ids, distances = index.search(vectors[0], k=3, ef=3, filter=predicate)
    exact_ids, exact_distances = index.search_exact(vectors[0], k=3, filter=predicate)
    assert ids.tolist() == exact_ids.tolist()
    assert np.allclose(distances, exact_distances)


def test_cross_cell_exact_ranking_limit_is_exposed_and_validated():
    with pytest.raises(ValueError, match="cross_cell_exact_ranking_limit"):
        prism_ann.Index(
            dim=2,
            num_attributes=1,
            cross_cell_exact_ranking_limit=8193,
        )

    index = prism_ann.Index(
        dim=2,
        num_attributes=1,
        m_local=2,
        m_greedy=2,
        m_random=0,
        beam_width=4,
        cross_cell_exact_ranking_limit=0,
        scan_threshold=0,
    )
    vectors = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
            [9.0, 9.0],
            [9.1, 9.0],
        ],
        dtype=np.float32,
    )
    attrs = np.array([[0], [0], [1], [1], [2], [2]], dtype=np.uint32)
    index.add(vectors, attrs)
    index.build()
    ids, distances = index.search(
        vectors[2], k=2, ef=4, filter={"attr_0": [1]}
    )
    assert set(ids.tolist()) == {2, 3}
    assert np.all(np.isfinite(distances))


def test_build_seed_is_exposed_as_an_unsigned_64_bit_value():
    prism_ann.Index(dim=4, num_attributes=1, build_seed=2**64 - 1)
    with pytest.raises(OverflowError):
        prism_ann.Index(dim=4, num_attributes=1, build_seed=2**64)


def test_python_constructor_defaults_match_core_policy():
    parameters = inspect.signature(prism_ann.Index).parameters
    assert parameters["m_local"].default == 48
    assert parameters["beam_width"].default == 128
    assert parameters["scan_threshold"].default == 20_000
    assert parameters["multi_cell_scan_threshold"].default == 500_000


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


def test_cosine_metric_and_exact_filtered_baseline():
    index = prism_ann.Index(dim=3, num_attributes=1, metric="cosine")
    vectors = np.array(
        [[10.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    attrs = np.array([[1], [1], [2]], dtype=np.uint32)
    index.add(vectors, attrs)
    index.build()

    ids, dists = index.search_exact(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        k=2,
        filter={"attr_0": [1]},
    )
    assert ids.tolist() == [0, 1]
    assert dists[0] == pytest.approx(0.0, abs=1e-6)


def test_python_inner_product_uses_exact_filtered_ranking():
    vectors = np.array(
        [[0.5, 0.0], [0.6, 0.1], [20.0, 0.0], [100.0, 100.0]],
        dtype=np.float32,
    )
    attrs = np.array([[1], [1], [1], [2]], dtype=np.uint32)
    index = prism_ann.Index(dim=2, num_attributes=1, metric="inner_product")
    index.add(vectors, attrs)
    index.build()

    query = np.array([1.0, 0.0], dtype=np.float32)
    ids, dists = index.search(query, k=2, ef=1, filter={"attr_0": [1]})
    exact_ids, exact_dists = index.search_exact(
        query, k=2, filter={"attr_0": [1]}
    )
    assert ids.tolist() == [2, 1]
    assert ids.tolist() == exact_ids.tolist()
    assert dists.tolist() == pytest.approx(exact_dists.tolist())


def test_noncontiguous_add_and_small_ef_are_supported():
    rng = np.random.default_rng(7)
    vectors = np.asfortranarray(rng.random((30, 4), dtype=np.float32))
    attrs = np.asfortranarray(np.zeros((30, 1), dtype=np.uint32))
    index = prism_ann.Index(dim=4, num_attributes=1, binary_rerank=0)
    index.add(vectors, attrs)
    index.build()

    # A row of this Fortran-order array is strided. Both single-query methods
    # must copy it before releasing the GIL rather than borrowing NumPy memory.
    assert not vectors[0].flags.c_contiguous
    ids, _ = index.search(vectors[0], k=10, ef=0, filter={"attr_0": [0]})
    exact_ids, _ = index.search_exact(vectors[0], k=10, filter={"attr_0": [0]})
    assert len(ids) == 10
    assert len(set(ids.tolist())) == 10
    assert len(exact_ids) == 10
    assert len(set(exact_ids.tolist())) == 10


def test_single_search_methods_release_the_gil():
    rng = np.random.default_rng(123)
    n, dim = 200_000, 64
    vectors = rng.random((n, dim), dtype=np.float32)
    attrs = np.zeros((n, 1), dtype=np.uint32)
    index = prism_ann.Index(
        dim=dim, num_attributes=1, metric="inner_product"
    )
    index.add(vectors, attrs)
    index.build()
    query = rng.random(dim, dtype=np.float32)

    def assert_releases_gil(call):
        trigger = threading.Event()
        progressed = threading.Event()

        def waiter():
            trigger.wait()
            progressed.set()

        thread = threading.Thread(target=waiter)
        thread.start()
        previous_interval = sys.getswitchinterval()
        try:
            # Prevent ordinary bytecode time-slicing between trigger.set() and
            # the native call. The waiter can progress only when Rust releases
            # the GIL (or after this interval is restored).
            sys.setswitchinterval(1.0)
            trigger.set()
            call()
            released = progressed.is_set()
        finally:
            sys.setswitchinterval(previous_interval)
            thread.join(timeout=1.0)
        assert released

    assert_releases_gil(lambda: index.search(query, k=10, ef=10))
    assert_releases_gil(lambda: index.search_exact(query, k=10))


def _build_small_ivf(dtype=np.float32, duplicate_tag=False):
    vectors = np.array(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]],
        dtype=dtype,
    )
    rows = [[0, 0] if duplicate_tag else [0], [0], [1], [1]]
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    indices = []
    for i, tags in enumerate(rows):
        indices.extend(tags)
        indptr[i + 1] = len(indices)
    index = prism_ann.IvfIndex(n_clusters=2, kmeans_iters=2)
    index.build(vectors, indptr, np.asarray(indices, dtype=np.int32), 2)
    return index, vectors


def test_ivf_empty_filter_searches_clusters_and_deduplicates_metadata():
    index, vectors = _build_small_ivf(duplicate_tag=True)
    ids = index.search(
        vectors[:1],
        np.array([0, 0], dtype=np.int64),
        np.array([], dtype=np.int32),
        k=4,
        ef=1,
        nprobe=2,
    )
    valid = ids[0][ids[0] != np.iinfo(np.uint32).max]
    assert set(valid.tolist()) == {0, 1, 2, 3}
    assert len(valid) == len(set(valid.tolist()))


def test_ivf_binary_rerank_uses_codes_aligned_after_internal_reorder():
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(12, 64)).astype(np.float32)

    # Alternating tags force the one-cluster tag-affinity sort to move original
    # ID 0, while an exact query gives its binary code a unique zero distance.
    rows = [[1] if point % 2 == 0 else [0] for point in range(len(vectors))]
    indptr = np.arange(len(rows) + 1, dtype=np.int64)
    indices = np.asarray([tags[0] for tags in rows], dtype=np.int32)
    index = prism_ann.IvfIndex(n_clusters=1, kmeans_iters=1)
    index.build(vectors, indptr, indices, 2)

    ids = index.search(
        vectors[[0]],
        np.array([0, 0], dtype=np.int64),
        np.array([], dtype=np.int32),
        k=4,
        ef=4,
        nprobe=1,
        binary_rerank=1,
    )
    valid = ids[0][ids[0] != np.iinfo(np.uint32).max]

    # 12 candidates exceed the rerank budget (binary_rerank * ef == 4), so
    # this exercises the binary prefilter rather than the full-scan branch.
    assert valid[0] == 0
    assert len(valid) == 4
    assert len(valid) == len(set(valid.tolist()))
    assert np.all(valid < len(vectors))


def test_ivf_rejects_extreme_finite_float32_vectors_and_queries():
    extreme = np.finfo(np.float32).max
    index = prism_ann.IvfIndex(n_clusters=1, kmeans_iters=1)
    with pytest.raises(ValueError, match="absolute magnitude"):
        index.build(
            np.full((1, 2), extreme, dtype=np.float32),
            np.array([0, 0], dtype=np.int64),
            np.array([], dtype=np.int32),
            1,
        )

    index, _ = _build_small_ivf(np.float32)
    with pytest.raises(ValueError, match="absolute magnitude"):
        index.search(
            np.full((1, 2), extreme, dtype=np.float32),
            np.array([0, 0], dtype=np.int64),
            np.array([], dtype=np.int32),
            k=1,
            ef=1,
            nprobe=1,
            binary_rerank=1,
        )


def test_ivf_rejects_dtype_mismatch_and_invalid_search_csr():
    index, vectors = _build_small_ivf(np.float32)
    empty_indices = np.array([], dtype=np.int32)

    with pytest.raises(ValueError, match="dtype"):
        index.search(
            vectors[:1].astype(np.uint8),
            np.array([0, 0], dtype=np.int64),
            empty_indices,
            nprobe=2,
        )

    with pytest.raises(ValueError, match="indptr length"):
        index.search(
            vectors[:1],
            np.array([0], dtype=np.int64),
            empty_indices,
            nprobe=2,
        )

    with pytest.raises(ValueError, match="nprobe"):
        index.search(
            vectors[:1],
            np.array([0, 0], dtype=np.int64),
            empty_indices,
            nprobe=0,
        )


def test_ivf_rejects_invalid_build_csr():
    index = prism_ann.IvfIndex(n_clusters=1, kmeans_iters=1)
    vectors = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="monotone"):
        index.build(
            vectors,
            np.array([0, 2, 1], dtype=np.int64),
            np.array([0], dtype=np.int32),
            1,
        )


@pytest.mark.parametrize(
    ("indptr", "indices", "message"),
    [
        ([-1, 0, 0], [], "start at zero"),
        ([0, 0, 2], [0], "within indices"),
        ([0, 0, 0], [0], "end at the indices length"),
        ([0, 1, 1], [-1], "outside the valid range"),
        ([0, 1, 1], [1], "outside the valid range"),
    ],
)
def test_ivf_rejects_all_malformed_build_csr_shapes(indptr, indices, message):
    index = prism_ann.IvfIndex(n_clusters=1, kmeans_iters=1)
    vectors = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match=message):
        index.build(
            vectors,
            np.asarray(indptr, dtype=np.int64),
            np.asarray(indices, dtype=np.int32),
            1,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.uint8])
def test_ivf_supports_documented_dtypes_and_conjunctive_tags(dtype):
    vectors = np.array(
        [[0, 0], [1, 0], [50, 50], [51, 50]],
        dtype=dtype,
    )
    rows = [[0, 1], [0], [1], [0, 1]]
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    indices = []
    for row, tags in enumerate(rows):
        indices.extend(tags)
        indptr[row + 1] = len(indices)

    index = prism_ann.IvfIndex(n_clusters=2, kmeans_iters=2)
    index.build(vectors, indptr, np.asarray(indices, dtype=np.int32), 3)
    ids = index.search(
        vectors[:2],
        np.array([0, 2, 3], dtype=np.int64),
        np.array([0, 1, 2], dtype=np.int32),
        k=4,
        ef=1,
        nprobe=2,
    )
    valid = ids[0][ids[0] != np.iinfo(np.uint32).max]
    assert set(valid.tolist()) == {0, 3}
    assert np.all(ids[1] == np.iinfo(np.uint32).max)
