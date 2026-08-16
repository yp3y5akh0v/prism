<h1 align="center">PRISM</h1>

<p align="center">
  <a href="https://crates.io/crates/prism-ann"><img src="https://badgen.net/crates/v/prism-ann" alt="crates.io"></a>
  <a href="https://github.com/yp3y5akh0v/prism/actions/workflows/ci.yml"><img src="https://github.com/yp3y5akh0v/prism/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/yp3y5akh0v/prism#rust"><img src="https://img.shields.io/badge/rust-1.80%2B-orange" alt="Rust 1.80+"></a>
  <a href="https://github.com/yp3y5akh0v/prism#python"><img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python 3.9+"></a>
  <a href="https://github.com/yp3y5akh0v/prism/blob/HEAD/LICENSE-APACHE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
</p>

Filtered vector search for Rust and Python. Filters are enforced exactly: every
returned record satisfies the predicate.

Add records, then build. The index is immutable once built.

## Filter model

`PrismIndex` gives each record one `u32` value per attribute. A filter is
`IN (...)` within an attribute, combined across attributes with `AND`, so
`attr_0 IN (3, 7) AND attr_2 IN (9)` selects on both.

`IvfIndex` uses a multi-valued tag model in which query tags are conjunctive.

`PrismIndex` materializes one cell per populated attribute combination, so
attribute cardinality drives index size.

## Python

Requires Python 3.9+ and Rust 1.83+. Building from source requires
[maturin](https://github.com/PyO3/maturin).

```console
pip install maturin
maturin develop --release
```

```python
import numpy as np
import prism_ann

index = prism_ann.Index(dim=128, num_attributes=2, metric="cosine")
vectors = np.random.default_rng(1).random((10_000, 128), dtype=np.float32)
attributes = np.random.default_rng(2).integers(
    0, 50, size=(10_000, 2), dtype=np.uint32
)
index.add(vectors, attributes)
index.build()

query = np.random.default_rng(3).random(128, dtype=np.float32)
predicate = {"attr_0": [3, 7]}

ids, distances = index.search(query, k=10, ef=200, filter=predicate)
exact_ids, exact_distances = index.search_exact(query, k=10, filter=predicate)
```

`batch_search` takes queries as a 2D array and returns `(nq, k)` arrays. Pass
`filter=` for one filter applied to every query, or `filters=` for a list of one
filter per query (`None` in a slot means no filter). Rows with fewer than `k`
eligible records are padded with the maximum `uint32` value as the ID and
infinite distances.

### Cluster x tag index

`IvfIndex` accepts `uint8` or `float32` L2 vectors and scipy-style CSR tag
metadata. Query dtype matches build dtype. An empty CSR row is an unfiltered
query; nonempty rows require every listed tag.

```python
n, dim, n_tags = 100_000, 192, 10_000
rng = np.random.default_rng(4)
vectors = rng.integers(0, 256, (n, dim), dtype=np.uint8)

indptr = np.arange(n + 1, dtype=np.int64)
indices = rng.integers(0, n_tags, n, dtype=np.int32)

index = prism_ann.IvfIndex(n_clusters=400, kmeans_iters=5)
index.build(vectors, indptr, indices, n_tags)

queries = vectors[:100]
query_indptr = np.arange(101, dtype=np.int64)
query_tags = rng.integers(0, n_tags, 100, dtype=np.int32)
result_ids = index.search(
    queries, query_indptr, query_tags, k=10, ef=50, nprobe=60
)
```

## Rust

Requires Rust 1.80+ for the core `prism-ann` crate.

```console
cargo build --release -p prism-ann
cargo test -p prism-ann --all-targets
```

```rust
use prism_ann::{Filter, PointStore, PrismConfig, PrismIndex, PrismResult};

fn search(
    vectors: Vec<f32>,
    attributes: Vec<Vec<u32>>,
    query: &[f32],
) -> PrismResult<()> {
    let store = PointStore::from_parts(vectors, 128, attributes)?;
    let index = PrismIndex::build(store, PrismConfig::default())?;
    let outcome = index.search(query, &Filter::eq(0, 3), 10, 200)?;

    println!("results: {}", outcome.results.len());
    println!("exact rescue: {}", outcome.diagnostics.used_exact_fallback);
    Ok(())
}
```

Errors are returned as `PrismError` through the `PrismResult<T>` alias. Rust
results use internal reordered IDs; call `original_id` to recover
insertion-order IDs. Python maps them automatically.

## Search behavior

Metrics are `"l2"`, `"cosine"`, and `"ip"` (alias `"inner_product"`). L2 is
reported as squared L2. Inner product uses an exact filtered scan.

`search` routes on filter selectivity, scanning or traversing the graph as the
configured thresholds direct. It returns `min(k, eligible)` records: `ef` is
promoted to at least `k`, and an exact eligible scan backs an underfilled
approximate result. `search_exact` takes the exact path unconditionally.

`build_seed` fixes construction: the same data, configuration and seed produce
the same graph regardless of thread count.

Graph construction and routing thresholds are configured on `PrismConfig`.

## License

[Apache-2.0](https://github.com/yp3y5akh0v/prism/blob/HEAD/LICENSE-APACHE)
