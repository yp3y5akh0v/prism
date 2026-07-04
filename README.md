# PRISM

Filtered vector search library with two algorithms:

- **Index** — partitions vectors by attribute values, builds local Vamana graphs per partition, and connects them with cross-partition edges and an expander overlay. For datasets with discrete label-based filters.
- **IvfIndex** — two-level inverted index (K-means clusters × tag posting lists) with multi-query cell batching (MQCB). For large-scale datasets with high tag cardinality.

Both support L2 distance, SQ8 quantization, and binary Hamming pre-filtering.

## Build

Requires Rust 1.80+.

```
cargo build --release -p prism-ann
```

## Python

Requires Python 3.8+ and [maturin](https://github.com/PyO3/maturin).

```
pip install maturin
cd crates/prism-python
maturin develop --release
```

### Index (graph-based)

For datasets with a fixed set of discrete attributes per vector.

```python
import numpy as np
import prism_ann

idx = prism_ann.Index(dim=128, num_attributes=2)

vectors = np.random.rand(10000, 128).astype(np.float32)
attributes = np.random.randint(0, 50, size=(10000, 2)).astype(np.uint32)
idx.add(vectors, attributes)
idx.build()

# Single query with filter: attr_0 must be 3 or 7
query = np.random.rand(128).astype(np.float32)
ids, dists = idx.search(query, k=10, filter={"attr_0": [3, 7]})

# Batch search (parallel, releases GIL)
queries = np.random.rand(100, 128).astype(np.float32)
filters = [{"attr_0": [i % 50]} for i in range(100)]
ids_batch, dists_batch = idx.batch_search(queries, k=10, ef=200, filters=filters)
```

### IvfIndex (IVF + tag posting lists)

For large-scale tagged datasets with uint8 vectors. Metadata is passed as a scipy-style CSR matrix (indptr + indices).

```python
import numpy as np
import prism_ann

n, dim, n_tags = 1_000_000, 192, 50000

idx = prism_ann.IvfIndex(n_clusters=4000, kmeans_iters=5)

vectors = np.random.randint(0, 256, (n, dim), dtype=np.uint8)

# CSR metadata: each vector has 1-5 tags
tags_per_vec = np.random.randint(1, 6, n)
indptr = np.zeros(n + 1, dtype=np.int64)
indptr[1:] = np.cumsum(tags_per_vec)
indices = np.random.randint(0, n_tags, int(indptr[-1]), dtype=np.int32)

idx.build(vectors, indptr, indices, n_tags)

# Batch search: each query requires specific tags
nq = 1000
queries = np.random.randint(0, 256, (nq, dim), dtype=np.uint8)
q_tags = np.random.randint(0, n_tags, nq, dtype=np.int32)
q_indptr = np.arange(nq + 1, dtype=np.int64)

result_ids = idx.search(queries, q_indptr, q_tags, k=10, ef=50, nprobe=60)
```

## License

Licensed under either of Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
or MIT license ([LICENSE-MIT](LICENSE-MIT)) at your option.
