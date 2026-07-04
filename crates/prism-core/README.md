# PRISM (`prism-ann`)

Filtered vector search: find the nearest vectors that also satisfy attribute
constraints, without scanning the whole set.

Two index types:

- **`PrismIndex`** - partitions vectors by attribute value, builds a local Vamana
  graph per partition, and connects them with cross-partition edges plus an
  expander overlay. For datasets with discrete label-based filters.
- **`ivf::IvfIndex`** - a two-level inverted index (K-means clusters x tag posting
  lists) with multi-query cell batching (MQCB). For large-scale datasets with high
  tag cardinality.

Both support L2 distance, SQ8 scalar quantization, and binary Hamming pre-filtering.

Public API: `PrismConfig`, `PrismIndex`, `PointStore`, `Filter`, `Metric`, plus the
`ivf`, `distance`, `quantize`, and `binary` modules. See the
[repository](https://github.com/yp3y5akh0v/prism) for usage examples and the
Python bindings (`prism-python`).

## License

Licensed under either of Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
or MIT license ([LICENSE-MIT](LICENSE-MIT)) at your option.
