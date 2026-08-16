# PRISM (`prism-ann`)

Filtered vector search for conjunctive equality/IN predicates over fixed scalar
`u32` attributes.

`PrismIndex` enforces predicates exactly and routes L2/cosine queries among
eligible-set scans, compatible local graphs, and predicate-aware whole-index
graph traversal according to selectivity and exact eligible cardinality.
Vector ranking is approximate outside the exhaustive path.
`search_exact` provides an exact filtered top-k baseline, and incomplete
approximate results fall back to the eligible scan so the API does not silently
underfill. Inner-product queries currently use that correctness-first exact
path.

The fallible Rust entry points return `PrismResult<T>`. In particular,
`PrismIndex::search` returns `PrismResult<SearchOutcome>`, whose `results` are
paired with the selectivity regime, physical primary execution, primary/final
counts, exact-fallback flag, separate successful-call total and
primary/fallback durations, and completeness status.
`batch_search` returns one such outcome per query; `search_exact` returns
`PrismResult<Vec<SearchResult>>`.

`ivf::IvfIndex` provides L2 search over geometric clusters x conjunctive tag
posting lists. It owns binary prefilter codes built after internal vector
reordering, and its tag-to-cluster lookup is sparse in the declared vocabulary
size. The codes are stored even when a query uses `binary_rerank=0`.
`IvfIndex` does not currently implement inner product or cosine.

The `PrismIndex` filter model does not include ranges, NOT, cross-attribute OR,
NULL/missing semantics, multi-valued record attributes, text, or geo predicates.
`ivf::IvfIndex` separately supports multi-valued record tags with conjunctive
query semantics.

Primary root exports: `PrismConfig`, `PrismIndex`, `PointStore`, `Filter`,
`Metric`, `PrismError`, `PrismResult`, `SearchPlan`, `SearchRegime`,
`SearchDiagnostics`, `SearchExecution`, `SearchOutcome`, and `SearchResult`,
plus the `ivf`, `distance`, `quantize`, and `binary` modules. See the [repository
README](https://github.com/yp3y5akh0v/prism) for examples, guarantees,
limitations, and benchmark instructions.

Version 0.3 is a breaking API change. Direct 0.2 callers must adopt the
fallible construction/search results and immutable getters, and rebuild
caller-managed persisted 0.2 index state through the validated constructors.

Invariant-bearing state in `PrismIndex`, `PointStore`, `PartitionTree`, `Graph`,
the quantized stores, and `IvfIndex` is exposed only through immutable getters.
Formerly public fields are now crate-private; component reconstruction uses
validated fallible constructors where provided. A complete persisted index is
reassembled with `PrismIndex::from_parts`, which additionally validates the
cross-component identifier, partition, cosine-normalization, quantized/binary
alignment, graph, medoid, and reachability invariants, including a canonical
zero-edge graph for exact/scan-only configurations. The completed index retains
one sorted graph CSR; local execution derives same-cell row slices from it, and
no separate local graph is accepted or exposed. SQ8 and binary presence is
likewise derived from the reachable configured paths; unused accelerator stores
must be empty. It is not a versioned file format or serializer. To prove
accelerator alignment it transiently rebuilds SQ8 and enabled binary state for
comparison, so reconstruction retains the graph but includes `O(n * dim)`
cold-load validation work and temporary comparison copies of the reachable
accelerators. It is not a zero-copy loader. Point/graph ID accessors return
`Option` for out-of-range IDs instead of panicking.
Public `PartitionTree` filtering/counting/collection/cell-lookup methods are
checked and return `PrismResult` (plus `Option` for a missing point lookup).

High-level f32 construction and query paths reject nonfinite coordinates and
extreme finite magnitudes outside
`abs(x) <= sqrt(f32::MAX / (8 * dim))`. Exact cosine/inner-product
accumulation, cosine normalization, cell/global centroid accumulation for
medoid selection, SQ8 range/quantization/reconstruction validation, and the
binary Walsh-Hadamard workspace use f64 internally while public distances remain
f32. Full-query-to-SQ8 candidate distance accumulates
in checked-domain f32, with runtime AVX2/FMA dispatch on supported x86-64 CPUs
and a scalar f32 fallback. The `force-scalar-sq8` feature exists only for
reference/portability A/B measurement; it is not the recommended production
configuration and does not change `graph_expansion`.

Low-level distance primitives still require equal-length slices and bypass the
high-level numeric check; mismatched lengths can panic and unsupported
magnitudes can yield a nonfinite f32 result.

`PrismConfig::graph_expansion` is the graph-traversal quality/work multiplier
and defaults to `3`: traversal uses the candidate budget
`min(searched_population, ef * graph_expansion)`. Local-cell search narrows the
expanded frontier to `ef` candidates before exact original-vector reranking;
global/MID traversal retains an expanded matching heap through exact reranking
before returning `k`.
`scan_threshold` is the inclusive general limit on total eligible points for a
query-wide HIGH/MID scan. At or below it, the eligible population is
exact-scored unless the configured binary prefilter reduces one global
candidate budget. Its default is `20_000`; `0` disables this general scan route.
When the complete index fits a nonzero `scan_threshold`, construction stores a
canonical zero-edge graph because no query can require graph work.
`multi_cell_scan_threshold` is the supplemental inclusive limit selecting that
same exact or binary-prefilter scan route for a filter over a proper subset of
more than one populated cell. It defaults to `500_000`, does not apply to
unfiltered/all-cell or one-cell queries, and does not by itself suppress graph
construction; `0` disables it. Both thresholds must be `0` to force graph work
for a proper multi-cell HIGH/MID query. Above all applicable thresholds,
HIGH/MID use a predicate-aware whole-index graph when available or merge
compatible local graphs. LOW and inner product remain
exact independently. Both defaults are informed by matched local crossovers,
not universal hardware/workload crossover or recall guarantees.
`SearchDiagnostics::primary_execution` reports exact scan, binary-prefilter
scan, local graph, global graph, or no-work execution independently of the
selectivity regime.
`build_seed` makes randomized graph construction reproducible for fixed input
and configuration, including across the tested one- and four-thread Rayon
builds. This is not a cross-version byte-format guarantee.
`m_random=0` disables the whole-index random-permutation overlay, not the seeded
random order used by local Vamana construction.

When graphs are required, large Vamana cells include a deterministic local
spanning chain, and builds that require the global graph connect cell medoids
with a deterministic spanning chain. `m_local` is a Vamana target/pruning
degree; the chain can add up to two neighbors per point beyond that target. The
0.3 defaults are `m_local=48` and construction `beam_width=128`; neither is a
recall guarantee.
Local Vamana construction uses the
[DiskANN two-pass structure](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf):
RobustPrune receives the complete GreedySearch expanded set, pass one uses
`alpha=1`, and pass two uses the finite `vamana_alpha >= 1`. Construction uses
full-precision original-vector distances rather than SQ8 codes; squared
L2/cosine pruning applies `alpha^2`.
This is an implementation-contract statement, not benchmark evidence. Graph
parameters are representability-checked and population-capped. Greedy
cross-cell construction reuses one target-cell table per source cell.
`cross_cell_exact_ranking_limit` defaults to `4096`: at or below it all foreign
medoids are exhaustively enumerated and SQ8-distance-ranked; above it, or at
`0`, a deterministic catalog-stratified pool bounded by `beam_width` and the
target count is ranked instead. The public hard limit is `8192`. "Exact" here
describes medoid enumeration only, not exact neighbor search or a recall
guarantee; bounded mode is a construction-resource/quality tradeoff.
When greedy cross-cell construction is active, let `k` denote the schema's
attribute count, not search top-k. Its effective strength is
`t_eff = min(config.t, k)`. Covering rejects
`C(k, t_eff) = C(k, min(t, k)) > MAX_COVERING_SUBSETS` (`4096`) rather than
materializing or sampling an unbounded projection set.

Setting `sigma_high == sigma_low` disables MID and skips whole-index greedy,
overlay, and medoid-backbone construction. L2/cosine local graphs are built
only when the index can require graph work above every applicable threshold.
Such HIGH queries enter every compatible local graph; at or below an applicable
total-eligible threshold, the route is a scan. Benchmark evidence
must separate the public threshold-routed execution, forced local-graph quality,
and exact f32 oracle; forced graph work is useful for testing `graph_expansion`
but is not the public planner's latency or throughput. None is universal-quality
evidence. The
[repository README](https://github.com/yp3y5akh0v/prism) links the benchmark
methodology and current evidence status.

L2 distances are squared L2. `SearchResult.id` is an internal reordered ID;
use `PrismIndex::original_id` when the insertion-order ID is required.

## License

Licensed under Apache-2.0 (`LICENSE-APACHE`).
