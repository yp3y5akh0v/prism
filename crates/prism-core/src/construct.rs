use super::binary::BinaryStore;
use super::distance::{self, Metric};
use super::error::{PrismError, PrismResult};
use super::graph::{AdjBuilder, Graph};
use super::partition::PartitionTree;
use super::point::PointStore;
use super::quantize::SQ8Store;

use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;

/// Maximum number of attribute projections materialized for greedy covering.
///
/// PRISM evaluates every `t`-attribute subset rather than silently sampling
/// them. Configurations whose binomial count exceeds this explicit resource
/// bound are rejected by [`PrismIndex::build`].
pub const MAX_COVERING_SUBSETS: usize = 4_096;

/// Hard resource bound for exact all-pairs cross-cell medoid ranking.
///
/// The work is quadratic in the number of populated cells. Larger catalogs
/// must use the deterministic bounded candidate-pool strategy instead. Here
/// "exact" means exhaustive foreign-cell enumeration; medoids are still ranked
/// with PRISM's SQ8 candidate distance rather than exact f32 distance.
pub const MAX_CROSS_CELL_EXACT_RANKING_LIMIT: usize = 8_192;

// Graph IDs are u32 and Vec capacity tops out at isize::MAX bytes, so anything
// above this bound must be rejected before arithmetic or allocation uses it.
const MAX_GRAPH_PARAMETER: usize = if (u32::MAX as usize) < (isize::MAX as usize) {
    u32::MAX as usize
} else {
    isize::MAX as usize
};

/// Configuration for PRISM index construction.
#[derive(Clone, Debug)]
pub struct PrismConfig {
    /// Local Vamana pruning-degree target. Large-cell connectivity adds a
    /// deterministic chain that can contribute up to two additional neighbors.
    /// The default is 48.
    pub m_local: usize,
    /// Greedy cross-partition neighbor-selection target.
    pub m_greedy: usize,
    /// Whole-index random-permutation overlay target (`0` disables it;
    /// otherwise even and at least 4). Deduplication and permutation fixed
    /// points can make the realized added degree smaller.
    pub m_random: usize,
    /// Covering strength for attribute-diverse selection.
    pub t: usize,
    /// Proximity-diversity tradeoff for cross-neighbor selection (0 = pure diversity).
    pub alpha: f32,
    /// Vamana pruning parameter. Must be finite and at least 1; PRISM defaults
    /// to 1.0.
    pub vamana_alpha: f32,
    /// Vamana construction search-list width `L`, also used to bound greedy
    /// cross-cell candidate discovery. It is normally at least `m_local`;
    /// larger values trade build work for graph quality. The default is 128.
    pub beam_width: usize,
    /// Maximum populated-cell count for exact all-pairs medoid ranking during
    /// greedy cross-cell construction.
    ///
    /// At or below this inclusive limit, all foreign cell medoids are ranked
    /// once per source cell and reused by every point in that cell. Above it,
    /// construction scores a deterministic stratified pool of at most
    /// `max(beam_width, 4 * target_count)` foreign cells, where
    /// `target_count = min(c - 1, beam_width, max(m_greedy, 4))`, then retains
    /// those nearest target cells. `0` always selects the bounded strategy.
    /// This is an explicit build-work/quality knob; it does not change the greedy phase's
    /// `m_greedy` selection target (other graph phases add separate edges).
    /// "Exact" refers to enumerating every foreign cell, not to the SQ8 medoid
    /// distance or final neighbor ranking.
    pub cross_cell_exact_ranking_limit: usize,
    /// Distance metric.
    pub metric: Metric,
    /// Selectivity threshold for HIGH regime.
    pub sigma_high: f32,
    /// Selectivity threshold for LOW regime.
    pub sigma_low: f32,
    /// Bridge budget multiplier for MID regime.
    pub beta: f32,
    /// Search pruning tolerance for filtered queries.
    pub epsilon: f32,
    /// Binary pre-filter rerank factor. Top `binary_rerank * ef` Hamming candidates
    /// are reranked with SQ8. `0` disables it; inner product always bypasses it.
    pub binary_rerank: usize,
    /// Maximum total eligible-point count for query-wide scan execution.
    ///
    /// HIGH and MID queries at or below this inclusive threshold score all
    /// eligible points exactly unless the optional binary prefilter applies;
    /// populations above every applicable scan threshold use graph traversal.
    /// LOW and inner-product policies remain exact independently of this
    /// setting. `0` disables this
    /// general scan route; a proper multi-cell filtered subset can still use
    /// [`Self::multi_cell_scan_threshold`], so both thresholds must be `0` to
    /// force graph traversal for that case. When the whole index fits this
    /// threshold, graph construction is skipped because no query can need it.
    /// The default is 20,000, informed by a matched local crossover
    /// measurement; it remains an
    /// explicit workload knob, not a universal crossover or recall guarantee.
    pub scan_threshold: usize,
    /// Maximum total eligible-point count for scan execution when a filter
    /// selects a proper subset of more than one partition cell.
    ///
    /// This inclusive threshold supplements [`Self::scan_threshold`] for
    /// fragmented filtered populations, where a query-wide exact scan (or the
    /// configured binary-prefilter scan) can be cheaper than predicate-aware
    /// graph traversal. It does not apply to unfiltered/all-cell queries or
    /// filters selecting exactly one cell, and therefore does not by itself
    /// suppress whole-index graph construction. `0` disables this additional
    /// route. The default is 500,000, informed by matched SIFT1M filtered
    /// crossover with the default binary prefilter disabled; it remains an
    /// explicit workload knob, not a universal crossover or recall guarantee.
    pub multi_cell_scan_threshold: usize,
    /// Quality/work multiplier for local Vamana and whole-index graph traversal.
    /// The candidate budget is capped at
    /// `min(searched_population, graph_expansion * ef)`. Local-cell search
    /// narrows that frontier back to `ef` before exact reranking; global/MID
    /// search can retain the expanded matching set. Must be at least 1. No
    /// setting guarantees a fixed recall or latency ratio.
    pub graph_expansion: usize,
    /// Seed for all randomized construction phases. Per-cell streams are
    /// derived from this value, so Rayon scheduling does not affect the graph.
    pub build_seed: u64,
}

impl Default for PrismConfig {
    fn default() -> Self {
        Self {
            m_local: 48,
            m_greedy: 12,
            m_random: 4,
            t: 2,
            alpha: 1.0,
            vamana_alpha: 1.0,
            beam_width: 128,
            cross_cell_exact_ranking_limit: 4_096,
            metric: Metric::L2,
            sigma_high: 0.10,
            sigma_low: 0.001,
            beta: 3.0,
            epsilon: 0.2,
            binary_rerank: 0,
            scan_threshold: 20_000,
            multi_cell_scan_threshold: 500_000,
            graph_expansion: 3,
            build_seed: 0x5052_4953_4d41_4e4e,
        }
    }
}

impl PrismConfig {
    /// Validate construction and routing parameters.
    pub fn validate(&self) -> PrismResult<()> {
        if self.m_local == 0 {
            return Err(PrismError::InvalidInput(
                "m_local must be greater than zero".into(),
            ));
        }
        if self.beam_width == 0 {
            return Err(PrismError::InvalidInput(
                "beam_width must be greater than zero".into(),
            ));
        }
        if self.m_random != 0 && (self.m_random < 4 || self.m_random % 2 != 0) {
            return Err(PrismError::InvalidInput(
                "m_random must be 0 or an even value of at least 4".into(),
            ));
        }
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err(PrismError::InvalidInput(
                "alpha must be finite and nonnegative".into(),
            ));
        }
        if !self.vamana_alpha.is_finite() || self.vamana_alpha < 1.0 {
            return Err(PrismError::InvalidInput(
                "vamana_alpha must be finite and at least 1".into(),
            ));
        }
        if !self.sigma_low.is_finite()
            || !self.sigma_high.is_finite()
            || !(0.0..=1.0).contains(&self.sigma_low)
            || !(0.0..=1.0).contains(&self.sigma_high)
            || self.sigma_low > self.sigma_high
        {
            return Err(PrismError::InvalidInput(
                "selectivity thresholds must satisfy 0 <= sigma_low <= sigma_high <= 1".into(),
            ));
        }
        if !self.beta.is_finite() || self.beta < 0.0 {
            return Err(PrismError::InvalidInput(
                "beta must be finite and nonnegative".into(),
            ));
        }
        if !self.epsilon.is_finite() || self.epsilon < 0.0 {
            return Err(PrismError::InvalidInput(
                "epsilon must be finite and nonnegative".into(),
            ));
        }
        if self.graph_expansion == 0 {
            return Err(PrismError::InvalidInput(
                "graph_expansion must be greater than zero".into(),
            ));
        }
        if self.cross_cell_exact_ranking_limit > MAX_CROSS_CELL_EXACT_RANKING_LIMIT {
            return Err(PrismError::InvalidInput(format!(
                "cross_cell_exact_ranking_limit must be between 0 and {MAX_CROSS_CELL_EXACT_RANKING_LIMIT}"
            )));
        }
        for (name, value) in [
            ("m_local", self.m_local),
            ("m_greedy", self.m_greedy),
            ("m_random", self.m_random),
            ("beam_width", self.beam_width),
        ] {
            if value > MAX_GRAPH_PARAMETER {
                return Err(PrismError::Overflow(format!(
                    "{name}={value} exceeds the maximum representable graph parameter {MAX_GRAPH_PARAMETER}"
                )));
            }
        }
        Ok(())
    }

    /// Whether the configuration enables whole-index routing in principle.
    /// Concrete construction additionally requires a population above the scan
    /// threshold and more than one partition cell; see
    /// [`Self::builds_global_graph`].
    pub(crate) fn uses_global_graph(&self) -> bool {
        self.metric != Metric::InnerProduct
            && self.sigma_high > self.sigma_low
            && (self.m_greedy > 0 || self.m_random > 0)
    }

    /// Whether any HIGH/MID query over this population can require graph work.
    pub(crate) fn builds_local_graph(&self, population: usize) -> bool {
        self.metric != Metric::InnerProduct
            && (self.scan_threshold == 0 || population > self.scan_threshold)
    }

    /// Whether construction must add cross-cell connectivity for the physical
    /// routes possible in this concrete index.
    pub(crate) fn builds_global_graph(&self, population: usize, cells: usize) -> bool {
        self.builds_local_graph(population) && cells > 1 && self.uses_global_graph()
    }

    /// Whether any reachable primary path needs scalar-quantized candidates.
    pub(crate) fn builds_sq8(&self, population: usize) -> bool {
        self.metric != Metric::InnerProduct
            && (self.builds_local_graph(population) || self.binary_rerank > 0)
    }
}

fn has_multiple_attribute_tuples(attrs: &[Vec<u32>], point_count: usize) -> bool {
    if point_count <= 1 {
        return false;
    }
    attrs.iter().any(|attribute| {
        let first = attribute[0];
        attribute[1..].iter().any(|&value| value != first)
    })
}

fn validate_build_configuration(
    config: &PrismConfig,
    num_attributes: usize,
    population: usize,
    multiple_cells: bool,
) -> PrismResult<()> {
    if population == 0 {
        return Err(PrismError::InvalidInput(
            "cannot build index from empty point store".into(),
        ));
    }
    config.validate()?;

    // Covering subsets are materialized only when greedy cross-cell construction
    // is reachable, so validate that condition before a consuming build starts.
    let populated_cells = if multiple_cells { 2 } else { 1 };
    if config.m_greedy > 0 && config.builds_global_graph(population, populated_cells) {
        checked_covering_subset_count(num_attributes, config.t.min(num_attributes))?;
    }
    Ok(())
}

/// Derived exact cosine row norms. This is intentionally not a persisted or
/// public component: it is cheap to reconstruct from the canonical point store
/// and keeping it derived prevents stale-cache/index combinations.
struct CosineNormCache {
    values: Vec<f64>,
}

impl CosineNormCache {
    fn empty() -> Self {
        Self { values: Vec::new() }
    }

    #[inline]
    fn len(&self) -> usize {
        self.values.len()
    }

    #[inline]
    fn get(&self, point: u32) -> f64 {
        self.values[point as usize]
    }

    fn zeroize(&mut self) {
        use zeroize::Zeroize;
        self.values.as_mut_slice().zeroize();
    }
}

impl Drop for CosineNormCache {
    fn drop(&mut self) {
        self.zeroize();
    }
}

fn derive_cosine_norm_cache(store: &PointStore, metric: Metric) -> PrismResult<CosineNormCache> {
    if metric != Metric::Cosine {
        return Ok(CosineNormCache::empty());
    }

    let n = store.len();
    // Own the allocation through the zeroizing wrapper before deriving any value,
    // so unwinding through `?` on a later malformed row still zeroizes the partial cache.
    let mut cache = CosineNormCache::empty();
    cache.values.try_reserve_exact(n).map_err(|error| {
        PrismError::Overflow(format!(
            "cannot allocate exact cosine norm cache for {n} points: {error}"
        ))
    })?;
    for point in 0..n as u32 {
        let norm_sq = distance::cosine_norm_squared(store.vector_unchecked(point));
        if norm_sq != 0.0 && (norm_sq - 1.0).abs() > 1.0e-4 {
            return Err(PrismError::InvalidFormat(format!(
                "cosine point {point} must be zero or unit-normalized, got squared norm {norm_sq}"
            )));
        }
        cache.values.push(norm_sq.sqrt());
    }
    Ok(cache)
}

/// The complete PRISM index.
pub struct PrismIndex {
    pub(crate) store: PointStore,
    pub(crate) tree: PartitionTree,
    pub(crate) graph: Graph,
    pub(crate) medoids: Vec<u32>,
    pub(crate) global_medoid: u32,
    /// Reverse mapping: point_id -> cell index.
    pub(crate) point_cell: Vec<u32>,
    /// Maps internal ID -> original ID.
    pub(crate) original_ids: Vec<u32>,
    /// Scalar-quantized vectors for distance computation.
    pub(crate) sq8: SQ8Store,
    /// Binary codes for Hamming pre-filter.
    pub(crate) binary: BinaryStore,
    /// Exact f64 norms of build-normalized cosine rows, indexed by internal ID.
    /// Empty for non-cosine indexes and derived rather than persisted.
    cosine_norms: CosineNormCache,
    pub(crate) config: PrismConfig,
}

impl PrismIndex {
    #[inline]
    pub(crate) fn cached_cosine_norm(&self, point: u32) -> f64 {
        self.cosine_norms.get(point)
    }

    #[inline]
    pub(crate) fn cached_cosine_norm_count(&self) -> usize {
        self.cosine_norms.len()
    }

    /// Validate raw build input without taking ownership of its vector or
    /// attribute allocations.
    ///
    /// This covers deterministic shape, numeric-domain, configuration, and
    /// schema-dependent covering-subset failures. A caller can run it before
    /// moving large staged buffers into [`Self::build`]. It does not reserve
    /// graph/accelerator memory or promise that later resource allocation will
    /// succeed.
    pub fn validate_build_input(
        vectors: &[f32],
        dim: usize,
        attrs: &[Vec<u32>],
        config: &PrismConfig,
    ) -> PrismResult<()> {
        let population = PointStore::validate_parts(vectors, dim, attrs)?;
        validate_build_configuration(
            config,
            attrs.len(),
            population,
            has_multiple_attribute_tuples(attrs, population),
        )
    }

    /// Immutable point storage used by this index.
    pub fn store(&self) -> &PointStore {
        &self.store
    }

    /// Immutable attribute partition tree.
    pub fn tree(&self) -> &PartitionTree {
        &self.tree
    }

    /// The retained graph CSR. It contains in-cell edges whenever graph work is
    /// enabled and adds cross-cell edges only when whole-index construction is
    /// enabled. Exact/scan-only indexes retain an n-node, zero-edge CSR.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Per-cell graph entries in tree-cell order. Exact-only indexes use each
    /// cell's deterministic first point because no graph entry is consumed.
    pub fn medoids(&self) -> &[u32] {
        &self.medoids
    }

    /// Whole-index graph entry. Indexes without global traversal use canonical
    /// point ID `0` because this value is never consumed by search.
    pub fn global_medoid(&self) -> u32 {
        self.global_medoid
    }

    /// Reverse mapping from internal point ID to tree-cell index.
    pub fn point_cell(&self) -> &[u32] {
        &self.point_cell
    }

    /// Mapping from internal reordered IDs to caller insertion-order IDs.
    pub fn original_ids(&self) -> &[u32] {
        &self.original_ids
    }

    /// Immutable scalar-quantized vector storage. Exact-only indexes expose a
    /// canonical empty store with the index dimension.
    pub fn sq8(&self) -> &SQ8Store {
        &self.sq8
    }

    /// Immutable binary-code storage.
    pub fn binary(&self) -> &BinaryStore {
        &self.binary
    }

    /// Validated construction and search configuration.
    pub fn config(&self) -> &PrismConfig {
        &self.config
    }

    /// Reassemble a persisted index from individually validated components.
    ///
    /// Component constructors validate their own storage. This constructor
    /// additionally validates every cross-component invariant relied on by
    /// unchecked search hot paths: common shapes, partition/store agreement,
    /// identifier maps, cosine normalization, deterministic SQ8/binary
    /// alignment, medoids, canonical graph edges, contiguous in-cell slicing
    /// and reachability, cross-edge prohibition for local-only indexes, and
    /// whole-index reachability when global traversal is enabled. It
    /// deliberately does not accept a partially valid index.
    /// Accelerator alignment is checked by transiently rebuilding SQ8 and any
    /// enabled binary store for comparison, which adds `O(n * dim)` work while
    /// retaining the persisted graph structure.
    ///
    /// This is a validation boundary, not a serialized format or a proof that
    /// the supplied parts were produced by the supplied configuration. A
    /// persistence owner must bind the configuration to its stored parts and
    /// version or invalidate them when construction semantics change.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        store: PointStore,
        tree: PartitionTree,
        graph: Graph,
        medoids: Vec<u32>,
        global_medoid: u32,
        point_cell: Vec<u32>,
        original_ids: Vec<u32>,
        sq8: SQ8Store,
        binary: BinaryStore,
        config: PrismConfig,
    ) -> PrismResult<Self> {
        config.validate()?;

        let n = store.len();
        if n == 0 {
            return Err(PrismError::InvalidFormat(
                "a PRISM index cannot contain zero points".into(),
            ));
        }
        if tree.num_attributes() != store.k() {
            return Err(PrismError::InvalidFormat(format!(
                "partition attribute count {} does not match point-store attribute count {}",
                tree.num_attributes(),
                store.k()
            )));
        }
        if tree.cells().is_empty() {
            return Err(PrismError::InvalidFormat(
                "a nonempty PRISM index must contain at least one partition cell".into(),
            ));
        }
        if graph.len() != n {
            return Err(PrismError::InvalidFormat(format!(
                "graph node count {} must equal point count {n}",
                graph.len()
            )));
        }
        let expected_sq8_points = if config.builds_sq8(n) { n } else { 0 };
        if sq8.dim() != store.dim() || sq8.len() != expected_sq8_points {
            return Err(PrismError::InvalidFormat(format!(
                "SQ8 shape ({}, {}) must have dimension {} and {expected_sq8_points} point codes for this configuration",
                sq8.len(),
                sq8.dim(),
                store.dim()
            )));
        }
        if binary.dim() != store.dim() {
            return Err(PrismError::InvalidFormat(format!(
                "binary-code dimension {} must equal point-store dimension {}",
                binary.dim(),
                store.dim()
            )));
        }
        let expected_binary_points =
            if config.binary_rerank > 0 && config.metric != Metric::InnerProduct {
                n
            } else {
                0
            };
        if binary.len() != expected_binary_points {
            return Err(PrismError::InvalidFormat(format!(
                "binary-code point count {} must equal {expected_binary_points} for this configuration",
                binary.len()
            )));
        }

        let cosine_norms = derive_cosine_norm_cache(&store, config.metric)?;

        // Shape alone cannot prove these codes belong to this reordered store,
        // so rebuild and compare: a same-shape array from another permutation
        // would otherwise silently poison candidate discovery.
        {
            let expected_sq8 = if config.builds_sq8(n) {
                SQ8Store::build(&store)
            } else {
                empty_sq8(store.dim())?
            };
            if sq8.codes() != expected_sq8.codes()
                || sq8.mins() != expected_sq8.mins()
                || sq8.scales() != expected_sq8.scales()
            {
                return Err(PrismError::InvalidFormat(
                    "SQ8 codes or metadata do not match the persisted point store".into(),
                ));
            }
        }
        {
            let expected_binary =
                if config.binary_rerank > 0 && config.metric != Metric::InnerProduct {
                    BinaryStore::build(&store)?
                } else {
                    BinaryStore::empty(store.dim())?
                };
            if binary.codes() != expected_binary.codes()
                || binary.code_words() != expected_binary.code_words()
                || binary.signs() != expected_binary.signs()
                || binary.block_size() != expected_binary.block_size()
            {
                return Err(PrismError::InvalidFormat(
                    "binary codes or transform metadata do not match the persisted point store"
                        .into(),
                ));
            }
        }
        if medoids.len() != tree.cells().len() {
            return Err(PrismError::InvalidFormat(format!(
                "medoid count {} must equal partition cell count {}",
                medoids.len(),
                tree.cells().len()
            )));
        }
        if global_medoid as usize >= n {
            return Err(PrismError::InvalidFormat(format!(
                "global medoid {global_medoid} is out of range for {n} points"
            )));
        }
        if point_cell.len() != n || original_ids.len() != n {
            return Err(PrismError::InvalidFormat(format!(
                "point-cell/original-id lengths ({}/{}) must both equal point count {n}",
                point_cell.len(),
                original_ids.len()
            )));
        }

        let mut seen_original = vec![false; n];
        for (internal_id, &original_id) in original_ids.iter().enumerate() {
            let original = original_id as usize;
            if original >= n || seen_original[original] {
                return Err(PrismError::InvalidFormat(format!(
                    "original_ids must be a permutation of 0..{n}; invalid value {original_id} at internal point {internal_id}"
                )));
            }
            seen_original[original] = true;
        }

        for (cell_index, cell) in tree.cells().iter().enumerate() {
            let medoid = medoids[cell_index];
            if cell.point_ids().binary_search(&medoid).is_err() {
                return Err(PrismError::InvalidFormat(format!(
                    "medoid {medoid} does not belong to partition cell {cell_index}"
                )));
            }
            if !config.builds_local_graph(n) && medoid != cell.point_ids()[0] {
                return Err(PrismError::InvalidFormat(format!(
                    "scan-only medoid for partition cell {cell_index} must be its deterministic first point {}",
                    cell.point_ids()[0]
                )));
            }
            for &point_id in cell.point_ids() {
                let point = point_id as usize;
                if point_cell[point] as usize != cell_index {
                    return Err(PrismError::InvalidFormat(format!(
                        "point_cell maps point {point_id} to cell {}, expected {cell_index}",
                        point_cell[point]
                    )));
                }
                for (attribute, &cell_value) in cell.values().iter().enumerate() {
                    if store.attr_unchecked(point_id, attribute) != cell_value {
                        return Err(PrismError::InvalidFormat(format!(
                            "point {point_id} attribute {attribute} does not match partition cell {cell_index}"
                        )));
                    }
                }
            }
        }

        if !config.builds_global_graph(n, tree.cells().len()) && global_medoid != 0 {
            return Err(PrismError::InvalidFormat(
                "an index without global graph traversal must use deterministic global_medoid 0"
                    .into(),
            ));
        }

        validate_canonical_graph(&graph, "graph")?;
        validate_local_reachability(&graph, tree.cells(), &medoids, config.builds_local_graph(n))?;

        if !config.builds_local_graph(n) {
            if graph.num_edges() != 0 {
                return Err(PrismError::InvalidFormat(
                    "an exact/scan-only index must not persist unused graph edges".into(),
                ));
            }
        } else if config.builds_global_graph(n, tree.cells().len()) {
            validate_reachable(&graph, global_medoid, n, "global graph")?;
        } else {
            validate_no_cross_cell_edges(&graph, &point_cell)?;
        }

        Ok(Self {
            store,
            tree,
            graph,
            medoids,
            global_medoid,
            point_cell,
            original_ids,
            sq8,
            binary,
            cosine_norms,
            config,
        })
    }

    /// Build a PRISM index from a [`PointStore`].
    pub fn build(mut store: PointStore, config: PrismConfig) -> PrismResult<Self> {
        let n = store.len;
        validate_build_configuration(
            &config,
            store.k(),
            n,
            has_multiple_attribute_tuples(&store.attrs, n),
        )?;

        // Normalizing at build makes SQ8-L2 approximate cosine ordering, since
        // L2^2 = 2 - 2cos on unit vectors. Rerank is scale-invariant, so
        // reported distances are unaffected.
        if config.metric == Metric::Cosine {
            let dim = store.dim;
            distance::normalize_rows(&mut store.vectors, dim);
        }

        let tree = PartitionTree::build(&store);
        let (store, tree, original_ids) = reorder_by_cell(store, tree)?;
        let cosine_norms = derive_cosine_norm_cache(&store, config.metric)?;
        let sq8 = if config.builds_sq8(n) {
            SQ8Store::build(&store)
        } else {
            empty_sq8(store.dim)?
        };
        let binary = if config.binary_rerank > 0 && config.metric != Metric::InnerProduct {
            BinaryStore::build(&store)?
        } else {
            BinaryStore::empty(store.dim)?
        };

        let mut point_cell = vec![0u32; n];
        for (ci, cell) in tree.cells.iter().enumerate() {
            for &pid in &cell.point_ids {
                point_cell[pid as usize] = ci as u32;
            }
        }

        // Inner product always scans exactly, and an index under the total scan
        // threshold makes every HIGH/MID eligible set scannable too. A graph would
        // be dead memory in both cases, so keep a canonical zero-edge CSR.
        let mut adj = AdjBuilder::new(n)?;
        if config.builds_local_graph(n) {
            build_local_edges(&store, &tree, &config, &mut adj)?;
        }

        let medoids = if config.builds_local_graph(n) {
            compute_medoids(&store, &tree, config.metric)
        } else {
            tree.cells.iter().map(|cell| cell.point_ids[0]).collect()
        };

        // A one-cell index always uses its local graph above the scan threshold,
        // so cross-cell phases only pay off when the router can pick a global graph.
        if config.builds_global_graph(n, tree.cells.len()) {
            // Cross-cell discovery must see only the completed local topology, so
            // this snapshot stays transient; the index derives its slices from the
            // final sorted CSR.
            let local_graph = adj.snapshot()?;
            build_greedy_cross_edges(
                &store,
                &tree,
                &medoids,
                &local_graph,
                &sq8,
                &point_cell,
                &config,
                &mut adj,
            )?;
            // Candidate discovery is the snapshot's only consumer, so releasing its
            // CSR here keeps the overlay and final freeze off peak memory.
            drop(local_graph);

            build_random_overlay_seeded(n, config.m_random, config.build_seed, &mut adj);

            // Neither greedy selection nor finitely many random permutations proves
            // connectivity. An undirected backbone over cell medoids completes that
            // invariant, cells themselves being connected by local spanning edges.
            add_medoid_backbone(&medoids, &mut adj)?;
        }

        let graph = adj.build()?;

        let global_medoid = if config.builds_global_graph(n, tree.cells.len()) {
            compute_global_medoid(&store, config.metric)
        } else {
            0
        };

        Ok(Self {
            store,
            tree,
            graph,
            medoids,
            global_medoid,
            point_cell,
            original_ids,
            sq8,
            binary,
            cosine_norms,
            config,
        })
    }
}

/// Canonical dimension-preserving SQ8 component for an exact-only index.
fn empty_sq8(dim: usize) -> PrismResult<SQ8Store> {
    SQ8Store::from_parts(Vec::new(), vec![0.0; dim], vec![1.0; dim], dim)
}

fn validate_canonical_graph(graph: &Graph, context: &str) -> PrismResult<()> {
    for point in 0..graph.len() as u32 {
        let neighbors = graph.neighbors_unchecked(point);
        if neighbors.binary_search(&point).is_ok() {
            return Err(PrismError::InvalidFormat(format!(
                "{context} contains a self-edge at point {point}"
            )));
        }
        if neighbors.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(PrismError::InvalidFormat(format!(
                "{context} neighbors for point {point} must be strictly increasing"
            )));
        }
    }
    Ok(())
}

fn in_cell_neighbors(graph: &Graph, point: u32, first: u32, last: u32) -> &[u32] {
    let neighbors = graph.neighbors_unchecked(point);
    let start = neighbors.partition_point(|&neighbor| neighbor < first);
    let end = neighbors.partition_point(|&neighbor| neighbor <= last);
    &neighbors[start..end]
}

fn validate_local_reachability(
    graph: &Graph,
    cells: &[super::partition::Cell],
    medoids: &[u32],
    require_reachability: bool,
) -> PrismResult<()> {
    for (cell_index, cell) in cells.iter().enumerate() {
        if cell
            .point_ids()
            .windows(2)
            .any(|pair| pair[0].checked_add(1) != Some(pair[1]))
        {
            return Err(PrismError::InvalidFormat(format!(
                "partition cell {cell_index} point IDs must form one contiguous range for local graph slicing"
            )));
        }
    }

    if !require_reachability {
        return Ok(());
    }

    let mut seen = vec![false; graph.len()];
    for (cell_index, cell) in cells.iter().enumerate() {
        let first = cell.point_ids().first().copied().ok_or_else(|| {
            PrismError::InvalidFormat(format!(
                "partition cell {cell_index} is empty during local graph validation"
            ))
        })?;
        let last = cell.point_ids().last().copied().ok_or_else(|| {
            PrismError::InvalidFormat(format!(
                "partition cell {cell_index} is empty during local graph validation"
            ))
        })?;
        let entry = medoids[cell_index];
        seen[entry as usize] = true;
        let mut pending = vec![entry];
        let mut reached = 1usize;
        while let Some(point) = pending.pop() {
            for &neighbor in in_cell_neighbors(graph, point, first, last) {
                if !seen[neighbor as usize] {
                    seen[neighbor as usize] = true;
                    reached += 1;
                    pending.push(neighbor);
                }
            }
        }
        if reached != cell.point_ids().len() {
            return Err(PrismError::InvalidFormat(format!(
                "local graph reaches {reached} of {} points in partition cell {cell_index}",
                cell.point_ids().len()
            )));
        }
    }
    Ok(())
}

fn validate_no_cross_cell_edges(graph: &Graph, point_cell: &[u32]) -> PrismResult<()> {
    for point in 0..graph.len() as u32 {
        let cell = point_cell[point as usize];
        if let Some(&neighbor) = graph
            .neighbors_unchecked(point)
            .iter()
            .find(|&&neighbor| point_cell[neighbor as usize] != cell)
        {
            return Err(PrismError::InvalidFormat(format!(
                "local-only graph edge {point}->{neighbor} crosses partition cells"
            )));
        }
    }
    Ok(())
}

fn validate_reachable(
    graph: &Graph,
    entry: u32,
    expected: usize,
    context: &str,
) -> PrismResult<()> {
    let mut seen = vec![false; graph.len()];
    seen[entry as usize] = true;
    let mut pending = vec![entry];
    let mut reached = 1usize;
    while let Some(point) = pending.pop() {
        for &neighbor in graph.neighbors_unchecked(point) {
            if !seen[neighbor as usize] {
                seen[neighbor as usize] = true;
                reached += 1;
                pending.push(neighbor);
            }
        }
    }
    if reached != expected {
        return Err(PrismError::InvalidFormat(format!(
            "{context} reaches {reached} of {expected} points from entry {entry}"
        )));
    }
    Ok(())
}

/// Reorder so points in the same cell are contiguous. Returns (store, tree, original_ids).
fn reorder_by_cell(
    store: PointStore,
    mut tree: PartitionTree,
) -> PrismResult<(PointStore, PartitionTree, Vec<u32>)> {
    let n = store.len;
    let dim = store.dim;
    let k = store.k();

    let mut new_order: Vec<u32> = Vec::with_capacity(n);
    for cell in &tree.cells {
        new_order.extend_from_slice(&cell.point_ids);
    }

    let mut old_to_new = vec![0u32; n];
    for (new_id, &old_id) in new_order.iter().enumerate() {
        old_to_new[old_id as usize] = new_id as u32;
    }

    let mut new_vectors = vec![0.0f32; n * dim];
    for (new_id, &old_id) in new_order.iter().enumerate() {
        let src = &store.vectors[old_id as usize * dim..(old_id as usize + 1) * dim];
        new_vectors[new_id * dim..(new_id + 1) * dim].copy_from_slice(src);
    }

    let mut new_attrs = Vec::with_capacity(k);
    for j in 0..k {
        let mut attr_col = vec![0u32; n];
        for (new_id, &old_id) in new_order.iter().enumerate() {
            attr_col[new_id] = store.attrs[j][old_id as usize];
        }
        new_attrs.push(attr_col);
    }

    for cell in &mut tree.cells {
        for pid in &mut cell.point_ids {
            *pid = old_to_new[*pid as usize];
        }
    }

    let new_store = PointStore::from_parts(new_vectors, dim, new_attrs)?;
    Ok((new_store, tree, new_order))
}

/// Build complete small-cell graphs or
/// [DiskANN's two-pass Vamana](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf)
/// plus PRISM's deterministic in-cell spanning backbone.
fn build_local_edges(
    store: &PointStore,
    tree: &PartitionTree,
    config: &PrismConfig,
    adj: &mut AdjBuilder,
) -> PrismResult<()> {
    let cell_edges: Vec<Vec<(u32, u32)>> = tree
        .cells
        .par_iter()
        .enumerate()
        .map(|(cell_idx, cell)| {
            let pts = &cell.point_ids;
            let mut edges = Vec::new();
            if pts.len() <= 1 {
                return edges;
            }

            if pts.len().saturating_sub(1) <= config.m_local {
                for i in 0..pts.len() {
                    for j in (i + 1)..pts.len() {
                        edges.push((pts[i], pts[j]));
                        edges.push((pts[j], pts[i]));
                    }
                }
            } else {
                let mut rng = StdRng::seed_from_u64(derive_build_seed(
                    config.build_seed,
                    0x4c4f_4341_4c00_0000 ^ cell_idx as u64,
                ));
                build_vamana_cell(store, pts, config, &mut edges, &mut rng);

                // Vamana's directed pruning does not prove every point is reachable
                // from the cell medoid. These chain edges give the cell a spanning
                // tree for at most two extra neighbors per point.
                for pair in pts.windows(2) {
                    edges.push((pair[0], pair[1]));
                    edges.push((pair[1], pair[0]));
                }

                // `local_graph` is snapshotted before the final freeze, so repeated
                // Vamana-pass and backbone edges must be removed here too.
                edges.sort_unstable();
                edges.dedup();
            }
            edges
        })
        .collect();

    for edges in cell_edges {
        for (src, dst) in edges {
            adj.add_edge(src, dst)?;
        }
    }
    Ok(())
}

/// Vamana construction within a single cell: full-precision GreedySearch and
/// RobustPrune, first with alpha=1 and then with the configured alpha.
fn build_vamana_cell(
    store: &PointStore,
    pts: &[u32],
    config: &PrismConfig,
    edges: &mut Vec<(u32, u32)>,
    rng: &mut impl Rng,
) {
    let n = pts.len();
    let r = config.m_local.min(n.saturating_sub(1));
    let beam = n.min(config.beam_width);

    let actual_r = r.min(n - 1);
    let mut graph: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            let mut neighbors = Vec::with_capacity(actual_r);
            while neighbors.len() < actual_r {
                let j = rng.gen_range(0..n);
                if j != i && !neighbors.contains(&j) {
                    neighbors.push(j);
                }
            }
            neighbors
        })
        .collect();

    let centroid = point_centroid(store, pts.iter().copied());
    let entry = (0..n)
        .min_by(|&a, &b| {
            let da = distance::distance(&centroid, store.vector_unchecked(pts[a]), config.metric);
            let db = distance::distance(&centroid, store.vector_unchecked(pts[b]), config.metric);
            da.total_cmp(&db)
        })
        .unwrap();

    for pass_alpha in [1.0, config.vamana_alpha] {
        let mut order: Vec<usize> = (0..n).collect();
        order.shuffle(rng);

        for &i in &order {
            // Vamana Algorithm 3 passes the complete set V of vertices expanded
            // by GreedySearch to RobustPrune, not merely its bounded final list.
            let search_results =
                vamana_search_full(store, config.metric, pts, &graph, entry, pts[i], beam);

            let mut candidates = search_results;
            for &nb in &graph[i] {
                if !candidates.contains(&nb) {
                    candidates.push(nb);
                }
            }

            graph[i] = robust_prune(store, pts, i, &candidates, pass_alpha, r, config.metric);

            let new_neighbors: Vec<usize> = graph[i].clone();
            for &j in &new_neighbors {
                if !graph[j].contains(&i) {
                    graph[j].push(i);
                    if graph[j].len() > r {
                        let cands: Vec<usize> = graph[j].clone();
                        graph[j] =
                            robust_prune(store, pts, j, &cands, pass_alpha, r, config.metric);
                    }
                }
            }
        }
    }

    for (i, neighbors) in graph.iter().enumerate() {
        for &j in neighbors {
            edges.push((pts[i], pts[j]));
        }
    }
}

/// Heap-ordered full-precision candidate distance between two stored points.
/// Vamana graph construction uses the original coordinates; SQ8 guides query
/// traversal only after the graph has been built.
#[inline]
fn build_cand_dist(store: &PointStore, metric: Metric, a: u32, b: u32) -> u32 {
    distance::ord_key(distance::distance(
        store.vector_unchecked(a),
        store.vector_unchecked(b),
        metric,
    ))
}

/// Full-precision mean of an internally proven nonempty point-ID stream.
///
/// Pairwise-distance validation bounds each component, but a long or
/// cancellation-heavy stream can still lose substantial information in a
/// naive f32 sum. Accumulating and dividing in f64 keeps medoid entry selection
/// finite and stable before the bounded mean is converted back to f32.
fn point_centroid(store: &PointStore, point_ids: impl IntoIterator<Item = u32>) -> Vec<f32> {
    let mut sums = vec![0.0f64; store.dim];
    let mut count = 0usize;
    for point in point_ids {
        for (sum, &component) in sums.iter_mut().zip(store.vector_unchecked(point)) {
            *sum += f64::from(component);
        }
        count += 1;
    }
    debug_assert!(count > 0, "centroids require at least one point");
    let inverse = 1.0 / count as f64;
    sums.into_iter()
        .map(|sum| {
            let mean = (sum * inverse) as f32;
            debug_assert!(mean.is_finite());
            mean
        })
        .collect()
}

/// Full-precision GreedySearch within a cell's evolving graph. Returns the
/// entire set `V` of expanded local vertices required by Vamana Algorithm 3.
fn vamana_search_full(
    store: &PointStore,
    metric: Metric,
    pts: &[u32],
    graph: &[Vec<usize>],
    entry: usize,
    query_id: u32,
    beam: usize,
) -> Vec<usize> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let mut expanded = vec![false; pts.len()];
    let mut in_list = vec![false; pts.len()];
    let mut expanded_vertices = Vec::new();
    let mut candidates: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
    let mut search_list: BinaryHeap<(u32, usize)> = BinaryHeap::new();

    let d = build_cand_dist(store, metric, query_id, pts[entry]);
    in_list[entry] = true;
    candidates.push(Reverse((d, entry)));
    search_list.push((d, entry));

    while let Some(Reverse((_distance, candidate))) = candidates.pop() {
        // Heap entries whose vertices were trimmed from L are stale. Expanded
        // vertices stay in V but are never expanded twice.
        if !in_list[candidate] || expanded[candidate] {
            continue;
        }
        expanded[candidate] = true;
        expanded_vertices.push(candidate);

        for &neighbor in &graph[candidate] {
            if expanded[neighbor] || in_list[neighbor] {
                continue;
            }
            let neighbor_distance = build_cand_dist(store, metric, query_id, pts[neighbor]);
            in_list[neighbor] = true;
            candidates.push(Reverse((neighbor_distance, neighbor)));
            search_list.push((neighbor_distance, neighbor));
            if search_list.len() > beam {
                let (_, removed) = search_list
                    .pop()
                    .expect("the overfull GreedySearch list must have a worst candidate");
                in_list[removed] = false;
            }
        }
    }

    expanded_vertices
}

/// Robust prune: rejects c if alpha * dist(c, selected) <= dist(p, c).
fn robust_prune(
    store: &PointStore,
    pts: &[u32],
    p: usize,
    candidates: &[usize],
    alpha: f32,
    r: usize,
    metric: Metric,
) -> Vec<usize> {
    let p_vec = store.vector_unchecked(pts[p]);
    let mut sorted: Vec<(usize, f32)> = candidates
        .iter()
        .filter(|&&c| c != p)
        .map(|&c| {
            (
                c,
                distance::distance(p_vec, store.vector_unchecked(pts[c]), metric),
            )
        })
        .collect();
    sorted.sort_by(|a, b| a.1.total_cmp(&b.1));
    sorted.dedup_by_key(|x| x.0);

    let mut selected: Vec<usize> = Vec::with_capacity(r);
    for &(c, d_pc) in &sorted {
        if selected.len() >= r {
            break;
        }
        let dominated = selected.iter().any(|&s| {
            let d_cs = distance::distance(
                store.vector_unchecked(pts[c]),
                store.vector_unchecked(pts[s]),
                metric,
            );
            // These distances are squared, but DiskANN defines alpha over the
            // unsquared metric, so its test becomes alpha^2 * d(c,s)^2 <= d(p,c)^2.
            let alpha_factor = match metric {
                Metric::L2 | Metric::Cosine => f64::from(alpha) * f64::from(alpha),
                Metric::InnerProduct => f64::from(alpha),
            };
            alpha_factor * f64::from(d_cs) <= f64::from(d_pc)
        });
        if !dominated {
            selected.push(c);
        }
    }
    selected
}

/// Fixed-width, source-cell-indexed target table for greedy cross edges.
struct CrossCellTargets {
    cells: Vec<u32>,
    targets_per_source: usize,
}

impl CrossCellTargets {
    #[inline]
    fn for_source(&self, source_cell: usize) -> &[u32] {
        let start = source_cell * self.targets_per_source;
        &self.cells[start..start + self.targets_per_source]
    }
}

#[inline]
fn cross_cell_target_count(cell_count: usize, config: &PrismConfig, beam: usize) -> usize {
    cell_count
        .saturating_sub(1)
        .min(config.m_greedy.max(4))
        .min(beam)
}

#[inline]
fn bounded_cross_cell_pool_size(cell_count: usize, target_count: usize, beam: usize) -> usize {
    cell_count
        .saturating_sub(1)
        .min(beam.max(target_count.saturating_mul(4)))
}

/// Rank target cells once per source cell rather than once per source point.
///
/// Small catalogs score every foreign medoid. Large catalogs score one
/// deterministic representative from each stratum of a bounded candidate
/// pool, then medoid-scores that sample without returning to quadratic work for
/// near-unique attribute tuples.
fn precompute_cross_cell_targets(
    medoids: &[u32],
    sq8: &SQ8Store,
    config: &PrismConfig,
    beam: usize,
) -> PrismResult<CrossCellTargets> {
    let cell_count = medoids.len();
    debug_assert!(cell_count > 1);
    let target_count = cross_cell_target_count(cell_count, config, beam);
    debug_assert!(target_count > 0);

    let exact_ranking = config.cross_cell_exact_ranking_limit > 0
        && cell_count <= config.cross_cell_exact_ranking_limit;
    let candidate_pool_size = if exact_ranking {
        cell_count - 1
    } else {
        bounded_cross_cell_pool_size(cell_count, target_count, beam)
    };
    debug_assert!(candidate_pool_size >= target_count);

    let table_len = cell_count.checked_mul(target_count).ok_or_else(|| {
        PrismError::Overflow("cross-cell target table length overflowed usize".into())
    })?;
    let mut cells = Vec::new();
    cells.try_reserve_exact(table_len).map_err(|error| {
        PrismError::Overflow(format!(
            "cannot allocate {table_len} cross-cell target identifiers: {error}"
        ))
    })?;
    cells.resize(table_len, 0u32);

    let scratch_allocation_failed = std::sync::atomic::AtomicBool::new(false);
    cells
        .par_chunks_mut(target_count)
        .enumerate()
        .for_each(|(source_cell, output)| {
            let mut candidates: Vec<(u32, u32)> = Vec::new();
            if candidates.try_reserve_exact(candidate_pool_size).is_err() {
                scratch_allocation_failed.store(true, std::sync::atomic::Ordering::Relaxed);
                return;
            }

            if exact_ranking {
                for target_cell in 0..cell_count {
                    if target_cell != source_cell {
                        candidates.push((
                            target_cell as u32,
                            distance::ord_key(
                                sq8.code_l2(medoids[source_cell], medoids[target_cell]),
                            ),
                        ));
                    }
                }
            } else {
                let foreign_count = cell_count - 1;
                for stratum in 0..candidate_pool_size {
                    // Compute stratum boundaries in u128 so even maximum u32
                    // point/cell catalogs cannot overflow on 32-bit targets.
                    let start = ((stratum as u128 * foreign_count as u128)
                        / candidate_pool_size as u128) as usize;
                    let end = (((stratum + 1) as u128 * foreign_count as u128)
                        / candidate_pool_size as u128) as usize;
                    let width = end - start;
                    debug_assert!(width > 0);
                    let stream =
                        0x4352_4f53_5354_5241u64 ^ ((source_cell as u64) << 32) ^ stratum as u64;
                    let offset =
                        (derive_build_seed(config.build_seed, stream) % width as u64) as usize;
                    let compressed_target = start + offset;
                    let target_cell = if compressed_target >= source_cell {
                        compressed_target + 1
                    } else {
                        compressed_target
                    };
                    candidates.push((
                        target_cell as u32,
                        distance::ord_key(sq8.code_l2(medoids[source_cell], medoids[target_cell])),
                    ));
                }
            }

            if candidates.len() > target_count {
                candidates.select_nth_unstable_by_key(target_count, |&(cell, dist)| (dist, cell));
                candidates.truncate(target_count);
            }
            candidates.sort_unstable_by_key(|&(cell, dist)| (dist, cell));
            for (slot, &(target_cell, _)) in candidates.iter().enumerate() {
                output[slot] = target_cell;
            }
        });

    if scratch_allocation_failed.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(PrismError::Overflow(format!(
            "cannot allocate a {candidate_pool_size}-cell cross-ranking scratch buffer"
        )));
    }

    Ok(CrossCellTargets {
        cells,
        targets_per_source: target_count,
    })
}

/// Greedy attribute-diverse cross-partition edges. SQ8 beam search for
/// candidate discovery, f32 rerank, parallelized across points.
#[allow(clippy::too_many_arguments)]
fn build_greedy_cross_edges(
    store: &PointStore,
    tree: &PartitionTree,
    medoids: &[u32],
    local_graph: &Graph,
    sq8: &SQ8Store,
    point_cell: &[u32],
    config: &PrismConfig,
    adj: &mut AdjBuilder,
) -> PrismResult<()> {
    if config.m_greedy == 0 || tree.cells.len() <= 1 {
        return Ok(());
    }

    let n = store.len;
    let k = store.k();
    let t = config.t.min(k);
    let beam = config.beam_width.min(n);
    let subset_count = checked_covering_subset_count(k, t)?;
    let subsets = generate_t_subsets(k, t, subset_count)?;
    // SQ8-L2 stays metric-consistent for L2 and build-normalized Cosine, though
    // quantization can still reorder near-ties. InnerProduct has no such bound.
    let use_sq8 = config.metric != Metric::InnerProduct;
    let target_cells = precompute_cross_cell_targets(medoids, sq8, config, beam)?;
    let per_cell_budget = beam.div_ceil(target_cells.targets_per_source);
    let local_search_width = per_cell_budget.max(config.m_local).min(beam);

    let point_edges: Vec<Vec<u32>> = (0..n as u32)
        .into_par_iter()
        .map(|p_id| -> PrismResult<Vec<u32>> {
            let p_cell_idx = point_cell[p_id as usize] as usize;
            let p_vec = store.vector_unchecked(p_id);

            // Draw per target cell rather than one global beam: a single foreign
            // cell filling the beam leaves every candidate sharing one attribute
            // tuple, and attribute-diverse selection then has nothing to select.
            let candidate_capacity = per_cell_budget
                .checked_mul(target_cells.targets_per_source)
                .ok_or_else(|| {
                    PrismError::Overflow(format!(
                        "cross-cell candidate capacity overflowed for point {p_id}"
                    ))
                })?;
            let mut all_cand_ids: Vec<u32> = Vec::new();
            all_cand_ids
                .try_reserve_exact(candidate_capacity)
                .map_err(|error| {
                    PrismError::Overflow(format!(
                        "cannot allocate {candidate_capacity} cross-cell candidates for point {p_id}: {error}"
                    ))
                })?;
            for &target_cell in target_cells.for_source(p_cell_idx) {
                let ci = target_cell as usize;
                let cell_size = tree.cells[ci].point_ids.len();

                if use_sq8 && cell_size > local_search_width.saturating_mul(2) {
                    let mut found =
                        beam_search_sq8(sq8, local_graph, p_id, medoids[ci], local_search_width);
                    found.sort_unstable_by_key(|&(_, distance)| distance);
                    for (id, _) in found.into_iter().take(per_cell_budget) {
                        all_cand_ids.push(id);
                    }
                } else if use_sq8 {
                    let mut scored: Vec<(u32, u32)> = Vec::new();
                    scored.try_reserve_exact(cell_size).map_err(|error| {
                        PrismError::Overflow(format!(
                            "cannot allocate {cell_size} SQ8 cross-cell scores for point {p_id}: {error}"
                        ))
                    })?;
                    scored.extend(
                        tree.cells[ci]
                            .point_ids
                            .iter()
                            .map(|&q| (q, distance::ord_key(sq8.code_l2(p_id, q)))),
                    );
                    scored.sort_unstable_by_key(|&(_, d)| d);
                    for &(id, _) in scored.iter().take(per_cell_budget) {
                        all_cand_ids.push(id);
                    }
                } else {
                    let mut scored: Vec<(u32, f32)> = Vec::new();
                    scored.try_reserve_exact(cell_size).map_err(|error| {
                        PrismError::Overflow(format!(
                            "cannot allocate {cell_size} exact cross-cell scores for point {p_id}: {error}"
                        ))
                    })?;
                    scored.extend(tree.cells[ci].point_ids.iter().map(|&q| {
                            (
                                q,
                                distance::distance(p_vec, store.vector_unchecked(q), config.metric),
                            )
                        }));
                    scored.sort_by(|a, b| a.1.total_cmp(&b.1));
                    for &(id, _) in scored.iter().take(per_cell_budget) {
                        all_cand_ids.push(id);
                    }
                }
            }

            let mut candidates: Vec<(u32, f32)> = Vec::new();
            candidates
                .try_reserve_exact(all_cand_ids.len())
                .map_err(|error| {
                    PrismError::Overflow(format!(
                        "cannot allocate {} exact cross-cell rerank scores for point {p_id}: {error}",
                        all_cand_ids.len()
                    ))
                })?;
            candidates.extend(all_cand_ids.iter().map(|&id| {
                    (
                        id,
                        distance::distance(p_vec, store.vector_unchecked(id), config.metric),
                    )
                }));
            candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
            candidates.truncate(beam);

            Ok(select_cross_neighbors(store, &candidates, config, &subsets))
        })
        .collect::<PrismResult<Vec<_>>>()?;

    for (p_id, neighbors) in point_edges.into_iter().enumerate() {
        for q_id in neighbors {
            adj.add_edge(p_id as u32, q_id)?;
        }
    }
    Ok(())
}

/// SQ8 beam search through a cell's local graph. Returns (point_id, sq8_distance).
fn beam_search_sq8(
    sq8: &SQ8Store,
    graph: &Graph,
    query_id: u32,
    entry: u32,
    beam: usize,
) -> Vec<(u32, u32)> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let mut visited = HashSet::new();
    let mut candidates: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
    let mut results: BinaryHeap<(u32, u32)> = BinaryHeap::new();

    let d = distance::ord_key(sq8.code_l2(query_id, entry));
    visited.insert(entry);
    candidates.push(Reverse((d, entry)));
    results.push((d, entry));

    while let Some(Reverse((d, c))) = candidates.pop() {
        if results.len() >= beam {
            if let Some(&(worst, _)) = results.peek() {
                if d > worst {
                    break;
                }
            }
        }

        for &w in graph.neighbors_unchecked(c) {
            if !visited.insert(w) {
                continue;
            }
            let wd = distance::ord_key(sq8.code_l2(query_id, w));
            candidates.push(Reverse((wd, w)));
            results.push((wd, w));
            if results.len() > beam {
                results.pop();
            }
        }
    }

    results.into_iter().map(|(d, id)| (id, d)).collect()
}

/// Attribute-diverse neighbor selection. Candidates sorted by distance.
pub(crate) fn select_cross_neighbors(
    store: &PointStore,
    candidates: &[(u32, f32)],
    config: &PrismConfig,
    subsets: &[Vec<usize>],
) -> Vec<u32> {
    let m_g = config.m_greedy;
    let alpha = config.alpha;

    if candidates.is_empty() || m_g == 0 {
        return Vec::new();
    }

    let mut covered: HashSet<AttributeTuple> = HashSet::new();
    let selection_limit = m_g.min(candidates.len());
    let mut selected = Vec::with_capacity(selection_limit);
    let mut available: Vec<bool> = vec![true; candidates.len()];

    for _ in 0..selection_limit {
        let mut best_idx = None;
        let mut best_score = f32::NEG_INFINITY;

        for (idx, &(q_id, dist)) in candidates.iter().enumerate() {
            if !available[idx] {
                continue;
            }

            let new_tuples = count_new_tuples(store, q_id, &covered, subsets);

            let score = if alpha == 0.0 || dist == 0.0 {
                new_tuples as f32
            } else {
                (new_tuples as f32 + 0.001) / dist.powf(alpha)
            };

            if score > best_score {
                best_score = score;
                best_idx = Some(idx);
            }
        }

        let Some(idx) = best_idx else { break };
        selected.push(candidates[idx].0);
        available[idx] = false;

        add_tuples(store, candidates[idx].0, &mut covered, subsets);
    }

    selected
}

/// Collision-free key for one projected attribute tuple.
pub(crate) type AttributeTuple = Box<[(usize, u32)]>;

#[inline]
fn tuple_key(combo: &[usize], store: &PointStore, q: u32) -> AttributeTuple {
    combo
        .iter()
        .map(|&j| (j, store.attr_unchecked(q, j)))
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

/// Count how many new t-tuples a candidate would contribute.
fn count_new_tuples(
    store: &PointStore,
    q: u32,
    covered: &HashSet<AttributeTuple>,
    subsets: &[Vec<usize>],
) -> usize {
    let mut count = 0;
    for combo in subsets {
        let key = tuple_key(combo, store, q);
        if !covered.contains(&key) {
            count += 1;
        }
    }
    count
}

/// Add all t-tuples of a point to the covered set.
pub(crate) fn add_tuples(
    store: &PointStore,
    q: u32,
    covered: &mut HashSet<AttributeTuple>,
    subsets: &[Vec<usize>],
) {
    for combo in subsets {
        let key = tuple_key(combo, store, q);
        covered.insert(key);
    }
}

/// Generate all t-element subsets of [0..k] after their bounded count has
/// been validated.
fn generate_t_subsets(k: usize, t: usize, subset_count: usize) -> PrismResult<Vec<Vec<usize>>> {
    let mut result = Vec::new();
    result.try_reserve_exact(subset_count).map_err(|error| {
        PrismError::Overflow(format!(
            "cannot allocate {subset_count} covering subsets: {error}"
        ))
    })?;

    if t == 0 {
        result.push(Vec::new());
        return Ok(result);
    }

    // Iterative lexicographic generation avoids recursion depth depending on
    // an externally supplied attribute count.
    let mut combo = Vec::new();
    combo.try_reserve_exact(t).map_err(|error| {
        PrismError::Overflow(format!("cannot allocate a {t}-attribute subset: {error}"))
    })?;
    combo.extend(0..t);

    loop {
        let mut projection = Vec::new();
        projection.try_reserve_exact(t).map_err(|error| {
            PrismError::Overflow(format!("cannot allocate a {t}-attribute subset: {error}"))
        })?;
        projection.extend_from_slice(&combo);
        result.push(projection);

        let Some(pivot) = (0..t).rev().find(|&i| combo[i] < k - t + i) else {
            break;
        };
        combo[pivot] += 1;
        for i in pivot + 1..t {
            combo[i] = combo[i - 1] + 1;
        }
    }

    debug_assert_eq!(result.len(), subset_count);
    Ok(result)
}

/// Convenience helper for the finite empirical-invariant test module. Production
/// construction validates the binomial bound and propagates allocation errors
/// instead of calling this infallible test convenience wrapper.
#[cfg(test)]
pub(crate) fn t_subsets(k: usize, t: usize) -> Vec<Vec<usize>> {
    let count = checked_covering_subset_count(k, t)
        .expect("verification fixture must stay within the covering-subset bound");
    generate_t_subsets(k, t, count)
        .expect("verification fixture covering subsets must be allocatable")
}

fn checked_covering_subset_count(k: usize, t: usize) -> PrismResult<usize> {
    if t > k {
        return Err(PrismError::InvalidInput(format!(
            "covering strength {t} cannot exceed attribute count {k}"
        )));
    }
    let choose = t.min(k - t);
    let mut count = 1u128;
    for i in 1..=choose {
        let numerator = (k - choose + i) as u128;
        count = count.checked_mul(numerator).ok_or_else(|| {
            PrismError::Overflow(format!(
                "binomial covering count C({k}, {t}) exceeds representable arithmetic"
            ))
        })? / i as u128;
        if count > MAX_COVERING_SUBSETS as u128 {
            return Err(PrismError::InvalidInput(format!(
                "covering strength t={t} over {k} attributes requires more than {MAX_COVERING_SUBSETS} subsets; lower t or reduce the attribute schema"
            )));
        }
    }
    Ok(count as usize)
}

/// Connect cell medoids in deterministic partition order. Together with each
/// cell's local spanning edges, this makes the whole graph connected whenever
/// search is allowed to enter it once from `global_medoid`.
fn add_medoid_backbone(medoids: &[u32], adj: &mut AdjBuilder) -> PrismResult<()> {
    for pair in medoids.windows(2) {
        adj.add_undirected(pair[0], pair[1])?;
    }
    Ok(())
}

/// Derive a reproducible stream with Sebastiano Vigna's
/// [SplitMix64 mixer](https://prng.di.unimi.it/splitmix64.c).
#[inline]
fn derive_build_seed(base: u64, stream: u64) -> u64 {
    let mut value = base
        .wrapping_add(stream)
        .wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Random whole-index edges via a permutation overlay. The
/// compatibility helper is deterministic; full builds use their configured
/// seed through `build_random_overlay_seeded`.
#[cfg(test)]
pub(crate) fn build_random_overlay(n: usize, m_random: usize, adj: &mut AdjBuilder) {
    build_random_overlay_seeded(n, m_random, PrismConfig::default().build_seed, adj);
}

fn build_random_overlay_seeded(n: usize, m_random: usize, build_seed: u64, adj: &mut AdjBuilder) {
    if m_random == 0 || n <= 1 {
        return;
    }
    let mut rng = StdRng::seed_from_u64(derive_build_seed(build_seed, 0x4f56_4552_4c41_5900));
    // A simple graph holds at most n-1 distinct neighbors per point, so rounds
    // beyond that add no degree and let an extreme config loop unbounded.
    let half = (m_random / 2).min(n.saturating_sub(1));

    for _ in 0..half {
        let mut perm: Vec<u32> = (0..n as u32).collect();
        perm.shuffle(&mut rng);
        for (i, &j) in perm.iter().enumerate() {
            if i as u32 != j {
                adj.add_undirected(i as u32, j)
                    .expect("random overlay permutation produced an invalid point ID");
            }
        }
    }
}

/// Per-cell medoid via centroid-nearest approximation.
fn compute_medoids(store: &PointStore, tree: &PartitionTree, metric: Metric) -> Vec<u32> {
    tree.cells
        .iter()
        .map(|cell| {
            let pts = &cell.point_ids;
            if pts.len() == 1 {
                return pts[0];
            }
            let centroid = point_centroid(store, pts.iter().copied());
            *pts.iter()
                .min_by(|&&a, &&b| {
                    let da = distance::distance(&centroid, store.vector_unchecked(a), metric);
                    let db = distance::distance(&centroid, store.vector_unchecked(b), metric);
                    da.total_cmp(&db)
                })
                .unwrap()
        })
        .collect()
}

/// Compute global medoid: the point closest to the centroid of the entire dataset.
fn compute_global_medoid(store: &PointStore, metric: Metric) -> u32 {
    let n = store.len;
    let centroid = point_centroid(store, 0..n as u32);
    (0..n as u32)
        .min_by(|&a, &b| {
            let da = distance::distance(&centroid, store.vector_unchecked(a), metric);
            let db = distance::distance(&centroid, store.vector_unchecked(b), metric);
            da.total_cmp(&db)
        })
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::super::filter::Filter;
    use super::super::point::PointStore;
    use super::super::search::SearchExecution;
    use super::*;

    #[test]
    fn cosine_norm_cache_uses_the_zeroizing_drop_path() {
        let mut cache = CosineNormCache {
            values: vec![0.0, 0.5, 1.0, 2.0],
        };
        cache.zeroize();
        assert_eq!(cache.values, vec![0.0; 4]);
        // `Drop` invokes the same operation again when this test returns.
    }

    #[test]
    fn centroid_accumulation_preserves_cancellation_and_extreme_finite_means() {
        let cancellation = PointStore::from_parts(
            vec![100_000_000.0, 1.0, -100_000_000.0],
            1,
            vec![vec![0, 0, 0]],
        )
        .unwrap();
        let mean = point_centroid(&cancellation, 0..3);
        assert!((mean[0] - 1.0 / 3.0).abs() <= f32::EPSILON);

        // Components near the dimension-one bound: their mean must stay finite even
        // though centroid arithmetic runs downstream of PointStore validation.
        let extreme =
            PointStore::from_parts(vec![6.0e18, 6.0e18, -6.0e18], 1, vec![vec![0, 0, 0]]).unwrap();
        let mean = point_centroid(&extreme, 0..3);
        assert!(mean[0].is_finite());
        assert!((f64::from(mean[0]) - 2.0e18).abs() / 2.0e18 < 1.0e-6);
    }

    fn persisted_fixture(metric: Metric) -> PrismIndex {
        persisted_fixture_with_binary(metric, 0)
    }

    fn persisted_fixture_with_binary(metric: Metric, binary_rerank: usize) -> PrismIndex {
        let mut store = PointStore::new(6, 2).unwrap();
        for point in 0..96usize {
            let vector: Vec<f32> = (0..6)
                .map(|dimension| {
                    let value = (point * 6 + dimension) as f32;
                    (value * 0.031).sin() + (value * 0.017).cos()
                })
                .collect();
            store
                .push(&vector, &[(point % 3) as u32, ((point / 3) % 2) as u32])
                .unwrap();
        }
        PrismIndex::build(
            store,
            PrismConfig {
                metric,
                binary_rerank,
                m_local: 6,
                m_greedy: 4,
                m_random: 4,
                beam_width: 20,
                scan_threshold: 0,
                ..Default::default()
            },
        )
        .unwrap()
    }

    fn reassemble(index: PrismIndex) -> PrismResult<PrismIndex> {
        let PrismIndex {
            store,
            tree,
            graph,
            medoids,
            global_medoid,
            point_cell,
            original_ids,
            sq8,
            binary,
            cosine_norms: _,
            config,
        } = index;
        PrismIndex::from_parts(
            store,
            tree,
            graph,
            medoids,
            global_medoid,
            point_cell,
            original_ids,
            sq8,
            binary,
            config,
        )
    }

    fn assert_derived_cosine_norm_cache(index: &PrismIndex) {
        if index.config.metric != Metric::Cosine {
            assert_eq!(index.cosine_norms.len(), 0);
            return;
        }

        assert_eq!(index.cosine_norms.len(), index.store.len);
        for point in 0..index.store.len as u32 {
            assert_eq!(
                index.cosine_norms.get(point).to_bits(),
                distance::cosine_norm(index.store.vector_unchecked(point)).to_bits(),
                "cached norm for internal point {point} must match its reordered row"
            );
        }
    }

    #[test]
    fn whole_index_below_scan_threshold_skips_graphs_and_roundtrips() {
        let mut store = PointStore::new(2, 2).unwrap();
        store.push(&[0.0, 0.0], &[0, 0]).unwrap();
        store.push(&[1.0, 0.0], &[0, 1]).unwrap();
        store.push(&[0.0, 1.0], &[1, 0]).unwrap();
        store.push(&[1.0, 1.0], &[1, 1]).unwrap();

        let config = PrismConfig {
            m_local: 2,
            m_greedy: 2,
            m_random: 4,
            t: 1,
            alpha: 0.0,
            beam_width: 10,
            ..Default::default()
        };

        let index = PrismIndex::build(store, config).unwrap();
        assert_eq!(index.tree.cells.len(), 4);
        assert_eq!(index.medoids.len(), 4);
        assert_eq!(index.graph.num_edges(), 0);
        assert!(index.sq8.is_empty());
        assert_eq!(index.sq8.dim(), index.store.dim());
        assert_eq!(index.global_medoid, 0);
        assert!(index
            .tree
            .cells
            .iter()
            .zip(&index.medoids)
            .all(|(cell, &medoid)| medoid == cell.point_ids[0]));

        let query = [0.2, 0.1];
        let expected = index.search_exact(&query, &Filter::none(), 3).unwrap();
        let outcome = index.search(&query, &Filter::none(), 3, 8).unwrap();
        assert_eq!(
            outcome.diagnostics.primary_execution,
            SearchExecution::ExactScan
        );
        assert_eq!(
            outcome
                .results
                .iter()
                .map(|result| (result.id, result.dist.to_bits()))
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|result| (result.id, result.dist.to_bits()))
                .collect::<Vec<_>>()
        );

        let rebuilt = reassemble(index).expect("scan-only persisted parts must reassemble");
        assert_eq!(rebuilt.graph.num_edges(), 0);
        assert!(rebuilt.sq8.is_empty());
        assert_eq!(
            rebuilt
                .search(&query, &Filter::none(), 3, 8)
                .unwrap()
                .diagnostics
                .primary_execution,
            SearchExecution::ExactScan
        );
    }

    #[test]
    fn scan_only_parts_reject_dead_sq8_accelerator_data() {
        let store =
            PointStore::from_parts(vec![0.0, 1.0, 2.0, 3.0], 1, vec![vec![0, 0, 1, 1]]).unwrap();
        let index = PrismIndex::build(store, PrismConfig::default()).unwrap();
        assert!(index.sq8.is_empty());
        let invalid_sq8 = SQ8Store::build(&index.store);
        let PrismIndex {
            store,
            tree,
            graph,
            medoids,
            global_medoid,
            point_cell,
            original_ids,
            sq8: _,
            binary,
            cosine_norms: _,
            config,
        } = index;
        assert!(matches!(
            PrismIndex::from_parts(
                store,
                tree,
                graph,
                medoids,
                global_medoid,
                point_cell,
                original_ids,
                invalid_sq8,
                binary,
                config,
            ),
            Err(PrismError::InvalidFormat(message)) if message.contains("0 point codes")
        ));
    }

    #[test]
    fn one_cell_above_threshold_builds_only_the_local_graph() {
        let mut store = PointStore::new(4, 1).unwrap();
        for point in 0..64usize {
            let vector: Vec<f32> = (0..4)
                .map(|dimension| ((point * 4 + dimension) as f32 * 0.17).sin())
                .collect();
            store.push(&vector, &[0]).unwrap();
        }
        let index = PrismIndex::build(
            store,
            PrismConfig {
                m_local: 8,
                m_greedy: 4,
                m_random: 4,
                beam_width: 24,
                scan_threshold: 16,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(index.tree.cells.len(), 1);
        assert!(index.graph.num_edges() > 0);
        assert!((0..index.store.len as u32).all(|point| index
            .graph
            .neighbors_unchecked(point)
            .iter()
            .all(|&neighbor| index.point_cell[neighbor as usize] == 0)));
        assert!(!index
            .config
            .builds_global_graph(index.store.len, index.tree.cells.len()));
    }

    #[test]
    fn vamana_pruning_receives_the_entire_greedy_search_visited_set() {
        let store = PointStore::from_parts(vec![0.0, 1.0, 2.0, 3.0], 1, vec![vec![0; 4]]).unwrap();
        let points = [0, 1, 2, 3];
        let graph = vec![vec![1], vec![2], vec![3], vec![]];

        let visited = vamana_search_full(&store, Metric::L2, &points, &graph, 0, 3, 2);

        assert_eq!(visited, vec![0, 1, 2, 3]);
        assert!(
            visited.len() > 2,
            "Vamana must not substitute the bounded final search list for V"
        );
    }

    #[test]
    fn vamana_alpha_uses_euclidean_semantics_with_squared_l2() {
        // Squared d(p,c)=2 and d(selected,c)=1, so alpha=1.5 must test
        // 1.5^2 * 1 <= 2, which is false and keeps c. Using an unsquared 1.5
        // would prune it.
        let store = PointStore::from_parts(vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0], 2, vec![vec![0; 3]])
            .unwrap();
        let points = [0, 1, 2];

        assert_eq!(
            robust_prune(&store, &points, 0, &[1, 2], 1.0, 2, Metric::L2),
            vec![1]
        );
        assert_eq!(
            robust_prune(&store, &points, 0, &[1, 2], 1.5, 2, Metric::L2),
            vec![1, 2]
        );
    }

    #[test]
    fn test_t_subsets() {
        let subs = t_subsets(4, 2);
        assert_eq!(subs.len(), 6); // C(4,2) = 6
        let subs = t_subsets(3, 1);
        assert_eq!(subs.len(), 3);
    }

    #[test]
    fn covering_strength_is_clamped_to_the_attribute_count() {
        let mut store = PointStore::new(1, 1).unwrap();
        store.push(&[0.0], &[0]).unwrap();
        store.push(&[1.0], &[1]).unwrap();

        let index = PrismIndex::build(
            store,
            PrismConfig {
                m_local: 1,
                m_greedy: 1,
                m_random: 0,
                t: usize::MAX,
                beam_width: 2,
                scan_threshold: 0,
                ..Default::default()
            },
        )
        .expect("effective covering strength must be min(t, attribute_count)");

        assert_eq!(index.config().t, usize::MAX);
        assert_eq!(index.tree().num_attributes(), 1);
    }

    #[test]
    fn borrowed_build_preflight_rejects_deterministic_input_and_schema_failures() {
        let extreme_vectors = vec![f32::MAX, f32::MAX];
        let one_attribute = vec![vec![0]];
        let error = PrismIndex::validate_build_input(
            &extreme_vectors,
            2,
            &one_attribute,
            &PrismConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            PrismError::InvalidInput(message) if message.contains("absolute magnitude")
        ));
        assert_eq!(extreme_vectors, vec![f32::MAX, f32::MAX]);

        let vectors = vec![0.0, 1.0];
        let attributes = vec![vec![0, 1]; 64];
        let error = PrismIndex::validate_build_input(
            &vectors,
            1,
            &attributes,
            &PrismConfig {
                m_greedy: 1,
                m_random: 0,
                t: 32,
                scan_threshold: 0,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            PrismError::InvalidInput(message)
                if message.contains("more than 4096 subsets")
        ));
        assert_eq!(vectors, vec![0.0, 1.0]);
        assert!(attributes.iter().all(|attribute| attribute == &[0, 1]));
    }

    #[test]
    fn excessive_covering_subset_count_is_rejected_before_materialization() {
        let mut store = PointStore::new(1, 64).unwrap();
        store.push(&[0.0], &vec![0; 64]).unwrap();
        store.push(&[1.0], &vec![1; 64]).unwrap();
        let error = PrismIndex::build(
            store,
            PrismConfig {
                m_local: 1,
                m_greedy: 1,
                m_random: 0,
                t: 32,
                beam_width: 2,
                scan_threshold: 0,
                ..Default::default()
            },
        )
        .err()
        .expect("C(64, 32) must exceed the explicit covering-subset bound");
        assert!(matches!(
            error,
            PrismError::InvalidInput(message)
                if message.contains("more than 4096 subsets")
        ));
    }

    #[test]
    fn tuple_keys_do_not_collide_for_wide_values_or_strength() {
        let store = PointStore::from_parts(
            vec![0.0, 1.0],
            1,
            vec![
                vec![0, 0],
                vec![0, 256],
                vec![7, 8],
                vec![9, 10],
                vec![11, 12],
            ],
        )
        .unwrap();

        assert_ne!(tuple_key(&[1], &store, 0), tuple_key(&[1], &store, 1));
        assert_ne!(
            tuple_key(&[0, 1, 2, 3, 4], &store, 0),
            tuple_key(&[0, 1, 2, 3, 4], &store, 1)
        );
    }

    #[test]
    fn equal_thresholds_skip_cross_cell_graph_phases() {
        let mut store = PointStore::new(3, 2).unwrap();
        for i in 0..100usize {
            store
                .push(
                    &[i as f32, (i as f32 * 0.1).sin(), (i as f32 * 0.1).cos()],
                    &[(i % 5) as u32, ((i / 5) % 4) as u32],
                )
                .unwrap();
        }
        let index = PrismIndex::build(
            store,
            PrismConfig {
                m_local: 4,
                m_greedy: 4,
                m_random: 4,
                t: 1,
                beam_width: 12,
                sigma_low: 0.01,
                sigma_high: 0.01,
                scan_threshold: 0,
                ..Default::default()
            },
        )
        .unwrap();

        for point in 0..index.store.len as u32 {
            assert!(index
                .graph
                .neighbors_unchecked(point)
                .iter()
                .all(|&neighbor| index.point_cell[neighbor as usize]
                    == index.point_cell[point as usize]));
        }
    }

    #[test]
    fn greedy_cross_edge_generator_exposes_multiple_attribute_cells() {
        let mut store = PointStore::new(3, 1).unwrap();
        for value in 0..5u32 {
            for i in 0..20usize {
                store
                    .push(
                        &[
                            value as f32 * 10.0 + i as f32 * 0.01,
                            (i as f32 * 0.17).sin(),
                            (i as f32 * 0.11).cos(),
                        ],
                        &[value],
                    )
                    .unwrap();
            }
        }

        let index = PrismIndex::build(
            store,
            PrismConfig {
                m_local: 4,
                m_greedy: 4,
                m_random: 0,
                t: 1,
                alpha: 0.0,
                beam_width: 8,
                scan_threshold: 0,
                ..Default::default()
            },
        )
        .unwrap();

        let source = index.tree.cells[0].point_ids[0];
        let source_cell = index.point_cell[source as usize];
        let target_cells: HashSet<u32> = index
            .graph
            .neighbors_unchecked(source)
            .iter()
            .map(|&neighbor| index.point_cell[neighbor as usize])
            .filter(|&cell| cell != source_cell)
            .collect();

        assert_eq!(
            target_cells.len(),
            4,
            "the generated candidate pool must let t=1 select one neighbor from every foreign attribute cell"
        );
    }

    #[test]
    fn cross_cell_targets_exactly_rank_foreign_medoids_once_per_source_cell() {
        let mut store = PointStore::new(3, 1).unwrap();
        for cell in 0..8u32 {
            for point in 0..2usize {
                store
                    .push(
                        &[
                            cell as f32 * 7.0 + point as f32 * 0.1,
                            (cell as f32 * 0.31).sin(),
                            (point as f32 * 0.7).cos(),
                        ],
                        &[cell],
                    )
                    .unwrap();
            }
        }
        let config = PrismConfig {
            m_local: 2,
            m_greedy: 4,
            m_random: 0,
            beam_width: 4,
            cross_cell_exact_ranking_limit: 8,
            scan_threshold: 0,
            ..Default::default()
        };
        let index = PrismIndex::build(store, config.clone()).unwrap();
        let targets =
            precompute_cross_cell_targets(&index.medoids, &index.sq8, &config, 4).unwrap();
        assert_eq!(targets.targets_per_source, 4);

        for source_cell in 0..index.tree.cells.len() {
            let mut expected: Vec<(u32, u32)> = (0..index.tree.cells.len())
                .filter(|&target_cell| target_cell != source_cell)
                .map(|target_cell| {
                    (
                        target_cell as u32,
                        distance::ord_key(
                            index
                                .sq8
                                .code_l2(index.medoids[source_cell], index.medoids[target_cell]),
                        ),
                    )
                })
                .collect();
            expected.sort_unstable_by_key(|&(cell, dist)| (dist, cell));
            expected.truncate(4);
            assert_eq!(
                targets.for_source(source_cell),
                expected.iter().map(|&(cell, _)| cell).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn high_cardinality_cross_cell_targets_are_bounded_deterministic_and_diverse() {
        const CELLS: usize = 128;
        let mut store = PointStore::new(3, 1).unwrap();
        for cell in 0..CELLS {
            store
                .push(
                    &[
                        cell as f32,
                        (cell as f32 * 0.17).sin(),
                        (cell as f32 * 0.11).cos(),
                    ],
                    &[cell as u32],
                )
                .unwrap();
        }
        let config = PrismConfig {
            m_local: 1,
            m_greedy: 4,
            m_random: 0,
            t: 1,
            alpha: 0.0,
            beam_width: 8,
            cross_cell_exact_ranking_limit: 8,
            scan_threshold: 0,
            ..Default::default()
        };
        let index = PrismIndex::build(store, config.clone()).unwrap();
        let first = precompute_cross_cell_targets(&index.medoids, &index.sq8, &config, 8).unwrap();
        let second = precompute_cross_cell_targets(&index.medoids, &index.sq8, &config, 8).unwrap();

        assert_eq!(first.targets_per_source, 4);
        assert_eq!(first.cells.len(), CELLS * 4);
        assert_eq!(first.cells, second.cells);
        assert_eq!(bounded_cross_cell_pool_size(CELLS, 4, 8), 16);
        assert!(bounded_cross_cell_pool_size(CELLS, 4, 8) < CELLS - 1);
        for source_cell in 0..CELLS {
            let targets = first.for_source(source_cell);
            assert!(targets.iter().all(|&target| target as usize != source_cell));
            assert_eq!(targets.iter().copied().collect::<HashSet<_>>().len(), 4);
        }

        let source_point = index.medoids[0];
        let foreign_cells: HashSet<u32> = index
            .graph
            .neighbors_unchecked(source_point)
            .iter()
            .map(|&neighbor| index.point_cell[neighbor as usize])
            .filter(|&cell| cell != 0)
            .collect();
        assert!(
            foreign_cells.len() >= 4,
            "bounded target sampling must still expose at least the greedy phase's four distinct foreign cells"
        );
    }

    #[test]
    fn construction_is_reproducible_for_a_fixed_seed() {
        fn fixture() -> PointStore {
            let mut store = PointStore::new(12, 2).unwrap();
            for point in 0..256usize {
                let vector: Vec<f32> = (0..12)
                    .map(|dim| {
                        let value = (point * 12 + dim) as f32;
                        (value * 0.013).sin() + (value * 0.007).cos()
                    })
                    .collect();
                store
                    .push(&vector, &[(point % 4) as u32, ((point / 4) % 4) as u32])
                    .unwrap();
            }
            store
        }

        let config = PrismConfig {
            m_local: 8,
            m_greedy: 4,
            m_random: 4,
            beam_width: 24,
            build_seed: 0x0123_4567_89ab_cdef,
            scan_threshold: 0,
            ..Default::default()
        };
        let single_thread = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let four_threads = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let first = single_thread
            .install(|| PrismIndex::build(fixture(), config.clone()))
            .unwrap();
        let second = four_threads
            .install(|| PrismIndex::build(fixture(), config.clone()))
            .unwrap();

        assert_eq!(first.original_ids, second.original_ids);
        assert_eq!(first.medoids, second.medoids);
        assert_eq!(first.global_medoid, second.global_medoid);
        assert_eq!(first.point_cell, second.point_cell);
        assert_eq!(first.store.vectors, second.store.vectors);
        assert_eq!(first.store.attrs, second.store.attrs);
        assert_eq!(first.graph.offsets(), second.graph.offsets());
        assert_eq!(first.graph.neighbor_ids(), second.graph.neighbor_ids());

        let mut different_config = config;
        different_config.build_seed ^= 0xa5a5_a5a5_a5a5_a5a5;
        let different = PrismIndex::build(fixture(), different_config).unwrap();
        assert_ne!(
            first.graph.neighbor_ids(),
            different.graph.neighbor_ids(),
            "changing build_seed should select a different randomized graph"
        );
    }

    #[test]
    fn graph_expansion_must_be_nonzero() {
        let config = PrismConfig {
            graph_expansion: 0,
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(PrismError::InvalidInput(message))
                if message == "graph_expansion must be greater than zero"
        ));
    }

    #[test]
    fn cross_cell_exact_ranking_limit_has_a_hard_resource_bound() {
        for limit in [0, 4_096, MAX_CROSS_CELL_EXACT_RANKING_LIMIT] {
            PrismConfig {
                cross_cell_exact_ranking_limit: limit,
                ..Default::default()
            }
            .validate()
            .unwrap();
        }
        assert!(matches!(
            PrismConfig {
                cross_cell_exact_ranking_limit: MAX_CROSS_CELL_EXACT_RANKING_LIMIT + 1,
                ..Default::default()
            }
            .validate(),
            Err(PrismError::InvalidInput(message))
                if message.contains("cross_cell_exact_ranking_limit")
        ));
    }

    #[test]
    fn vamana_alpha_must_follow_the_algorithm_domain() {
        for vamana_alpha in [0.0, 0.999, f32::NAN, f32::INFINITY] {
            let config = PrismConfig {
                vamana_alpha,
                ..Default::default()
            };
            assert!(matches!(
                config.validate(),
                Err(PrismError::InvalidInput(message))
                    if message == "vamana_alpha must be finite and at least 1"
            ));
        }
        PrismConfig {
            vamana_alpha: 1.0,
            ..Default::default()
        }
        .validate()
        .unwrap();
    }

    #[test]
    fn extreme_graph_parameters_return_errors_instead_of_panicking_or_looping() {
        fn fixture() -> PointStore {
            let mut store = PointStore::new(1, 1).unwrap();
            store.push(&[0.0], &[0]).unwrap();
            store.push(&[1.0], &[1]).unwrap();
            store
        }

        for config in [
            PrismConfig {
                m_local: usize::MAX,
                ..Default::default()
            },
            PrismConfig {
                m_greedy: usize::MAX,
                ..Default::default()
            },
            PrismConfig {
                beam_width: usize::MAX,
                ..Default::default()
            },
            PrismConfig {
                m_random: usize::MAX - 1,
                ..Default::default()
            },
        ] {
            assert!(matches!(
                PrismIndex::build(fixture(), config),
                Err(PrismError::Overflow(_))
            ));
        }
    }

    #[test]
    fn large_representable_graph_parameters_are_bounded_by_the_population() {
        let mut store = PointStore::new(1, 1).unwrap();
        for point in 0..4u32 {
            store.push(&[point as f32], &[point]).unwrap();
        }
        let index = PrismIndex::build(
            store,
            PrismConfig {
                m_local: 1_000_000,
                m_greedy: 1_000_000,
                m_random: 1_000_000,
                beam_width: 1_000_000,
                t: 1,
                scan_threshold: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(index.store.len(), 4);
        assert!(index.graph.num_edges() <= 12);
    }

    #[test]
    fn persisted_index_parts_roundtrip_search_for_every_metric() {
        let query = [0.2, -0.1, 0.4, 0.3, -0.2, 0.7];
        let filter = Filter::eq(0, 1);
        for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
            let original = persisted_fixture(metric);
            assert_derived_cosine_norm_cache(&original);
            let expected = original.search(&query, &filter, 8, 32).unwrap();
            let rebuilt = reassemble(original).expect("valid persisted parts must reassemble");
            assert_derived_cosine_norm_cache(&rebuilt);
            let actual = rebuilt.search(&query, &filter, 8, 32).unwrap();
            assert_eq!(
                expected
                    .results
                    .iter()
                    .map(|result| (result.id, result.dist.to_bits()))
                    .collect::<Vec<_>>(),
                actual
                    .results
                    .iter()
                    .map(|result| (result.id, result.dist.to_bits()))
                    .collect::<Vec<_>>(),
                "persisted {metric:?} index must preserve search results"
            );
            assert_eq!(expected.diagnostics.plan, actual.diagnostics.plan);
            assert_eq!(
                expected.diagnostics.used_exact_fallback,
                actual.diagnostics.used_exact_fallback
            );
        }
    }

    #[test]
    fn cosine_norm_cache_rebuilds_from_parts_and_preserves_near_ties_bit_exactly() {
        const DIM: usize = 64;
        let base: Vec<f32> = (0..DIM)
            .map(|i| (i as f32 * 0.071).sin() + 0.3 * (i as f32 * 0.113).cos())
            .collect();
        let epsilon = 0.001f32;
        let mut near_one = base.clone();
        near_one[17] += epsilon;
        let mut near_two = base.clone();
        near_two[17] += f32::from_bits(epsilon.to_bits() + 1);
        let rows = vec![
            base.clone(),
            near_one,
            vec![0.0; DIM],
            base.iter().map(|value| -*value).collect(),
            near_two,
            base.iter().map(|value| *value * 2.0).collect(),
        ];
        let point_count = rows.len();
        let vectors = rows.into_iter().flatten().collect();
        // Alternating tuples force build-time point reordering, so this also
        // verifies that cached norms follow internal rather than original IDs.
        let attrs = vec![vec![1, 0, 1, 0, 1, 0]];
        let store = PointStore::from_parts(vectors, DIM, attrs).unwrap();
        let mut index = PrismIndex::build(
            store,
            PrismConfig {
                metric: Metric::Cosine,
                ..Default::default()
            },
        )
        .unwrap();
        assert_derived_cosine_norm_cache(&index);
        let zero_internal = index
            .original_ids
            .iter()
            .position(|&original| original == 2)
            .unwrap();
        assert_eq!(index.cosine_norms.get(zero_internal as u32), 0.0);

        let query: Vec<f32> = base.iter().map(|value| *value * 1.0e12).collect();
        let normalized_query = distance::normalized(&query);
        let mut public: Vec<_> = (0..point_count as u32)
            .map(|id| super::super::search::SearchResult {
                id,
                dist: distance::cosine(&normalized_query, index.store.vector_unchecked(id)),
            })
            .collect();
        public.sort_by(|left, right| {
            left.dist
                .total_cmp(&right.dist)
                .then(left.id.cmp(&right.id))
        });
        public.truncate(point_count / 2);
        let signature = |results: &[super::super::search::SearchResult]| {
            results
                .iter()
                .map(|result| (result.id, result.dist.to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            signature(
                &index
                    .search_exact(&query, &Filter::none(), point_count / 2)
                    .unwrap()
            ),
            signature(&public)
        );

        // The cache is deliberately not a from_parts argument. Poisoning this
        // instance proves restoration derives fresh values from stored rows.
        index.cosine_norms.values.fill(123.0);
        let rebuilt = reassemble(index).unwrap();
        assert_derived_cosine_norm_cache(&rebuilt);
        assert_eq!(
            signature(
                &rebuilt
                    .search_exact(&query, &Filter::none(), point_count / 2)
                    .unwrap()
            ),
            signature(&public)
        );
    }

    #[test]
    fn persisted_index_rejects_non_permutation_original_ids() {
        let index = persisted_fixture(Metric::L2);
        let PrismIndex {
            store,
            tree,
            graph,
            medoids,
            global_medoid,
            point_cell,
            mut original_ids,
            sq8,
            binary,
            cosine_norms: _,
            config,
        } = index;
        original_ids[1] = original_ids[0];
        assert!(matches!(
            PrismIndex::from_parts(
                store,
                tree,
                graph,
                medoids,
                global_medoid,
                point_cell,
                original_ids,
                sq8,
                binary,
                config,
            ),
            Err(PrismError::InvalidFormat(message)) if message.contains("original_ids")
        ));
    }

    #[test]
    fn persisted_local_only_index_rejects_cross_cell_edges() {
        let mut store = PointStore::new(2, 1).unwrap();
        for point in 0..8usize {
            store
                .push(
                    &[point as f32, (point as f32 * 0.3).sin()],
                    &[(point / 4) as u32],
                )
                .unwrap();
        }
        let index = PrismIndex::build(
            store,
            PrismConfig {
                m_local: 2,
                m_greedy: 0,
                m_random: 0,
                beam_width: 4,
                scan_threshold: 0,
                ..Default::default()
            },
        )
        .unwrap();
        let mut adjacency: Vec<Vec<u32>> = (0..index.graph.len() as u32)
            .map(|point| index.graph.neighbors_unchecked(point).to_vec())
            .collect();
        let left = index.tree.cells[0].point_ids[0];
        let right = index.tree.cells[1].point_ids[0];
        adjacency[left as usize].push(right);
        adjacency[left as usize].sort_unstable();
        adjacency[left as usize].dedup();
        let invalid_graph = Graph::from_adj(&adjacency).unwrap();

        let PrismIndex {
            store,
            tree,
            graph: _,
            medoids,
            global_medoid,
            point_cell,
            original_ids,
            sq8,
            binary,
            cosine_norms: _,
            config,
        } = index;
        assert!(matches!(
            PrismIndex::from_parts(
                store,
                tree,
                invalid_graph,
                medoids,
                global_medoid,
                point_cell,
                original_ids,
                sq8,
                binary,
                config,
            ),
            Err(PrismError::InvalidFormat(message)) if message.contains("crosses partition cells")
        ));
    }

    #[test]
    fn persisted_index_rejects_disconnected_global_graph() {
        let index = persisted_fixture(Metric::L2);
        assert!(index.config.uses_global_graph());
        let local_adjacency: Vec<Vec<u32>> = (0..index.graph.len() as u32)
            .map(|point| {
                let cell = index.point_cell[point as usize];
                index
                    .graph
                    .neighbors_unchecked(point)
                    .iter()
                    .copied()
                    .filter(|&neighbor| index.point_cell[neighbor as usize] == cell)
                    .collect()
            })
            .collect();
        let disconnected = Graph::from_adj(&local_adjacency).unwrap();
        let PrismIndex {
            store,
            tree,
            graph: _,
            medoids,
            global_medoid,
            point_cell,
            original_ids,
            sq8,
            binary,
            cosine_norms: _,
            config,
        } = index;
        assert!(matches!(
            PrismIndex::from_parts(
                store,
                tree,
                disconnected,
                medoids,
                global_medoid,
                point_cell,
                original_ids,
                sq8,
                binary,
                config,
            ),
            Err(PrismError::InvalidFormat(message)) if message.contains("reaches")
        ));
    }

    #[test]
    fn persisted_index_rejects_sq8_codes_from_another_point_order() {
        let index = persisted_fixture(Metric::L2);
        let mut codes = index.sq8.codes().to_vec();
        let dim = index.sq8.dim();
        let last = (index.sq8.len() - 1) * dim;
        assert_ne!(&codes[..dim], &codes[last..last + dim]);
        for dimension in 0..dim {
            codes.swap(dimension, last + dimension);
        }
        let invalid_sq8 = SQ8Store::from_parts(
            codes,
            index.sq8.mins().to_vec(),
            index.sq8.scales().to_vec(),
            dim,
        )
        .unwrap();
        let PrismIndex {
            store,
            tree,
            graph,
            medoids,
            global_medoid,
            point_cell,
            original_ids,
            sq8: _,
            binary,
            cosine_norms: _,
            config,
        } = index;
        assert!(matches!(
            PrismIndex::from_parts(
                store,
                tree,
                graph,
                medoids,
                global_medoid,
                point_cell,
                original_ids,
                invalid_sq8,
                binary,
                config,
            ),
            Err(PrismError::InvalidFormat(message)) if message.contains("SQ8 codes")
        ));
    }

    #[test]
    fn persisted_index_rejects_binary_transform_from_another_code_store() {
        let index = persisted_fixture_with_binary(Metric::L2, 2);
        let mut signs = index.binary.signs().to_vec();
        signs[0] = -signs[0];
        let invalid_binary = BinaryStore::from_parts(
            index.binary.codes().to_vec(),
            index.binary.code_words(),
            signs,
            index.binary.block_size(),
        )
        .unwrap();
        let PrismIndex {
            store,
            tree,
            graph,
            medoids,
            global_medoid,
            point_cell,
            original_ids,
            sq8,
            binary: _,
            cosine_norms: _,
            config,
        } = index;
        assert!(matches!(
            PrismIndex::from_parts(
                store,
                tree,
                graph,
                medoids,
                global_medoid,
                point_cell,
                original_ids,
                sq8,
                invalid_binary,
                config,
            ),
            Err(PrismError::InvalidFormat(message)) if message.contains("binary codes")
        ));
    }

    #[test]
    fn persisted_cosine_index_rejects_a_late_unnormalized_store_row() {
        let index = persisted_fixture(Metric::Cosine);
        let PrismIndex {
            mut store,
            tree,
            graph,
            medoids,
            global_medoid,
            point_cell,
            original_ids,
            sq8,
            binary,
            cosine_norms: _,
            config,
        } = index;
        // Corrupting the final row means derivation has already accumulated values,
        // so the error must leave through the partial cache's zeroizing Drop.
        let last_row_start = (store.len - 1) * store.dim;
        for value in &mut store.vectors[last_row_start..] {
            *value *= 2.0;
        }
        assert!(matches!(
            PrismIndex::from_parts(
                store,
                tree,
                graph,
                medoids,
                global_medoid,
                point_cell,
                original_ids,
                sq8,
                binary,
                config,
            ),
            Err(PrismError::InvalidFormat(message)) if message.contains("unit-normalized")
        ));
    }
}
