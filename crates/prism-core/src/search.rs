use super::construct::PrismIndex;
use super::distance;
use super::error::{PrismError, PrismResult};
use super::filter::Filter;
use super::point::validate_f32_distance_domain;

use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::{Duration, Instant};

/// A search result: (point_id, distance).
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// Internal reordered point ID. Use `PrismIndex::original_id` to recover
    /// the caller's insertion-order ID.
    pub id: u32,
    pub dist: f32,
}

/// Selected routing regime for a filter's primary search phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchRegime {
    /// Broad-filter policy. The eligible population is scanned at or below the
    /// applicable configured threshold; larger populations use either the
    /// predicate-aware global graph or the compatible local graphs, as reported
    /// by [`SearchExecution`].
    High,
    /// Intermediate-selectivity policy. Eligible populations within an
    /// applicable threshold may use a total exact/binary scan; populations
    /// above every applicable threshold use graph traversal.
    Mid,
    /// Exhaustive search over eligible cells for selective filters.
    Low,
    /// Correctness-first exhaustive path used for inner product.
    Exact,
}

/// Physical work used by the primary search phase.
///
/// This is separate from [`SearchRegime`], which describes the selectivity
/// policy decision. In particular, a HIGH or MID query may use an exact or
/// binary-prefilter scan when its total eligible population is small.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchExecution {
    /// No candidate work was required, for example when `k == 0`.
    NoWork,
    /// Every eligible point was scored with the configured exact metric.
    ExactScan,
    /// One eligible-only binary-code prefilter budget was applied across the
    /// complete matching population before SQ8 selection and exact reranking.
    BinaryPrefilterScan,
    /// Each compatible cell's local graph was traversed and the candidates were
    /// merged. No cross-cell graph edge was followed.
    LocalGraph,
    /// The predicate-aware whole-index graph was traversed from one entry.
    GlobalGraph,
}

/// Observable filter-planning facts used for diagnostics and benchmarks.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchPlan {
    pub regime: SearchRegime,
    pub matching_cells: usize,
    pub matching_points: usize,
    pub selectivity: f32,
}

/// Per-query evidence separating the planner regime, physical primary work,
/// and any exact underfill rescue.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchDiagnostics {
    pub plan: SearchPlan,
    /// Physical work used by the primary phase, before any exact fallback.
    pub primary_execution: SearchExecution,
    pub primary_result_count: usize,
    pub used_exact_fallback: bool,
    /// Successful `search` call duration, including validation, planning,
    /// primary traversal, optional fallback, and result diagnostics.
    pub total_elapsed: Duration,
    pub primary_elapsed: Duration,
    pub fallback_elapsed: Duration,
    pub target_result_count: usize,
    pub final_result_count: usize,
    pub complete: bool,
}

/// Search results together with the route and fallback facts that produced them.
#[derive(Clone, Debug)]
pub struct SearchOutcome {
    pub results: Vec<SearchResult>,
    pub diagnostics: SearchDiagnostics,
}

/// Bitset for O(1) visited tracking, sized to the cell for L1 cache locality.
struct Bitset {
    bits: Vec<u64>,
}

impl Bitset {
    fn new(n: usize) -> Self {
        Self {
            bits: vec![0u64; n.div_ceil(64)],
        }
    }

    /// Returns true if the bit was newly set (not previously visited).
    #[inline]
    fn insert(&mut self, i: u32) -> bool {
        let word = i as usize >> 6;
        let bit = 1u64 << (i & 63);
        if self.bits[word] & bit != 0 {
            false
        } else {
            self.bits[word] |= bit;
            true
        }
    }

    /// Check if a bit is set without modifying the bitset.
    #[inline]
    fn contains(&self, i: u32) -> bool {
        let word = i as usize >> 6;
        let bit = 1u64 << (i & 63);
        self.bits[word] & bit != 0
    }
}

/// Prefetch into L1.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
#[inline]
unsafe fn prefetch_t0(ptr: *const u8) {
    std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
}

/// Software prefetch hint.
#[inline(always)]
fn prefetch_read(ptr: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        prefetch_t0(ptr);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = ptr;
}

/// Prefetch `len` bytes starting at `ptr`.
#[inline(always)]
fn prefetch_range(ptr: *const u8, len: usize) {
    let mut offset = 0;
    while offset < len {
        prefetch_read(unsafe { ptr.add(offset) });
        offset += 64;
    }
}

/// Ordered f32 wrapper for use in BinaryHeap.
#[derive(Clone, Copy)]
struct OrdF32(f32);

impl PartialEq for OrdF32 {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_cmp(&other.0).is_eq()
    }
}
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Insert into the exact-distance top-k heap while retaining the historical
/// strict-distance replacement rule. Point IDs are visited in reordered ID
/// order, so equal-distance boundary ties continue to keep the lower IDs.
#[inline]
fn heap_insert_exact(heap: &mut BinaryHeap<(OrdF32, u32)>, dist: f32, id: u32, cap: usize) {
    if heap.len() < cap {
        heap.push((OrdF32(dist), id));
    } else if let Some(&(OrdF32(worst), _)) = heap.peek() {
        if dist < worst {
            heap.pop();
            heap.push((OrdF32(dist), id));
        }
    }
}

/// Select and order a dense exact scan by the public `(distance, point_id)`
/// contract. Selecting once avoids maintaining an almost-full heap when `k`
/// is a large fraction of the eligible population.
fn finish_dense_exact_scan(mut results: Vec<SearchResult>, k: usize) -> Vec<SearchResult> {
    let order = |left: &SearchResult, right: &SearchResult| {
        left.dist
            .total_cmp(&right.dist)
            .then(left.id.cmp(&right.id))
    };
    if results.len() > k {
        results.select_nth_unstable_by(k, order);
        results.truncate(k);
    }
    results.sort_unstable_by(order);
    results
}

/// Insert into a bounded max-heap of (u32_dist, point_id), keeping only the `cap` smallest.
#[inline]
fn heap_insert_sq8(heap: &mut BinaryHeap<(u32, u32)>, dist: u32, id: u32, cap: usize) {
    if heap.len() < cap {
        heap.push((dist, id));
    } else if let Some(&(worst, _)) = heap.peek() {
        if dist < worst {
            heap.pop();
            heap.push((dist, id));
        }
    }
}

/// Insert a wide Hamming distance without truncating codes above u32::MAX bits.
#[inline]
fn heap_insert_hamming(heap: &mut BinaryHeap<(u64, u32)>, dist: u64, id: u32, cap: usize) {
    if heap.len() < cap {
        heap.push((dist, id));
    } else if let Some(&(worst, _)) = heap.peek() {
        if dist < worst {
            heap.pop();
            heap.push((dist, id));
        }
    }
}

impl PrismIndex {
    /// Apply the caller-visible graph quality/work multiplier. Local-cell
    /// search narrows this frontier back to `ef` candidates; global/MID search
    /// can exact-rerank the expanded set. Neither path becomes an exact scan.
    #[inline]
    fn expanded_graph_ef(&self, ef: usize, population: usize) -> usize {
        ef.saturating_mul(self.config.graph_expansion)
            .min(population)
            .max(1)
    }

    /// Convert an internal reordered point ID to its insertion-order ID.
    /// Returns `None` when `internal_id` is outside this index.
    #[inline]
    pub fn original_id(&self, internal_id: u32) -> Option<u32> {
        self.original_ids.get(internal_id as usize).copied()
    }

    /// Plan a filter using exact cell-posting cardinalities.
    pub fn plan_filter(&self, filter: &Filter) -> PrismResult<SearchPlan> {
        filter.validate(self.store.k())?;
        let cells = self.tree.filter_cells_unchecked(filter.constraints());
        let matching_points = self.tree.count_points_unchecked(&cells).ok_or_else(|| {
            PrismError::Overflow("matching point count exceeds addressable memory".into())
        })?;
        let selectivity = matching_points as f32 / self.store.len as f32;
        let regime = if self.config.metric == distance::Metric::InnerProduct {
            SearchRegime::Exact
        } else if selectivity >= self.config.sigma_high {
            SearchRegime::High
        } else if selectivity > self.config.sigma_low {
            SearchRegime::Mid
        } else {
            SearchRegime::Low
        };
        Ok(SearchPlan {
            regime,
            matching_cells: cells.len(),
            matching_points,
            selectivity,
        })
    }

    /// Filtered k-NN search with automatic regime selection.
    ///
    /// HIGH/MID graph or binary-prefilter execution remains approximate; a
    /// HIGH/MID exact scan is exact. Exact fallback repairs an
    /// underfilled result count, not a full-but-wrong approximate neighbor set;
    /// use [`Self::search_exact`] or measure filtered Recall@k when exact
    /// ranking matters. The returned diagnostics expose the selectivity regime,
    /// the physical primary execution, and any fallback work.
    pub fn search(
        &self,
        query: &[f32],
        filter: &Filter,
        k: usize,
        ef: usize,
    ) -> PrismResult<SearchOutcome> {
        let search_start = Instant::now();
        if query.len() != self.store.dim {
            return Err(PrismError::InvalidInput(format!(
                "query dimension {} does not match index dimension {}",
                query.len(),
                self.store.dim
            )));
        }
        validate_f32_distance_domain(query, self.store.dim, "query")?;
        let plan = self.plan_filter(filter)?;
        let target_result_count = k.min(plan.matching_points);
        if target_result_count == 0 {
            return Ok(SearchOutcome {
                results: Vec::new(),
                diagnostics: SearchDiagnostics {
                    plan,
                    primary_execution: SearchExecution::NoWork,
                    primary_result_count: 0,
                    used_exact_fallback: false,
                    total_elapsed: search_start.elapsed(),
                    primary_elapsed: Duration::ZERO,
                    fallback_elapsed: Duration::ZERO,
                    target_result_count,
                    final_result_count: 0,
                    complete: true,
                },
            });
        }

        // Match the build-time Cosine normalization so code-space distances
        // approximate the intended metric. Cosine is scale-invariant in both
        // arguments, so reported distances are unchanged.
        let normalized;
        let query = if self.config.metric == distance::Metric::Cosine {
            normalized = distance::normalized(query);
            normalized.as_slice()
        } else {
            query
        };

        let cell_indices = self.tree.filter_cells_unchecked(filter.constraints());
        let ef = ef.max(k);
        let primary_start = Instant::now();
        let below_general_scan_threshold =
            self.config.scan_threshold > 0 && plan.matching_points <= self.config.scan_threshold;
        let proper_multi_cell_subset =
            plan.matching_cells > 1 && plan.matching_cells < self.tree.cells.len();
        let below_multi_cell_scan_threshold = self.config.multi_cell_scan_threshold > 0
            && proper_multi_cell_subset
            && plan.matching_points <= self.config.multi_cell_scan_threshold;
        let below_scan_threshold = below_general_scan_threshold || below_multi_cell_scan_threshold;
        let (mut results, primary_execution) = match plan.regime {
            SearchRegime::Exact | SearchRegime::Low => (
                self.regime_low(query, &cell_indices, k),
                SearchExecution::ExactScan,
            ),
            SearchRegime::High | SearchRegime::Mid if below_scan_threshold => {
                self.scan_eligible_population(query, &cell_indices, plan.matching_points, k, ef)?
            }
            SearchRegime::High | SearchRegime::Mid => {
                self.regime_graph(query, &cell_indices, k, ef)
            }
        };
        let primary_elapsed = primary_start.elapsed();
        let primary_result_count = results.len();

        // Approximate traversal is allowed to miss the exact neighbors, but it
        // must not silently underfill when enough eligible points exist.
        let used_exact_fallback = !matches!(plan.regime, SearchRegime::Exact | SearchRegime::Low)
            && results.len() < target_result_count;
        let mut fallback_elapsed = Duration::ZERO;
        if used_exact_fallback {
            let fallback_start = Instant::now();
            results = self.regime_low(query, &cell_indices, k);
            fallback_elapsed = fallback_start.elapsed();
        }
        debug_assert!(results
            .iter()
            .all(|result| filter.matches(&self.store, result.id)));
        let final_result_count = results.len();
        let total_elapsed = search_start.elapsed();
        Ok(SearchOutcome {
            results,
            diagnostics: SearchDiagnostics {
                plan,
                primary_execution,
                primary_result_count,
                used_exact_fallback,
                total_elapsed,
                primary_elapsed,
                fallback_elapsed,
                target_result_count,
                final_result_count,
                complete: final_result_count == target_result_count,
            },
        })
    }

    /// Exact filtered top-k baseline over every eligible point.
    pub fn search_exact(
        &self,
        query: &[f32],
        filter: &Filter,
        k: usize,
    ) -> PrismResult<Vec<SearchResult>> {
        if query.len() != self.store.dim {
            return Err(PrismError::InvalidInput(format!(
                "query dimension {} does not match index dimension {}",
                query.len(),
                self.store.dim
            )));
        }
        validate_f32_distance_domain(query, self.store.dim, "query")?;
        filter.validate(self.store.k())?;
        if k == 0 {
            return Ok(Vec::new());
        }

        let normalized;
        let query = if self.config.metric == distance::Metric::Cosine {
            normalized = distance::normalized(query);
            normalized.as_slice()
        } else {
            query
        };
        let cell_indices = self.tree.filter_cells_unchecked(filter.constraints());
        let results = self.regime_low(query, &cell_indices, k);
        debug_assert!(results
            .iter()
            .all(|result| filter.matches(&self.store, result.id)));
        Ok(results)
    }

    /// Heap-ordered candidate distance from the query to point `p`. L2 and
    /// build-normalized Cosine use scale-aware asymmetric SQ8 distance;
    /// InnerProduct uses the exact f32 metric (and is routed to exhaustive
    /// search by the public entry points).
    #[inline]
    fn cand_dist(&self, query: &[f32], p: u32) -> u32 {
        match self.config.metric {
            distance::Metric::L2 | distance::Metric::Cosine => {
                distance::ord_key(self.sq8.asymmetric_l2(query, p))
            }
            distance::Metric::InnerProduct => distance::ord_key(distance::distance(
                query,
                self.store.vector_unchecked(p),
                distance::Metric::InnerProduct,
            )),
        }
    }

    /// Exact-rerank candidate IDs using the configured public metric. Cosine
    /// shares one query norm and the private per-row cache across every
    /// physical route (binary prefilter, local graph, and global/MID graph).
    fn exact_rerank<I>(&self, query: &[f32], candidates: I, k: usize) -> Vec<SearchResult>
    where
        I: IntoIterator<Item = u32>,
    {
        let mut results: Vec<SearchResult> = match self.config.metric {
            distance::Metric::Cosine => {
                debug_assert_eq!(self.cached_cosine_norm_count(), self.store.len);
                let query_norm = distance::cosine_norm(query);
                candidates
                    .into_iter()
                    .map(|id| SearchResult {
                        id,
                        dist: distance::cosine_with_norms(
                            query,
                            self.store.vector_unchecked(id),
                            query_norm,
                            self.cached_cosine_norm(id),
                        ),
                    })
                    .collect()
            }
            metric => candidates
                .into_iter()
                .map(|id| SearchResult {
                    id,
                    dist: distance::distance(query, self.store.vector_unchecked(id), metric),
                })
                .collect(),
        };
        results.sort_by(|left, right| {
            left.dist
                .total_cmp(&right.dist)
                .then(left.id.cmp(&right.id))
        });
        results.truncate(k);
        results
    }

    /// In-cell sub-slice of one sorted full-graph CSR row. Reordering makes a
    /// cell's point IDs contiguous, so local traversal needs no duplicate CSR
    /// and does not inspect cross-cell neighbors.
    #[inline]
    fn local_neighbors(&self, point: u32) -> &[u32] {
        let cell_idx = self.point_cell[point as usize] as usize;
        let points = &self.tree.cells[cell_idx].point_ids;
        let first = points[0];
        let last = points[points.len() - 1];
        let neighbors = self.graph.neighbors_unchecked(point);
        let start = neighbors.partition_point(|&neighbor| neighbor < first);
        let end = neighbors.partition_point(|&neighbor| neighbor <= last);
        &neighbors[start..end]
    }

    /// Scan the entire eligible population with one query-wide budget.
    ///
    /// The optional binary prefilter is intentionally global rather than
    /// multiplying its budget by the number of compatible cells.
    fn scan_eligible_population(
        &self,
        query: &[f32],
        cell_indices: &[usize],
        matching_points: usize,
        k: usize,
        ef: usize,
    ) -> PrismResult<(Vec<SearchResult>, SearchExecution)> {
        debug_assert!(matching_points > 0);
        let rerank_budget = self.config.binary_rerank.saturating_mul(ef);
        if self.config.binary_rerank == 0 || matching_points <= rerank_budget {
            return Ok((
                self.regime_low(query, cell_indices, k),
                SearchExecution::ExactScan,
            ));
        }

        let query_binary = self.binary.encode_query(query)?;
        let mut hamming_heap: BinaryHeap<(u64, u32)> = BinaryHeap::new();
        for &cell_idx in cell_indices {
            for &point in &self.tree.cells[cell_idx].point_ids {
                let hamming = distance::hamming(&query_binary, self.binary.code_unchecked(point));
                heap_insert_hamming(&mut hamming_heap, hamming, point, rerank_budget);
            }
        }

        let mut candidate_heap: BinaryHeap<(u32, u32)> = BinaryHeap::new();
        for (_, point) in hamming_heap {
            heap_insert_sq8(&mut candidate_heap, self.cand_dist(query, point), point, ef);
        }
        let results = self.exact_rerank(query, candidate_heap.into_iter().map(|(_, id)| id), k);
        Ok((results, SearchExecution::BinaryPrefilterScan))
    }

    /// HIGH/MID graph execution after the eligible-population scan decision.
    fn regime_graph(
        &self,
        query: &[f32],
        cell_indices: &[usize],
        k: usize,
        ef: usize,
    ) -> (Vec<SearchResult>, SearchExecution) {
        debug_assert!(!cell_indices.is_empty());
        if self.config.uses_global_graph() && cell_indices.len() > 1 {
            (
                self.regime_mid(
                    query,
                    cell_indices,
                    k,
                    self.expanded_graph_ef(ef, self.store.len),
                ),
                SearchExecution::GlobalGraph,
            )
        } else {
            (
                self.regime_local_graph(query, cell_indices, k, ef),
                SearchExecution::LocalGraph,
            )
        }
    }

    /// Traverse every compatible local graph and merge one query-wide SQ8
    /// candidate heap before exact reranking.
    fn regime_local_graph(
        &self,
        query: &[f32],
        cell_indices: &[usize],
        k: usize,
        ef: usize,
    ) -> Vec<SearchResult> {
        let mut merged: BinaryHeap<(u32, u32)> = BinaryHeap::new();

        // Visit cells nearest-medoid-first so the global heap tightens early.
        let mut ranked: Vec<(usize, u32)> = cell_indices
            .iter()
            .map(|&cell_idx| (cell_idx, self.cand_dist(query, self.medoids[cell_idx])))
            .collect();
        ranked.sort_unstable_by_key(|&(_, distance)| distance);

        for (cell_idx, _) in ranked {
            for (candidate_distance, point) in self.search_cell_graph(query, cell_idx, ef) {
                heap_insert_sq8(&mut merged, candidate_distance, point, ef);
            }
        }

        self.exact_rerank(query, merged.into_iter().map(|(_, id)| id), k)
    }

    /// Predicate-aware whole-index routing for HIGH/MID graph execution.
    /// Traverses the full graph, using non-matching nodes as bridges when their
    /// bridge score exceeds tau. SQ8 traversal, f32 rerank.
    fn regime_mid(
        &self,
        query: &[f32],
        compatible_cells: &[usize],
        k: usize,
        ef: usize,
    ) -> Vec<SearchResult> {
        if compatible_cells.is_empty() {
            return Vec::new();
        }

        let n_cells = self.tree.cells.len();
        let mut cell_match = vec![false; n_cells];
        for &ci in compatible_cells {
            cell_match[ci] = true;
        }

        let entry = if compatible_cells.len() == n_cells {
            // Singleton tuples make a full medoid scan pointless: the global medoid
            // is already construction's entry point for unfiltered traversal.
            self.global_medoid
        } else {
            compatible_cells
                .iter()
                .map(|&ci| {
                    let d = self.cand_dist(query, self.medoids[ci]);
                    (d, self.medoids[ci])
                })
                .min_by_key(|&(d, _)| d)
                .unwrap()
                .1
        };

        let entry_dist = self.cand_dist(query, entry);

        let mut visited = Bitset::new(self.store.len);
        visited.insert(entry);

        let mut candidates: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(u32, u32)> = BinaryHeap::new();

        candidates.push(Reverse((entry_dist, entry)));
        results.push((entry_dist, entry));

        let bridge_budget = (self.config.beta * ef as f32) as usize;
        let mut bridges_used = 0usize;
        let epsilon_factor = ((1.0 + self.config.epsilon) * (1.0 + self.config.epsilon)) as f64;

        let n_f: usize = compatible_cells
            .iter()
            .map(|&ci| self.tree.cells[ci].point_ids.len())
            .sum();
        let sigma = n_f as f32 / self.store.len as f32;
        let tau = sigma / (1.0 + sigma);

        while let Some(Reverse((d, c))) = candidates.pop() {
            if results.len() >= ef {
                if let Some(&(worst, _)) = results.peek() {
                    let d_value = distance::from_ord_key(d) as f64;
                    let worst_value = distance::from_ord_key(worst) as f64;
                    if d_value > worst_value * epsilon_factor {
                        break;
                    }
                }
            }

            let neighbors = self.graph.neighbors_unchecked(c);
            let sq8_dim = self.store.dim;

            let mut unvisited_buf: Vec<u32> = Vec::with_capacity(neighbors.len());
            for &w in neighbors {
                if visited.insert(w) {
                    unvisited_buf.push(w);
                    prefetch_range(self.sq8.code_unchecked(w).as_ptr(), sq8_dim);
                }
            }

            for &w in &unvisited_buf {
                let wd = self.cand_dist(query, w);
                let w_cell = self.point_cell[w as usize];

                if cell_match[w_cell as usize] {
                    heap_insert_sq8(&mut results, wd, w, ef);
                    candidates.push(Reverse((wd, w)));
                } else {
                    let w_neighbors = self.graph.neighbors_unchecked(w);
                    if !w_neighbors.is_empty() {
                        let matching_unvisited = w_neighbors
                            .iter()
                            .filter(|&&u| {
                                cell_match[self.point_cell[u as usize] as usize]
                                    && !visited.contains(u)
                            })
                            .count();
                        let fraction = matching_unvisited as f32 / w_neighbors.len() as f32;

                        let radius = results
                            .peek()
                            .map_or(1.0f32, |&(worst, _)| distance::from_ord_key(worst));
                        let candidate_distance = distance::from_ord_key(wd);
                        let bridge_score =
                            fraction / (1.0 + candidate_distance / radius.max(f32::EPSILON));

                        if bridges_used < bridge_budget && bridge_score > tau {
                            candidates.push(Reverse((wd, w)));
                            bridges_used += 1;
                        }
                    }
                }
            }
        }

        self.exact_rerank(query, results.into_iter().map(|(_, id)| id), k)
    }

    /// Code-space beam search within a cell's local graph. Returns (id, cand_dist) pairs.
    fn greedy_search_cell_sq8(&self, query: &[f32], cell_idx: usize, ef: usize) -> Vec<(u32, u32)> {
        let pts = &self.tree.cells[cell_idx].point_ids;
        let base = pts[0];
        let sq8_dim = self.store.dim;

        let entry = self.medoids[cell_idx];
        let entry_dist = self.cand_dist(query, entry);

        let mut visited = Bitset::new(pts.len());
        visited.insert(entry - base);

        let mut candidates: BinaryHeap<Reverse<(u32, u32)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(u32, u32)> = BinaryHeap::new();
        let mut unvisited: Vec<u32> = Vec::with_capacity(32);

        candidates.push(Reverse((entry_dist, entry)));
        results.push((entry_dist, entry));

        while let Some(Reverse((d, c))) = candidates.pop() {
            if results.len() >= ef {
                if let Some(&(worst, _)) = results.peek() {
                    if d > worst {
                        break;
                    }
                }
            }

            unvisited.clear();
            for &w in self.local_neighbors(c) {
                if visited.insert(w - base) {
                    unvisited.push(w);
                    prefetch_range(self.sq8.code_unchecked(w).as_ptr(), sq8_dim);
                }
            }

            for &w in &unvisited {
                let wd = self.cand_dist(query, w);
                if results.len() < ef {
                    candidates.push(Reverse((wd, w)));
                    results.push((wd, w));
                } else if let Some(&(worst, _)) = results.peek() {
                    if wd < worst {
                        results.pop();
                        results.push((wd, w));
                        candidates.push(Reverse((wd, w)));
                    }
                }
            }
        }

        results
            .into_vec()
            .into_iter()
            .map(|(d, id)| (id, d))
            .collect()
    }

    /// REGIME_LOW: brute-force within compatible cells for very selective filters.
    fn regime_low(&self, query: &[f32], cell_indices: &[usize], k: usize) -> Vec<SearchResult> {
        if k == 0 {
            return Vec::new();
        }
        let matching_points: usize = cell_indices
            .iter()
            .map(|&cell_idx| self.tree.cells[cell_idx].point_ids.len())
            .sum();
        let mut heap: BinaryHeap<(OrdF32, u32)> = BinaryHeap::new();
        match self.config.metric {
            distance::Metric::Cosine => {
                // Reuse each row's derived exact f64 norm rather than assuming
                // normalized f32 rows have norm 1.0, which could reorder close ties.
                debug_assert_eq!(self.cached_cosine_norm_count(), self.store.len);
                let query_norm = distance::cosine_norm(query);
                if k >= matching_points.saturating_sub(k) {
                    let mut results = Vec::with_capacity(matching_points);
                    for &ci in cell_indices {
                        for &p in &self.tree.cells[ci].point_ids {
                            results.push(SearchResult {
                                id: p,
                                dist: distance::cosine_with_norms(
                                    query,
                                    self.store.vector_unchecked(p),
                                    query_norm,
                                    self.cached_cosine_norm(p),
                                ),
                            });
                        }
                    }
                    return finish_dense_exact_scan(results, k);
                }
                for &ci in cell_indices {
                    for &p in &self.tree.cells[ci].point_ids {
                        let dist = distance::cosine_with_norms(
                            query,
                            self.store.vector_unchecked(p),
                            query_norm,
                            self.cached_cosine_norm(p),
                        );
                        heap_insert_exact(&mut heap, dist, p, k);
                    }
                }
            }
            metric => {
                for &ci in cell_indices {
                    for &p in &self.tree.cells[ci].point_ids {
                        // Every point in a selected full-tuple cell already satisfies
                        // the predicate, so re-evaluating it per scored point would
                        // only repeat attribute work.
                        let dist =
                            distance::distance(query, self.store.vector_unchecked(p), metric);
                        heap_insert_exact(&mut heap, dist, p, k);
                    }
                }
            }
        }
        let mut results: Vec<SearchResult> = heap
            .into_iter()
            .map(|(OrdF32(d), id)| SearchResult { id, dist: d })
            .collect();
        results.sort_by(|a, b| a.dist.total_cmp(&b.dist).then(a.id.cmp(&b.id)));
        results
    }

    /// Batched filtered search with genuine per-query diagnostics. Queries run
    /// independently in Rayon so each outcome's primary/fallback timings and
    /// fallback flag describe that query rather than an aggregate batch phase.
    pub fn batch_search(
        &self,
        queries: &[f32],
        filters: &[Filter],
        nq: usize,
        k: usize,
        ef: usize,
    ) -> PrismResult<Vec<SearchOutcome>> {
        let expected_values = nq.checked_mul(self.store.dim).ok_or_else(|| {
            PrismError::Overflow("query batch shape exceeds addressable memory".into())
        })?;
        if queries.len() != expected_values {
            return Err(PrismError::InvalidInput(format!(
                "query data length {} does not equal nq * dimension ({expected_values})",
                queries.len()
            )));
        }
        if filters.len() != nq {
            return Err(PrismError::InvalidInput(format!(
                "filter count {} does not equal query count {nq}",
                filters.len()
            )));
        }
        validate_f32_distance_domain(queries, self.store.dim, "query")?;

        (0..nq)
            .into_par_iter()
            .map(|query_idx| {
                let start = query_idx * self.store.dim;
                let query = &queries[start..start + self.store.dim];
                self.search(query, &filters[query_idx], k, ef)
            })
            .collect()
    }

    /// Search one compatible cell's local graph, returning code-distance/id
    /// pairs narrowed to the caller's candidate budget.
    fn search_cell_graph(&self, query: &[f32], cell_idx: usize, ef: usize) -> Vec<(u32, u32)> {
        let pts = &self.tree.cells[cell_idx].point_ids;
        let mut heap: BinaryHeap<(u32, u32)> = BinaryHeap::new();

        // Explore a wider frontier than the final rerank heap, bounding the
        // caller-visible multiplier by this cell's population.
        let ef_cell = self.expanded_graph_ef(ef, pts.len());
        let local = self.greedy_search_cell_sq8(query, cell_idx, ef_cell);
        for (id, dist) in local {
            heap_insert_sq8(&mut heap, dist, id, ef);
        }
        heap.into_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::super::construct::{PrismConfig, PrismIndex};
    use super::super::distance;
    use super::super::error::PrismError;
    use super::super::filter::Filter;
    use super::super::graph::Graph;
    use super::super::point::PointStore;
    use super::{SearchExecution, SearchRegime};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;

    fn build_test_index() -> PrismIndex {
        let mut store = PointStore::new(2, 1).unwrap();
        for i in 0..10 {
            let x = (i as f32) * 0.1;
            let attr = if i < 5 { 0 } else { 1 };
            store.push(&[x, x], &[attr]).unwrap();
        }
        let config = PrismConfig {
            m_local: 4,
            m_greedy: 2,
            m_random: 4,
            t: 1,
            alpha: 0.0,
            beam_width: 10,
            ..Default::default()
        };
        PrismIndex::build(store, config).unwrap()
    }

    #[test]
    fn test_search_no_filter() {
        let index = build_test_index();
        let outcome = index.search(&[0.25, 0.25], &Filter::none(), 3, 10).unwrap();
        assert_eq!(
            outcome.diagnostics.primary_execution,
            SearchExecution::ExactScan
        );
        let results = outcome.results;
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.dist >= 0.0);
        }
    }

    #[test]
    fn filtered_high_diagnostics_distinguish_total_scan_and_graph_work() {
        let mut store = PointStore::new(2, 1).unwrap();
        for (attribute, count) in [(0u32, 4usize), (1, 12), (2, 20)] {
            for point in 0..count {
                store
                    .push(
                        &[attribute as f32 * 10.0 + point as f32 * 0.01, point as f32],
                        &[attribute],
                    )
                    .unwrap();
            }
        }
        let mut index = PrismIndex::build(
            store,
            PrismConfig {
                m_local: 4,
                m_greedy: 1,
                m_random: 0,
                t: 1,
                beam_width: 12,
                sigma_high: 0.10,
                sigma_low: 0.01,
                scan_threshold: 15,
                multi_cell_scan_threshold: 0,
                ..Default::default()
            },
        )
        .unwrap();
        let filter = Filter::new(vec![(0, vec![0, 1])]);
        assert_eq!(
            index.plan_filter(&filter).unwrap().regime,
            SearchRegime::High
        );

        // The two compatible cells contain 16 points in total. The threshold
        // is inclusive and the exact primary uses one query-wide top-k heap.
        index.config.scan_threshold = 16;
        let scanned = index.search(&[0.0, 0.0], &filter, 5, 8).unwrap();
        assert_eq!(
            scanned.diagnostics.primary_execution,
            SearchExecution::ExactScan
        );
        assert!(!scanned.diagnostics.used_exact_fallback);
        let exact = index.search_exact(&[0.0, 0.0], &filter, 5).unwrap();
        assert_eq!(
            scanned
                .results
                .iter()
                .map(|result| (result.id, result.dist))
                .collect::<Vec<_>>(),
            exact
                .iter()
                .map(|result| (result.id, result.dist))
                .collect::<Vec<_>>()
        );
        index.config.scan_threshold = 15;
        let globally_graphed = index.search(&[0.0, 0.0], &filter, 1, 4).unwrap();
        assert_eq!(
            globally_graphed.diagnostics.primary_execution,
            SearchExecution::GlobalGraph
        );
        assert!(!globally_graphed.diagnostics.used_exact_fallback);

        // The additional fragmented-population threshold is independently
        // inclusive for a proper subset of more than one cell.
        index.config.scan_threshold = 0;
        index.config.multi_cell_scan_threshold = 16;
        let multi_cell_scanned = index.search(&[0.0, 0.0], &filter, 5, 8).unwrap();
        assert_eq!(
            multi_cell_scanned.diagnostics.primary_execution,
            SearchExecution::ExactScan
        );
        assert_eq!(
            multi_cell_scanned
                .results
                .iter()
                .map(|result| (result.id, result.dist))
                .collect::<Vec<_>>(),
            exact
                .iter()
                .map(|result| (result.id, result.dist))
                .collect::<Vec<_>>()
        );
        index.config.multi_cell_scan_threshold = 15;
        assert_eq!(
            index
                .search(&[0.0, 0.0], &filter, 1, 4)
                .unwrap()
                .diagnostics
                .primary_execution,
            SearchExecution::GlobalGraph
        );

        // Even an unbounded multi-cell threshold does not apply to all cells
        // or to exactly one compatible cell.
        index.config.multi_cell_scan_threshold = usize::MAX;
        assert_eq!(
            index
                .search(&[0.0, 0.0], &Filter::none(), 1, 4)
                .unwrap()
                .diagnostics
                .primary_execution,
            SearchExecution::GlobalGraph
        );
        assert_eq!(
            index
                .search(&[20.0, 0.0], &Filter::eq(0, 2), 1, 4)
                .unwrap()
                .diagnostics
                .primary_execution,
            SearchExecution::LocalGraph
        );

        // Disable global routing in this test-only mutable fixture. Both cells
        // are then traversed locally, but the query still has one primary path.
        index.config.multi_cell_scan_threshold = 0;
        index.config.m_greedy = 0;
        let locally_graphed = index.search(&[0.0, 0.0], &filter, 1, 4).unwrap();
        assert_eq!(
            locally_graphed.diagnostics.primary_execution,
            SearchExecution::LocalGraph
        );
        assert!(!locally_graphed.diagnostics.used_exact_fallback);

        let no_work = index.search(&[0.0, 0.0], &filter, 0, 4).unwrap();
        assert_eq!(
            no_work.diagnostics.primary_execution,
            SearchExecution::NoWork
        );

        let empty = index.search(&[0.0, 0.0], &Filter::eq(0, 99), 1, 4).unwrap();
        assert_eq!(empty.diagnostics.primary_execution, SearchExecution::NoWork);
    }

    #[test]
    fn local_graph_execution_slices_cross_edges_out_of_the_single_csr() {
        let store = PointStore::from_parts(vec![0.0, 1.0, 100.0], 1, vec![vec![0, 0, 1]]).unwrap();
        let mut index = PrismIndex::build(
            store,
            PrismConfig {
                m_local: 1,
                m_greedy: 1,
                m_random: 0,
                beam_width: 4,
                sigma_high: 0.5,
                sigma_low: 0.0,
                scan_threshold: 0,
                ..Default::default()
            },
        )
        .unwrap();

        // Row 0 holds in-cell neighbor 1 and cross-cell neighbor 2, and the CSR is
        // sorted, so local_neighbors must return only the contiguous in-cell slice.
        index.graph = Graph::from_adj(&[vec![1, 2], vec![0], vec![0]]).unwrap();
        index.medoids[0] = 0;
        assert_eq!(index.local_neighbors(0), &[1]);

        let filter = Filter::eq(0, 0);
        let outcome = index.search(&[100.0], &filter, 2, 3).unwrap();
        assert_eq!(
            outcome.diagnostics.primary_execution,
            SearchExecution::LocalGraph
        );
        assert!(!outcome.diagnostics.used_exact_fallback);
        assert_eq!(outcome.results.len(), 2);
        assert!(outcome
            .results
            .iter()
            .all(|result| filter.matches(&index.store, result.id)));
    }

    #[test]
    fn unfiltered_scan_threshold_is_inclusive_and_zero_forces_graph() {
        fn build(point_count: usize, scan_threshold: usize) -> PrismIndex {
            let mut store = PointStore::new(1, 1).unwrap();
            for point in 0..point_count {
                store.push(&[point as f32], &[0]).unwrap();
            }
            PrismIndex::build(
                store,
                PrismConfig {
                    m_local: 2,
                    m_greedy: 0,
                    m_random: 0,
                    beam_width: 4,
                    scan_threshold,
                    ..Default::default()
                },
            )
            .unwrap()
        }

        for (point_count, expected) in [
            (3, SearchExecution::ExactScan),
            (4, SearchExecution::ExactScan),
            (5, SearchExecution::LocalGraph),
        ] {
            let index = build(point_count, 4);
            let outcome = index.search(&[3.25], &Filter::none(), 2, 4).unwrap();
            assert_eq!(outcome.diagnostics.primary_execution, expected);
            assert!(!outcome.diagnostics.used_exact_fallback);
            if expected == SearchExecution::ExactScan {
                let exact = index.search_exact(&[3.25], &Filter::none(), 2).unwrap();
                assert_eq!(
                    outcome
                        .results
                        .iter()
                        .map(|result| (result.id, result.dist))
                        .collect::<Vec<_>>(),
                    exact
                        .iter()
                        .map(|result| (result.id, result.dist))
                        .collect::<Vec<_>>()
                );
            }
        }

        let forced_graph = build(3, 0).search(&[1.5], &Filter::none(), 2, 4).unwrap();
        assert_eq!(
            forced_graph.diagnostics.primary_execution,
            SearchExecution::LocalGraph
        );
    }

    #[test]
    fn search_rejects_queries_outside_the_index_distance_domain() {
        let index = build_test_index();
        assert!(matches!(
            index.search(&[f32::MAX, 0.0], &Filter::none(), 1, 1),
            Err(PrismError::InvalidInput(_))
        ));
        assert!(matches!(
            index.search_exact(&[0.0, f32::MAX], &Filter::none(), 1),
            Err(PrismError::InvalidInput(_))
        ));
        assert!(matches!(
            index.batch_search(&[f32::MAX, 0.0], &[Filter::none()], 1, 1, 1),
            Err(PrismError::InvalidInput(_))
        ));
    }

    #[test]
    fn unfiltered_single_and_batch_paths_are_graph_only_when_complete() {
        let mut store = PointStore::new(1, 1).unwrap();
        for point in 0..20usize {
            store.push(&[point as f32], &[0]).unwrap();
        }
        let mut index = PrismIndex::build(
            store,
            PrismConfig {
                m_local: 2,
                m_greedy: 0,
                m_random: 0,
                beam_width: 4,
                graph_expansion: 3,
                scan_threshold: 0,
                ..Default::default()
            },
        )
        .unwrap();

        // Only the first six points are reachable from the construction entry, and
        // point 19 is the exact nearest neighbor. A full result set excluding it
        // proves no exact scan, prefilter scan, or underfill fallback ran.
        let mut adjacency = vec![Vec::new(); 20];
        for point in 0..5u32 {
            adjacency[point as usize].push(point + 1);
        }
        index.graph = Graph::from_adj(&adjacency).unwrap();
        index.global_medoid = 0;
        index.medoids[0] = 0;

        let filter = Filter::none();
        let single = index.search(&[19.0], &filter, 2, 2).unwrap();
        assert_eq!(single.results.len(), 2);
        assert_eq!(
            single.diagnostics.primary_execution,
            SearchExecution::LocalGraph
        );
        assert!(!single.diagnostics.used_exact_fallback);
        assert!(single.results.iter().all(|result| result.id <= 5));
        assert!(single.results.iter().all(|result| result.id != 19));

        let batch = index.batch_search(&[19.0], &[filter], 1, 2, 2).unwrap();
        assert_eq!(batch[0].results.len(), 2);
        assert!(!batch[0].diagnostics.used_exact_fallback);
        assert!(batch[0].results.iter().all(|result| result.id <= 5));
        assert!(batch[0].results.iter().all(|result| result.id != 19));
    }

    #[test]
    fn equal_threshold_unfiltered_search_visits_every_local_graph() {
        let mut store = PointStore::new(2, 1).unwrap();
        for cell in 0..4u32 {
            for point in 0..40usize {
                store
                    .push(
                        &[
                            cell as f32 * 100.0 + point as f32 * 0.01,
                            (point as f32 * 0.13).sin(),
                        ],
                        &[cell],
                    )
                    .unwrap();
            }
        }
        let index = PrismIndex::build(
            store,
            PrismConfig {
                m_local: 8,
                m_greedy: 4,
                m_random: 4,
                beam_width: 24,
                sigma_low: 0.10,
                sigma_high: 0.10,
                graph_expansion: 3,
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

        // The global medoid sits near the middle cells while the exact nearest
        // neighbors are all in the last, so a single entry into a disconnected
        // global graph cannot pass.
        let query = [300.2, 0.5];
        let approximate = index.search(&query, &Filter::none(), 10, 32).unwrap();
        let exact = index.search_exact(&query, &Filter::none(), 10).unwrap();
        assert_eq!(approximate.results.len(), 10);
        assert_eq!(
            approximate.diagnostics.primary_execution,
            SearchExecution::LocalGraph
        );
        assert!(!approximate.diagnostics.used_exact_fallback);
        assert_eq!(
            approximate
                .results
                .iter()
                .map(|result| result.id)
                .collect::<Vec<_>>(),
            exact.iter().map(|result| result.id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn global_unfiltered_route_connects_adversarial_greedy_components() {
        // With m_greedy=1 and no random overlay, nearest foreign-cell edges
        // alone form two disconnected pairs: {0,1} and {100,101}. A count-only
        // fallback cannot detect the wrong component when k=2.
        let mut store = PointStore::new(1, 1).unwrap();
        for (attribute, coordinate) in [(0, 0.0), (1, 1.0), (2, 100.0), (3, 101.0)] {
            store.push(&[coordinate], &[attribute]).unwrap();
        }
        let index = PrismIndex::build(
            store,
            PrismConfig {
                m_local: 1,
                m_greedy: 1,
                m_random: 0,
                t: 1,
                beam_width: 4,
                graph_expansion: 3,
                scan_threshold: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(index.config.uses_global_graph());

        let mut reachable = HashSet::new();
        let mut pending = vec![index.global_medoid];
        while let Some(point) = pending.pop() {
            if reachable.insert(point) {
                pending.extend(index.graph.neighbors_unchecked(point));
            }
        }
        assert_eq!(reachable.len(), index.store.len);

        // Query the geometric pair opposite the global entry so this would
        // return two confidently wrong neighbors without the backbone.
        let entry_coordinate = index.store.vector_unchecked(index.global_medoid)[0];
        let query = if entry_coordinate < 50.0 {
            [100.9]
        } else {
            [0.1]
        };
        let approximate = index.search(&query, &Filter::none(), 2, 4).unwrap();
        let exact = index.search_exact(&query, &Filter::none(), 2).unwrap();
        assert_eq!(
            approximate.diagnostics.primary_execution,
            SearchExecution::GlobalGraph
        );
        assert!(!approximate.diagnostics.used_exact_fallback);
        assert_eq!(
            approximate
                .results
                .iter()
                .map(|result| result.id)
                .collect::<Vec<_>>(),
            exact.iter().map(|result| result.id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn graph_expansion_saturates_before_population_cap() {
        let mut index = build_test_index();
        index.config.graph_expansion = usize::MAX;
        assert_eq!(index.expanded_graph_ef(2, 123), 123);
    }

    #[test]
    fn forced_local_vamana_fixture_recall_is_at_least_ninety_percent_across_tested_build_seeds() {
        const N: usize = 2_500;
        const DIM: usize = 128;
        const NQ: usize = 20;
        const K: usize = 10;
        const EF: usize = 64;

        let mut data_rng = StdRng::seed_from_u64(0x4441_5441);
        let vectors: Vec<f32> = (0..N * DIM).map(|_| data_rng.gen::<f32>()).collect();
        let queries: Vec<f32> = (0..NQ * DIM).map(|_| data_rng.gen::<f32>()).collect();

        for build_seed in [7, 0x1234_5678, u64::MAX - 17] {
            let store = PointStore::from_parts(vectors.clone(), DIM, vec![vec![0; N]]).unwrap();
            let index = PrismIndex::build(
                store,
                PrismConfig {
                    m_local: 16,
                    m_greedy: 0,
                    m_random: 0,
                    beam_width: 32,
                    graph_expansion: 3,
                    build_seed,
                    scan_threshold: 0,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(index.tree.cells.len(), 1);
            assert!(
                index.tree.cells[0].point_ids.len() > index.config.m_local.saturating_add(1),
                "fixture must take the local-Vamana construction path"
            );

            let mut hits = 0usize;
            for query_idx in 0..NQ {
                let query = &queries[query_idx * DIM..(query_idx + 1) * DIM];
                // Calling the primitive directly with scan_threshold=0 forces the
                // graph branch, leaving no underfill fallback or cell scan to mask it.
                let candidates = index.search_cell_graph(query, 0, EF);
                assert_eq!(candidates.len(), EF);
                let mut approximate: Vec<(f32, u32)> = candidates
                    .into_iter()
                    .map(|(_, id)| {
                        (
                            distance::distance(
                                query,
                                index.store.vector_unchecked(id),
                                distance::Metric::L2,
                            ),
                            id,
                        )
                    })
                    .collect();
                approximate
                    .sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
                approximate.truncate(K);

                let mut exact: Vec<(f32, u32)> = (0..N as u32)
                    .map(|id| {
                        (
                            distance::distance(
                                query,
                                index.store.vector_unchecked(id),
                                distance::Metric::L2,
                            ),
                            id,
                        )
                    })
                    .collect();
                exact.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
                let ground_truth: HashSet<u32> =
                    exact.into_iter().take(K).map(|(_, id)| id).collect();
                hits += approximate
                    .iter()
                    .filter(|(_, id)| ground_truth.contains(id))
                    .count();
            }

            let recall = hits as f64 / (NQ * K) as f64;
            eprintln!("build_seed={build_seed}: forced-graph fixture Recall@{K}={recall:.4}");
            assert!(
                recall >= 0.90,
                "forced-graph fixture Recall@{K}={recall:.4} for build_seed={build_seed}; expected >=0.90 on this fixture"
            );
        }
    }

    #[test]
    fn original_id_maps_reordered_search_results() {
        let mut store = PointStore::new(2, 1).unwrap();
        store.push(&[100.0, 0.0], &[1]).unwrap(); // original 0
        store.push(&[0.0, 0.0], &[0]).unwrap(); // original 1, exact nearest
        store.push(&[200.0, 0.0], &[1]).unwrap(); // original 2
        store.push(&[10.0, 0.0], &[0]).unwrap(); // original 3
        let index = PrismIndex::build(store, PrismConfig::default()).unwrap();

        assert_eq!(index.original_ids, vec![1, 3, 0, 2]);
        let result = index.search_exact(&[0.0, 0.0], &Filter::none(), 1).unwrap();
        assert_eq!(result[0].id, 0);
        assert_eq!(index.original_id(result[0].id), Some(1));
        assert_eq!(index.original_id(u32::MAX), None);
    }

    #[test]
    fn test_search_with_filter() {
        let index = build_test_index();
        let filter = Filter::eq(0, 1);
        let results = index.search(&[0.5, 0.5], &filter, 3, 10).unwrap().results;
        assert!(!results.is_empty());
        for r in &results {
            assert!(filter.matches(&index.store, r.id));
        }
    }

    #[test]
    fn test_graph_search_mid_selectivity() {
        let dim = 16;
        let n = 2000;
        let n_vals = 20;
        let mut store = PointStore::new(dim, 1).unwrap();
        for i in 0..n {
            let vec: Vec<f32> = (0..dim).map(|d| ((i * dim + d) as f32).sin()).collect();
            store.push(&vec, &[(i % n_vals) as u32]).unwrap();
        }
        let config = PrismConfig {
            m_local: 4,
            m_greedy: 2,
            m_random: 4,
            t: 1,
            beam_width: 10,
            scan_threshold: 0,
            ..Default::default()
        };
        let index = PrismIndex::build(store, config).unwrap();

        let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.3).sin()).collect();
        let filter = Filter::eq(0, 0);
        let k = 5;
        let ef = 10;

        let outcome = index.search(&query, &filter, k, ef).unwrap();
        assert_eq!(outcome.diagnostics.plan.regime, SearchRegime::Mid);
        assert_eq!(
            outcome.diagnostics.primary_execution,
            SearchExecution::LocalGraph
        );
        assert!(!outcome.results.is_empty());
        assert!(outcome.results.len() <= k);
        for r in &outcome.results {
            assert!(filter.matches(&index.store, r.id));
        }
        for w in outcome.results.windows(2) {
            assert!(w[0].dist <= w[1].dist);
        }
    }

    #[test]
    fn test_search_empty_filter() {
        let index = build_test_index();
        let filter = Filter::eq(0, 99);
        let results = index.search(&[0.0, 0.0], &filter, 3, 10).unwrap().results;
        assert!(results.is_empty());
    }

    #[test]
    fn test_regime_mid_single_cell_uses_local_graph() {
        // 20 attribute values x 100 points/value = 2000 points; sigma_high=0.10,
        // each value is 5% selectivity, so single-value filters route to MID.
        let dim = 16;
        let n = 2000;
        let n_vals = 20;
        let mut store = PointStore::new(dim, 1).unwrap();
        for i in 0..n {
            let vec: Vec<f32> = (0..dim).map(|d| ((i * dim + d) as f32).sin()).collect();
            store.push(&vec, &[(i % n_vals) as u32]).unwrap();
        }
        let config = PrismConfig {
            m_local: 4,
            m_greedy: 4,
            m_random: 4,
            t: 1,
            beam_width: 20,
            sigma_high: 0.10,
            sigma_low: 0.001,
            beta: 3.0,
            epsilon: 0.2,
            scan_threshold: 0,
            ..Default::default()
        };
        let index = PrismIndex::build(store, config).unwrap();

        // Value 0 matches 100 of 2000 points = 5% selectivity = MID regime.
        let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.3).sin()).collect();
        let filter = Filter::eq(0, 0);
        let k = 5;
        let ef = 50;

        let outcome = index.search(&query, &filter, k, ef).unwrap();
        assert_eq!(outcome.diagnostics.plan.regime, SearchRegime::Mid);
        assert_eq!(
            outcome.diagnostics.primary_execution,
            SearchExecution::LocalGraph
        );
        assert!(!outcome.results.is_empty());
        assert!(outcome.results.len() <= k);
        for r in &outcome.results {
            assert!(filter.matches(&index.store, r.id));
        }
        for w in outcome.results.windows(2) {
            assert!(w[0].dist <= w[1].dist);
        }
    }

    #[test]
    fn batch_search_reports_mid_exact_below_the_total_threshold() {
        let dim = 8;
        let n = 1000;
        let n_vals = 10;
        let mut store = PointStore::new(dim, 1).unwrap();
        for i in 0..n {
            let vec: Vec<f32> = (0..dim).map(|d| ((i * dim + d) as f32).sin()).collect();
            store.push(&vec, &[(i % n_vals) as u32]).unwrap();
        }
        let config = PrismConfig {
            m_local: 4,
            m_greedy: 4,
            m_random: 4,
            t: 1,
            beam_width: 20,
            sigma_high: 0.11,
            sigma_low: 0.001,
            scan_threshold: 100,
            ..Default::default()
        };
        let index = PrismIndex::build(store, config).unwrap();

        let k = 3;
        let ef = 20;
        let nq = 3;

        // Query 0 is unfiltered (HIGH global graph); queries 1 and 2 filter to 10%
        // selectivity, hitting MID policy but the inclusive exact crossover at 100.
        let queries: Vec<f32> = (0..nq)
            .flat_map(|qi| (0..dim).map(move |d| ((qi * dim + d) as f32 * 0.5).sin()))
            .collect();
        let filters = vec![Filter::none(), Filter::eq(0, 0), Filter::eq(0, 5)];

        let results = index.batch_search(&queries, &filters, nq, k, ef).unwrap();
        assert_eq!(results.len(), nq);
        assert_eq!(results[0].diagnostics.plan.regime, SearchRegime::High);
        assert_eq!(
            results[0].diagnostics.primary_execution,
            SearchExecution::GlobalGraph
        );
        assert!(results[1..]
            .iter()
            .all(|result| result.diagnostics.plan.regime == SearchRegime::Mid));
        assert!(results[1..]
            .iter()
            .all(|result| result.diagnostics.primary_execution == SearchExecution::ExactScan));
        for (qi, res) in results.iter().enumerate() {
            assert!(!res.results.is_empty(), "query {} returned no results", qi);
            assert!(res.results.len() <= k);
            for r in &res.results {
                assert!(filters[qi].matches(&index.store, r.id));
            }
        }
    }

    #[test]
    fn mid_proper_multi_cell_scan_threshold_is_inclusive() {
        let mut store = PointStore::new(2, 1).unwrap();
        for attribute in 0..3u32 {
            for point in 0..4usize {
                store
                    .push(
                        &[attribute as f32 * 10.0 + point as f32, point as f32],
                        &[attribute],
                    )
                    .unwrap();
            }
        }
        let mut index = PrismIndex::build(
            store,
            PrismConfig {
                m_local: 2,
                m_greedy: 1,
                m_random: 0,
                beam_width: 4,
                sigma_high: 0.80,
                sigma_low: 0.10,
                scan_threshold: 0,
                multi_cell_scan_threshold: 8,
                ..Default::default()
            },
        )
        .unwrap();
        let filter = Filter::new(vec![(0, vec![0, 1])]);
        let query = [5.0, 1.0];

        let scanned = index.search(&query, &filter, 3, 4).unwrap();
        assert_eq!(scanned.diagnostics.plan.regime, SearchRegime::Mid);
        assert_eq!(scanned.diagnostics.plan.matching_cells, 2);
        assert_eq!(scanned.diagnostics.plan.matching_points, 8);
        assert_eq!(
            scanned.diagnostics.primary_execution,
            SearchExecution::ExactScan
        );
        let exact = index.search_exact(&query, &filter, 3).unwrap();
        assert_eq!(
            scanned
                .results
                .iter()
                .map(|result| (result.id, result.dist))
                .collect::<Vec<_>>(),
            exact
                .iter()
                .map(|result| (result.id, result.dist))
                .collect::<Vec<_>>()
        );

        index.config.multi_cell_scan_threshold = 7;
        assert_eq!(
            index
                .search(&query, &filter, 1, 4)
                .unwrap()
                .diagnostics
                .primary_execution,
            SearchExecution::GlobalGraph
        );
    }

    #[test]
    fn inner_product_candidates_survive_l2_blind_spot() {
        // 59 decoys hug the query in L2 with tiny dot products; one high-norm
        // point is the true IP winner but the L2-farthest point in the set. An
        // SQ8-L2 candidate heap (ef < n) would evict it before the rerank.
        let mut store = PointStore::new(2, 1).unwrap();
        for i in 0..59 {
            let j = (i as f32) * 0.001;
            store.push(&[0.5 + j, j], &[0]).unwrap();
        }
        store.push(&[20.0, 0.0], &[0]).unwrap();
        let config = PrismConfig {
            m_local: 4,
            m_greedy: 2,
            m_random: 4,
            t: 1,
            beam_width: 10,
            metric: distance::Metric::InnerProduct,
            binary_rerank: 0,
            ..Default::default()
        };
        let index = PrismIndex::build(store, config).unwrap();

        let results = index
            .search(&[1.0, 0.0], &Filter::none(), 1, 8)
            .unwrap()
            .results;
        assert_eq!(results[0].id, 59, "true IP winner must reach the rerank");
        assert!((results[0].dist - (-20.0)).abs() < 1e-3);
    }

    #[test]
    fn inner_product_uses_exact_path_even_when_binary_prefilter_is_requested() {
        let mut store = PointStore::new(2, 1).unwrap();
        for _ in 0..99 {
            store.push(&[1.0, 0.0], &[0]).unwrap();
        }
        store.push(&[10.0, 100.0], &[0]).unwrap();
        let index = PrismIndex::build(
            store,
            PrismConfig {
                metric: distance::Metric::InnerProduct,
                binary_rerank: 4,
                ..Default::default()
            },
        )
        .unwrap();

        let results = index
            .search(&[1.0, 0.0], &Filter::none(), 1, 1)
            .unwrap()
            .results;
        assert_eq!(index.original_id(results[0].id), Some(99));
        assert!((results[0].dist + 10.0).abs() < 1e-6);
        assert_eq!(
            index.plan_filter(&Filter::none()).unwrap().regime,
            SearchRegime::Exact
        );
        assert!(index.binary.is_empty());
    }

    #[test]
    fn scale_aware_sq8_search_handles_anisotropic_dimensions() {
        let store =
            PointStore::from_parts(vec![100.0, 0.0, 0.0, 1.0, 1000.0, 0.0], 2, vec![vec![0; 3]])
                .unwrap();
        let index = PrismIndex::build(
            store,
            PrismConfig {
                binary_rerank: 0,
                scan_threshold: 0,
                ..Default::default()
            },
        )
        .unwrap();

        let results = index
            .search(&[0.0, 0.0], &Filter::none(), 1, 1)
            .unwrap()
            .results;
        assert_eq!(index.original_id(results[0].id), Some(1));
        assert!((results[0].dist - 1.0).abs() < 1e-6);
    }

    #[test]
    fn search_promotes_small_ef_and_completes_small_cell() {
        let index = build_test_index();
        let filter = Filter::eq(0, 0);
        let results = index.search(&[0.0, 0.0], &filter, 5, 0).unwrap().results;
        assert_eq!(results.len(), 5);
        assert!(results
            .iter()
            .all(|result| filter.matches(&index.store, result.id)));

        let exact = index.search_exact(&[0.0, 0.0], &filter, 5).unwrap();
        assert_eq!(exact.len(), 5);
        assert_eq!(
            results.iter().map(|result| result.id).collect::<Vec<_>>(),
            exact.iter().map(|result| result.id).collect::<Vec<_>>()
        );
    }

    fn build_routing_fixture(metric: distance::Metric) -> PrismIndex {
        let mut store = PointStore::new(3, 2).unwrap();
        for i in 0..1000usize {
            let x = i as f32 / 1000.0;
            store
                .push(
                    &[x, x * x + 0.01, (i as f32 * 0.017).sin()],
                    &[(i % 100) as u32, ((i / 10) % 10) as u32],
                )
                .unwrap();
        }
        PrismIndex::build(
            store,
            PrismConfig {
                metric,
                m_local: 4,
                m_greedy: 4,
                m_random: 0,
                t: 1,
                beam_width: 20,
                sigma_high: 0.10,
                sigma_low: 0.01,
                ..Default::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn plans_exact_cardinality_and_routes_threshold_boundaries() {
        let index = build_routing_fixture(distance::Metric::L2);
        let cases = [
            (Filter::none(), 1000, SearchRegime::High),
            (
                Filter::new(vec![(0, (0..10).collect())]),
                100,
                SearchRegime::High,
            ),
            (
                Filter::new(vec![(0, (0..5).collect())]),
                50,
                SearchRegime::Mid,
            ),
            (Filter::eq(0, 0), 10, SearchRegime::Low),
            (Filter::eq(0, 999), 0, SearchRegime::Low),
        ];

        for (filter, expected_count, expected_regime) in cases {
            let brute_count = (0..index.store.len as u32)
                .filter(|&point| filter.matches(&index.store, point))
                .count();
            let plan = index.plan_filter(&filter).unwrap();
            assert_eq!(plan.matching_points, brute_count);
            assert_eq!(plan.matching_points, expected_count);
            assert_eq!(plan.regime, expected_regime);
        }

        let mut no_mid = build_routing_fixture(distance::Metric::L2);
        no_mid.config.sigma_low = 0.10;
        no_mid.config.sigma_high = 0.10;
        for filter in [
            Filter::none(),
            Filter::new(vec![(0, (0..5).collect())]),
            Filter::eq(0, 0),
        ] {
            assert_ne!(
                no_mid.plan_filter(&filter).unwrap().regime,
                SearchRegime::Mid
            );
        }
    }

    #[test]
    fn search_and_batch_preserve_predicates_and_counts_across_regimes() {
        let index = build_routing_fixture(distance::Metric::L2);
        let filters = vec![
            Filter::none(),
            Filter::new(vec![(0, (0..20).collect())]),
            Filter::new(vec![(0, (0..5).collect()), (1, vec![0, 1, 2])]),
            Filter::eq(0, 0),
            Filter::eq(0, 999),
        ];
        let query = [0.2, 0.05, -0.3];
        let k = 7;

        for filter in &filters {
            let eligible = (0..index.store.len as u32)
                .filter(|&point| filter.matches(&index.store, point))
                .count();
            let results = index.search(&query, filter, k, 0).unwrap().results;
            assert_eq!(results.len(), k.min(eligible));
            assert!(results
                .iter()
                .all(|result| filter.matches(&index.store, result.id)));
            let unique: std::collections::HashSet<_> =
                results.iter().map(|result| result.id).collect();
            assert_eq!(unique.len(), results.len());
            assert!(results.windows(2).all(|pair| pair[0].dist <= pair[1].dist));
        }

        let queries: Vec<f32> = filters.iter().flat_map(|_| query).collect();
        let results = index
            .batch_search(&queries, &filters, filters.len(), k, 0)
            .unwrap();
        for (filter, row) in filters.iter().zip(results.iter()) {
            let eligible = (0..index.store.len as u32)
                .filter(|&point| filter.matches(&index.store, point))
                .count();
            assert_eq!(row.results.len(), k.min(eligible));
            assert!(row
                .results
                .iter()
                .all(|result| filter.matches(&index.store, result.id)));
        }
    }

    #[test]
    fn search_exact_matches_independent_bruteforce_for_every_metric() {
        for metric in [
            distance::Metric::L2,
            distance::Metric::Cosine,
            distance::Metric::InnerProduct,
        ] {
            let index = build_routing_fixture(metric);
            let filter = Filter::new(vec![(0, vec![1, 7, 13, 42]), (1, vec![0, 3, 7])]);
            let query = [0.37, -0.2, 0.91];
            let k = 9;
            let actual = index.search_exact(&query, &filter, k).unwrap();

            let normalized;
            let exact_query = if metric == distance::Metric::Cosine {
                normalized = distance::normalized(&query);
                normalized.as_slice()
            } else {
                query.as_slice()
            };
            let mut expected: Vec<_> = (0..index.store.len as u32)
                .filter(|&point| filter.matches(&index.store, point))
                .map(|id| super::SearchResult {
                    id,
                    dist: distance::distance(exact_query, index.store.vector_unchecked(id), metric),
                })
                .collect();
            expected.sort_by(|a, b| a.dist.total_cmp(&b.dist).then(a.id.cmp(&b.id)));
            expected.truncate(k);

            assert_eq!(
                actual.iter().map(|result| result.id).collect::<Vec<_>>(),
                expected.iter().map(|result| result.id).collect::<Vec<_>>()
            );
            for (found, reference) in actual.iter().zip(expected.iter()) {
                assert!((found.dist - reference.dist).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn optimized_exact_cosine_scan_matches_public_distance_bit_for_bit() {
        const DIM: usize = 16;
        let base: Vec<f32> = (0..DIM)
            .map(|i| (i as f32 * 0.19).sin() - 0.2 * (i as f32 * 0.07).cos())
            .collect();
        let mut near_one = base.clone();
        near_one[5] += 0.001;
        let mut near_two = base.clone();
        near_two[5] += f32::from_bits(0.001f32.to_bits() + 1);
        let extreme: Vec<f32> = (0..DIM)
            .map(|i| if i % 2 == 0 { 1.0e18 } else { -1.0e18 })
            .collect();

        let rows = vec![
            base.clone(),
            base.iter().map(|value| *value * 2.0).collect(),
            near_one,
            near_two,
            base.iter().map(|value| -*value).collect(),
            vec![0.0; DIM],
            extreme,
            (0..DIM).map(|i| (i as f32 * 0.31).cos()).collect(),
        ];
        let point_count = rows.len();
        let vectors: Vec<f32> = rows.into_iter().flatten().collect();
        let store = PointStore::from_parts(vectors, DIM, vec![vec![0; point_count]]).unwrap();
        let index = PrismIndex::build(
            store,
            PrismConfig {
                metric: distance::Metric::Cosine,
                ..Default::default()
            },
        )
        .unwrap();

        let query: Vec<f32> = base.iter().map(|value| *value * 1.0e18).collect();
        let all = index
            .search_exact(&query, &Filter::none(), index.store.len)
            .unwrap();
        let selected = index
            .search_exact(&query, &Filter::none(), point_count / 2)
            .unwrap();
        let normalized_query = distance::normalized(&query);
        let mut expected: Vec<_> = (0..index.store.len as u32)
            .map(|id| super::SearchResult {
                id,
                dist: distance::cosine(&normalized_query, index.store.vector_unchecked(id)),
            })
            .collect();
        expected.sort_by(|left, right| {
            left.dist
                .total_cmp(&right.dist)
                .then(left.id.cmp(&right.id))
        });

        assert_eq!(
            all.iter()
                .map(|result| (result.id, result.dist.to_bits()))
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|result| (result.id, result.dist.to_bits()))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            selected
                .iter()
                .map(|result| (result.id, result.dist.to_bits()))
                .collect::<Vec<_>>(),
            expected
                .iter()
                .take(point_count / 2)
                .map(|result| (result.id, result.dist.to_bits()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn every_cosine_rerank_route_preserves_public_distance_bits() {
        const DIM: usize = 32;
        const POINTS: usize = 128;
        let base: Vec<f32> = (0..DIM)
            .map(|dimension| {
                (dimension as f32 * 0.071).sin() + 0.3 * (dimension as f32 * 0.113).cos()
            })
            .collect();
        let mut store = PointStore::new(DIM, 1).unwrap();
        for point in 0..POINTS {
            let mut row: Vec<f32> = (0..DIM)
                .map(|dimension| {
                    ((point * 17 + dimension) as f32 * 0.013).sin()
                        + 0.25 * ((point + dimension * 7) as f32 * 0.021).cos()
                })
                .collect();
            if point == 0 {
                row.clone_from(&base);
            } else if point == 1 {
                row.clone_from(&base);
                row[17] += 0.001;
            } else if point == 2 {
                row.fill(0.0);
            }
            store.push(&row, &[(point % 2) as u32]).unwrap();
        }
        let index = PrismIndex::build(
            store,
            PrismConfig {
                metric: distance::Metric::Cosine,
                binary_rerank: 2,
                m_local: 8,
                m_greedy: 4,
                m_random: 4,
                beam_width: 16,
                scan_threshold: 0,
                multi_cell_scan_threshold: 0,
                ..Default::default()
            },
        )
        .unwrap();
        let query = distance::normalized(&base);
        let assert_public_bits = |results: &[super::SearchResult]| {
            for result in results {
                assert_eq!(
                    result.dist.to_bits(),
                    distance::cosine(&query, index.store.vector_unchecked(result.id)).to_bits(),
                    "reranked distance for point {} changed",
                    result.id
                );
            }
        };

        let exact_rerank = index.exact_rerank(&query, 0..POINTS as u32, 12);
        assert_public_bits(&exact_rerank);

        let all_cells: Vec<usize> = (0..index.tree.cells.len()).collect();
        let (binary, execution) = index
            .scan_eligible_population(&query, &all_cells, POINTS, 8, 8)
            .unwrap();
        assert_eq!(execution, SearchExecution::BinaryPrefilterScan);
        assert_public_bits(&binary);

        let local = index.regime_local_graph(&query, &[0], 8, 16);
        assert_public_bits(&local);

        let global = index.regime_mid(&query, &all_cells, 8, 32);
        assert_public_bits(&global);
    }

    #[test]
    fn mid_traversal_can_use_a_nonmatching_bridge() {
        let store =
            PointStore::from_parts(vec![1.0, 2.0, 1.0], 1, vec![vec![0, 0, 1], vec![0, 1, 0]])
                .unwrap();
        let mut index = PrismIndex::build(
            store,
            PrismConfig {
                m_local: 1,
                m_greedy: 1,
                m_random: 0,
                beam_width: 4,
                sigma_high: 0.90,
                sigma_low: 0.10,
                beta: 0.0,
                scan_threshold: 0,
                multi_cell_scan_threshold: 0,
                ..Default::default()
            },
        )
        .unwrap();
        // Eligible A (0) can reach eligible B (1) only through nonmatching X (2).
        index.graph = Graph::from_adj(&[vec![2], vec![], vec![1]]).unwrap();
        let filter = Filter::eq(0, 0);
        let cells = index.tree.filter_cells(filter.constraints()).unwrap();
        assert_eq!(
            index.plan_filter(&filter).unwrap().regime,
            SearchRegime::Mid
        );

        let without_bridge = index.regime_mid(&[0.0], &cells, 2, 2);
        assert_eq!(without_bridge.len(), 1);

        let rescued = index.search(&[0.0], &filter, 2, 2).unwrap();
        let exact = index.search_exact(&[0.0], &filter, 2).unwrap();
        assert_eq!(
            rescued.diagnostics.primary_execution,
            SearchExecution::GlobalGraph
        );
        assert_eq!(rescued.diagnostics.primary_result_count, 1);
        assert!(rescued.diagnostics.used_exact_fallback);
        assert!(rescued.diagnostics.complete);
        assert!(
            rescued.diagnostics.total_elapsed
                >= rescued.diagnostics.primary_elapsed + rescued.diagnostics.fallback_elapsed
        );
        assert_eq!(rescued.results.len(), 2);
        assert_eq!(
            rescued
                .results
                .iter()
                .map(|result| result.id)
                .collect::<Vec<_>>(),
            exact.iter().map(|result| result.id).collect::<Vec<_>>()
        );

        // The batched path has its own underfill detection, so it must make the
        // same exact replacement on this deliberately incomplete MID row.
        let batch = index
            .batch_search(&[0.0], std::slice::from_ref(&filter), 1, 2, 2)
            .unwrap();
        assert_eq!(batch[0].results.len(), 2);
        assert!(batch[0].diagnostics.used_exact_fallback);
        assert!(
            batch[0].diagnostics.total_elapsed
                >= batch[0].diagnostics.primary_elapsed + batch[0].diagnostics.fallback_elapsed
        );
        assert_eq!(
            batch[0]
                .results
                .iter()
                .map(|result| result.id)
                .collect::<Vec<_>>(),
            exact.iter().map(|result| result.id).collect::<Vec<_>>()
        );

        index.config.beta = 1.0;
        let with_bridge = index.regime_mid(&[0.0], &cells, 2, 2);
        assert_eq!(with_bridge.len(), 2);
        assert!(with_bridge
            .iter()
            .all(|result| filter.matches(&index.store, result.id)));
    }

    #[test]
    fn cosine_candidates_survive_unnormalized_inputs() {
        // The best-angle point has a huge norm and is L2-farthest from the raw
        // query, so a raw SQ8-L2 heap would evict it. Build-time normalization is
        // what makes code distance an angular approximation here.
        let mut store = PointStore::new(2, 1).unwrap();
        for i in 0..59 {
            let j = (i as f32) * 0.001;
            store.push(&[j, 1.0 + j], &[0]).unwrap();
        }
        store.push(&[50.0, 1.0], &[0]).unwrap();
        let config = PrismConfig {
            m_local: 4,
            m_greedy: 2,
            m_random: 4,
            t: 1,
            beam_width: 10,
            metric: distance::Metric::Cosine,
            binary_rerank: 0,
            scan_threshold: 0,
            ..Default::default()
        };
        let index = PrismIndex::build(store, config).unwrap();

        let results = index
            .search(&[3.0, 0.0], &Filter::none(), 1, 8)
            .unwrap()
            .results;
        assert_eq!(results[0].id, 59, "best-angle point must reach the rerank");
        assert!(
            results[0].dist < 0.01,
            "dist {} is not ~1-cos",
            results[0].dist
        );
    }

    #[test]
    fn cosine_index_keeps_subnormal_finite_vectors_and_queries_well_defined() {
        let smallest = f32::from_bits(1);
        let store = PointStore::from_parts(vec![smallest, 0.0, 0.0, smallest], 2, vec![vec![0, 0]])
            .unwrap();
        let index = PrismIndex::build(
            store,
            PrismConfig {
                metric: distance::Metric::Cosine,
                m_greedy: 0,
                m_random: 0,
                scan_threshold: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(index.store.vectors.iter().all(|value| value.is_finite()));

        let outcome = index
            .search(&[smallest, 0.0], &Filter::none(), 1, 2)
            .unwrap();
        assert_eq!(index.original_id(outcome.results[0].id), Some(0));
        assert_eq!(outcome.results[0].dist, 0.0);
    }

    #[test]
    fn multi_cell_binary_prefilter_uses_one_query_wide_budget() {
        // The binary pre-filter is an approximation; results stay valid and
        // ordered, just not necessarily identical to the exact-scan path.
        let dim = 64;
        let n = 2500;
        let n_vals = 10;
        let mut store = PointStore::new(dim, 1).unwrap();
        for i in 0..n {
            let vec: Vec<f32> = (0..dim)
                .map(|d| ((i * dim + d) as f32 * 0.01).sin())
                .collect();
            store.push(&vec, &[(i % n_vals) as u32]).unwrap();
        }

        let config_binary = PrismConfig {
            m_local: 4,
            m_greedy: 2,
            m_random: 4,
            t: 1,
            beam_width: 10,
            binary_rerank: 4,
            ..Default::default()
        };
        let index_binary = PrismIndex::build(store, config_binary).unwrap();

        let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.3).sin()).collect();
        // Each of the three compatible cells holds 100 points and fits the binary
        // budget alone, but their 300-point total does not, proving the budget
        // applies once per query rather than once per cell.
        let filter = Filter::new(vec![(0, vec![0, 1, 2])]);
        let k = 10;
        let ef = 64;
        let per_cell = n / n_vals;
        let eligible = per_cell * 3;
        let binary_budget = index_binary.config.binary_rerank * ef;
        assert!(per_cell <= binary_budget);
        assert!(eligible > binary_budget);

        let outcome_binary = index_binary.search(&query, &filter, k, ef).unwrap();
        assert_eq!(
            outcome_binary.diagnostics.primary_execution,
            SearchExecution::BinaryPrefilterScan
        );
        let results_binary = outcome_binary.results;
        assert_eq!(results_binary.len(), k);
        for r in &results_binary {
            assert!(filter.matches(&index_binary.store, r.id));
        }
        for w in results_binary.windows(2) {
            assert!(w[0].dist <= w[1].dist);
        }
    }

    #[test]
    fn test_binary_prefilter_batch() {
        let dim = 32;
        let n = 500;
        let n_vals = 5;
        let mut store = PointStore::new(dim, 1).unwrap();
        for i in 0..n {
            let vec: Vec<f32> = (0..dim)
                .map(|d| ((i * dim + d) as f32 * 0.02).sin())
                .collect();
            store.push(&vec, &[(i % n_vals) as u32]).unwrap();
        }

        let config = PrismConfig {
            m_local: 4,
            m_greedy: 2,
            m_random: 4,
            t: 1,
            beam_width: 10,
            binary_rerank: 4,
            ..Default::default()
        };
        let index = PrismIndex::build(store, config).unwrap();

        let nq = 5;
        let k = 5;
        let ef = 20;
        let queries: Vec<f32> = (0..nq)
            .flat_map(|qi| (0..dim).map(move |d| ((qi * dim + d) as f32 * 0.1).sin()))
            .collect();
        let filters: Vec<Filter> = (0..nq)
            .map(|qi| {
                if qi == 0 {
                    Filter::none()
                } else {
                    Filter::eq(0, (qi % n_vals) as u32)
                }
            })
            .collect();

        let results = index.batch_search(&queries, &filters, nq, k, ef).unwrap();
        assert_eq!(results.len(), nq);
        for (qi, res) in results.iter().enumerate() {
            assert!(!res.results.is_empty(), "query {} returned no results", qi);
            assert!(res.results.len() <= k);
            for r in &res.results {
                assert!(filters[qi].matches(&index.store, r.id));
            }
        }
    }
}
