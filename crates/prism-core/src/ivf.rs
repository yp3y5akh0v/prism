//! PRISM's [IVF^2-inspired](https://openreview.net/forum?id=kXw8E3xT7O)
//! filtered-IVF variant.
//!
//! Published IVF^2 indexes filter postings separately; PRISM uses one global
//! K-means partition with per-cluster tag postings and multi-query cell batching
//! (MQCB).

use super::binary::BinaryStore;
use super::distance;
use super::error::{PrismError, PrismResult};
use super::point::validate_f32_distance_domain;

use rand::prelude::*;
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Mutex;

/// CSR sparse matrix (same layout as scipy.sparse.csr_matrix).
pub struct SpMat {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) indptr: Vec<i64>,
    pub(crate) indices: Vec<i32>,
}

impl SpMat {
    /// Construct and validate CSR metadata.
    pub fn new(rows: usize, cols: usize, indptr: Vec<i64>, indices: Vec<i32>) -> PrismResult<Self> {
        let matrix = Self {
            rows,
            cols,
            indptr,
            indices,
        };
        matrix.validate(rows)?;
        Ok(matrix)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn indptr(&self) -> &[i64] {
        &self.indptr
    }

    pub fn indices(&self) -> &[i32] {
        &self.indices
    }

    /// Validate CSR shape, offsets, and tag IDs.
    pub fn validate(&self, expected_rows: usize) -> PrismResult<()> {
        let max_cols = i32::MAX as usize + 1;
        if self.cols > max_cols {
            return Err(PrismError::Overflow(format!(
                "CSR column count {} exceeds the i32 tag identifier space ({max_cols})",
                self.cols
            )));
        }
        if self.rows != expected_rows {
            return Err(PrismError::InvalidInput(format!(
                "CSR row count {} does not match expected row count {expected_rows}",
                self.rows
            )));
        }
        let expected_indptr = self
            .rows
            .checked_add(1)
            .ok_or_else(|| PrismError::Overflow("CSR row count is too large".into()))?;
        if self.indptr.len() != expected_indptr {
            return Err(PrismError::InvalidInput(format!(
                "CSR indptr length {} must equal rows + 1 ({expected_indptr})",
                self.indptr.len()
            )));
        }
        if self.indptr.first().copied() != Some(0) {
            return Err(PrismError::InvalidInput(
                "CSR indptr must start at zero".into(),
            ));
        }
        let nnz = i64::try_from(self.indices.len()).map_err(|_| {
            PrismError::Overflow("CSR nonzero count exceeds the i64 offset space".into())
        })?;
        for pair in self.indptr.windows(2) {
            if pair[0] < 0 || pair[0] > pair[1] || pair[1] > nnz {
                return Err(PrismError::InvalidInput(
                    "CSR indptr must be nonnegative, monotone, and within indices".into(),
                ));
            }
        }
        if self.indptr.last().copied() != Some(nnz) {
            return Err(PrismError::InvalidInput(
                "CSR indptr must end at the indices length".into(),
            ));
        }
        if let Some(&tag) = self
            .indices
            .iter()
            .find(|&&tag| tag < 0 || tag as usize >= self.cols)
        {
            return Err(PrismError::InvalidInput(format!(
                "CSR tag {tag} is outside the valid range 0..{}",
                self.cols
            )));
        }
        Ok(())
    }
}

/// Type-erased flat vector storage (u8 or f32).
pub enum VecStore {
    U8(Vec<u8>),
    F32(Vec<f32>),
}

impl VecStore {
    pub fn len(&self) -> usize {
        match self {
            Self::U8(values) => values.len(),
            Self::F32(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_u8(&self) -> Option<&[u8]> {
        match self {
            Self::U8(values) => Some(values),
            Self::F32(_) => None,
        }
    }

    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32(values) => Some(values),
            Self::U8(_) => None,
        }
    }
}

impl Drop for VecStore {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        match self {
            Self::U8(values) => values.zeroize(),
            Self::F32(values) => values.zeroize(),
        }
    }
}

/// Borrowed query batch (flat, nq x dim).
pub enum QueryStore<'a> {
    U8(&'a [u8]),
    F32(&'a [f32]),
}

/// Single query vector slice.
enum QueryVec<'a> {
    U8(&'a [u8]),
    F32(&'a [f32]),
}

/// Distance suitable for heap ordering. For u8: raw u64 from l2_sq8.
/// For f32: f32::to_bits() (monotonic for non-negative IEEE 754 floats).
#[inline]
fn compute_dist(store: &VecStore, gid: usize, query: &QueryVec, dim: usize) -> u64 {
    match (store, query) {
        (VecStore::U8(v), QueryVec::U8(q)) => distance::l2_sq8(q, &v[gid * dim..(gid + 1) * dim]),
        (VecStore::F32(v), QueryVec::F32(q)) => {
            distance::l2_squared(q, &v[gid * dim..(gid + 1) * dim]).to_bits() as u64
        }
        _ => unreachable!("mismatched vector/query types"),
    }
}

/// Filtered-IVF index with global geometric clusters and per-cluster tag
/// postings.
pub struct IvfIndex {
    /// Reordered vectors (contiguous per cluster).
    pub(crate) vectors: VecStore,
    /// Mapping: reordered_id -> original_id.
    pub(crate) original_ids: Vec<u32>,
    /// Cluster boundaries: cluster `c` spans
    /// `cluster_starts[c]..cluster_starts[c + 1]`.
    pub(crate) cluster_starts: Vec<u32>,
    /// Per-cluster tag index offsets.
    tag_offsets: Vec<u32>,
    /// (tag_id, posting_start, posting_len) triples, sorted by tag_id within each cluster.
    tag_index: Vec<(u32, u32, u32)>,
    /// Flat array of local IDs for all (cluster, tag) posting lists.
    posting_ids: Vec<u32>,
    /// Per-tag list of clusters containing matching vectors.
    tag_clusters: HashMap<u32, Vec<u16>>,
    /// Declared tag-space cardinality. Sparse tag maps never allocate by this
    /// value, but callers still need it for validation.
    n_tags: usize,
    /// Binary codes built from vectors after all IVF reordering. Keeping them
    /// inside the index prevents caller-supplied, order-misaligned prefilters.
    binary: BinaryStore,
    /// Vector dimensionality.
    pub(crate) dim: usize,
    /// Number of clusters.
    pub(crate) n_clusters: usize,
}

impl IvfIndex {
    pub fn vectors(&self) -> &VecStore {
        &self.vectors
    }

    pub fn original_ids(&self) -> &[u32] {
        &self.original_ids
    }

    pub fn cluster_starts(&self) -> &[u32] {
        &self.cluster_starts
    }

    pub fn clusters_for_tag(&self, tag: usize) -> Option<&[u16]> {
        u32::try_from(tag)
            .ok()
            .and_then(|tag| self.tag_clusters.get(&tag))
            .map(Vec::as_slice)
    }

    pub fn n_tags(&self) -> usize {
        self.n_tags
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn n_clusters(&self) -> usize {
        self.n_clusters
    }

    pub fn len(&self) -> usize {
        self.original_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.original_ids.is_empty()
    }

    /// Build PRISM's filtered-IVF variant from clustered vectors and metadata.
    ///
    /// Reorders vectors by cluster, sorts within each cluster by most popular
    /// tag (tag-affinity sort), and builds per-cluster tag posting lists.
    pub fn build(
        base: &VecStore,
        base_meta: &SpMat,
        assignments: &[u16],
        n: usize,
        dim: usize,
        n_clusters: usize,
    ) -> PrismResult<Self> {
        if n == 0 {
            return Err(PrismError::InvalidInput(
                "IVF index requires at least one vector".into(),
            ));
        }
        if dim == 0 {
            return Err(PrismError::InvalidInput(
                "IVF vector dimension must be greater than zero".into(),
            ));
        }
        if n > u32::MAX as usize {
            return Err(PrismError::Overflow(
                "IVF point count exceeds the u32 identifier space".into(),
            ));
        }
        if !(1..=n).contains(&n_clusters) {
            return Err(PrismError::InvalidInput(
                "IVF cluster count must be between 1 and the point count".into(),
            ));
        }
        if n_clusters > u16::MAX as usize + 1 {
            return Err(PrismError::Overflow(
                "IVF cluster count exceeds the u16 assignment space".into(),
            ));
        }
        let expected_values = n.checked_mul(dim).ok_or_else(|| {
            PrismError::Overflow("IVF vector shape exceeds addressable memory".into())
        })?;
        if base.len() != expected_values {
            return Err(PrismError::InvalidInput(format!(
                "IVF vector data length {} must equal n * dimension ({expected_values})",
                base.len()
            )));
        }
        if let VecStore::F32(values) = base {
            validate_f32_distance_domain(values, dim, "IVF vector")?;
        }
        if assignments.len() != n {
            return Err(PrismError::InvalidInput(format!(
                "IVF assignment count {} must equal point count {n}",
                assignments.len()
            )));
        }
        if let Some((point, cluster)) = assignments
            .iter()
            .enumerate()
            .find(|(_, cluster)| **cluster as usize >= n_clusters)
        {
            return Err(PrismError::InvalidInput(format!(
                "IVF point {point} has out-of-range cluster {cluster}"
            )));
        }
        base_meta.validate(n)?;

        // Frequencies are sparse in the declared tag space, so a `cols`-sized
        // vector turns a large sparse vocabulary into a memory-denial input.
        let mut tag_freq: HashMap<u32, usize> = HashMap::new();
        tag_freq
            .try_reserve(base_meta.indices.len())
            .map_err(|error| {
                PrismError::Overflow(format!("cannot allocate IVF tag frequencies: {error}"))
            })?;
        for &tag in &base_meta.indices {
            let frequency = tag_freq.entry(tag as u32).or_insert(0);
            *frequency = frequency
                .checked_add(1)
                .ok_or_else(|| PrismError::Overflow("IVF tag frequency exceeds usize".into()))?;
        }

        let mut cluster_sizes = vec![0u32; n_clusters];
        for &a in assignments {
            cluster_sizes[a as usize] = cluster_sizes[a as usize]
                .checked_add(1)
                .ok_or_else(|| PrismError::Overflow("IVF cluster size exceeds u32".into()))?;
        }
        let mut cluster_starts = vec![0u32; n_clusters + 1];
        for i in 0..n_clusters {
            cluster_starts[i + 1] = cluster_starts[i]
                .checked_add(cluster_sizes[i])
                .ok_or_else(|| PrismError::Overflow("IVF cluster offsets exceed u32".into()))?;
        }

        let mut position = cluster_starts[..n_clusters].to_vec();
        let mut new_order = vec![0u32; n];
        for (i, &ci_raw) in assignments.iter().enumerate().take(n) {
            let ci = ci_raw as usize;
            let new_id = position[ci] as usize;
            new_order[new_id] = i as u32;
            position[ci] = position[ci]
                .checked_add(1)
                .ok_or_else(|| PrismError::Overflow("IVF reorder offsets exceed u32".into()))?;
        }

        macro_rules! reorder_and_sort {
            ($base_data:expr, $zero:expr, $T:ty) => {{
                let mut vecs = vec![$zero; n * dim];
                for (new_id, &old_id) in new_order.iter().enumerate() {
                    let src = &$base_data[old_id as usize * dim..(old_id as usize + 1) * dim];
                    vecs[new_id * dim..(new_id + 1) * dim].copy_from_slice(src);
                }

                for ci in 0..n_clusters {
                    let cs = cluster_starts[ci] as usize;
                    let ce = cluster_starts[ci + 1] as usize;
                    if ce - cs <= 1 {
                        continue;
                    }

                    let mut sort_keys: Vec<(u32, usize)> = (0..ce - cs)
                        .map(|local| {
                            let old_id = new_order[cs + local] as usize;
                            let ms = base_meta.indptr[old_id] as usize;
                            let me = base_meta.indptr[old_id + 1] as usize;
                            let tag = base_meta.indices[ms..me]
                                .iter()
                                .max_by_key(|&&t| tag_freq.get(&(t as u32)).copied().unwrap_or(0))
                                .map(|&t| t as u32)
                                .unwrap_or(u32::MAX);
                            (tag, local)
                        })
                        .collect();
                    sort_keys.sort_unstable_by_key(|&(tag, _)| tag);

                    let old_vecs: Vec<$T> = vecs[cs * dim..ce * dim].to_vec();
                    let old_ids: Vec<u32> = new_order[cs..ce].to_vec();
                    for (new_local, &(_, old_local)) in sort_keys.iter().enumerate() {
                        vecs[(cs + new_local) * dim..(cs + new_local + 1) * dim]
                            .copy_from_slice(&old_vecs[old_local * dim..(old_local + 1) * dim]);
                        new_order[cs + new_local] = old_ids[old_local];
                    }
                }
                vecs
            }};
        }

        let vectors = match base {
            VecStore::U8(data) => VecStore::U8(reorder_and_sort!(data, 0u8, u8)),
            VecStore::F32(data) => VecStore::F32(reorder_and_sort!(data, 0.0f32, f32)),
        };

        // old_to_new must come AFTER the intra-cluster tag-affinity sort.
        let mut old_to_new = vec![0u32; n];
        for (new_id, &old_id) in new_order.iter().enumerate() {
            old_to_new[old_id as usize] = new_id as u32;
        }

        let mut all_tag_entries: Vec<Vec<(u32, u32, u32)>> = Vec::with_capacity(n_clusters);
        let mut all_posting_ids: Vec<u32> = Vec::new();

        let mut cluster_maps: Vec<std::collections::HashMap<u32, Vec<u32>>> = (0..n_clusters)
            .map(|_| std::collections::HashMap::new())
            .collect();

        for old_id in 0..n {
            let new_id = old_to_new[old_id] as usize;
            let ci = assignments[old_id] as usize;
            let local_id = new_id - cluster_starts[ci] as usize;

            let start = base_meta.indptr[old_id] as usize;
            let end = base_meta.indptr[old_id + 1] as usize;
            let mut row_tags: Vec<i32> = base_meta.indices[start..end].to_vec();
            row_tags.sort_unstable();
            row_tags.dedup();
            for tag in row_tags {
                cluster_maps[ci]
                    .entry(tag as u32)
                    .or_default()
                    .push(local_id as u32);
            }
        }

        for cluster_map in cluster_maps.iter_mut().take(n_clusters) {
            let mut entries: Vec<(u32, Vec<u32>)> = cluster_map.drain().collect();
            entries.sort_unstable_by_key(|&(tag, _)| tag);

            let mut cluster_entries = Vec::new();
            cluster_entries
                .try_reserve_exact(entries.len())
                .map_err(|error| {
                    PrismError::Overflow(format!(
                        "cannot allocate {} IVF tag-index entries: {error}",
                        entries.len()
                    ))
                })?;
            for (tag, mut ids) in entries {
                ids.sort_unstable();
                let posting_end =
                    all_posting_ids
                        .len()
                        .checked_add(ids.len())
                        .ok_or_else(|| {
                            PrismError::Overflow(
                                "IVF posting storage exceeds addressable memory".into(),
                            )
                        })?;
                if posting_end > u32::MAX as usize {
                    return Err(PrismError::Overflow(
                        "IVF posting storage exceeds the u32 offset space".into(),
                    ));
                }
                let posting_start = u32::try_from(all_posting_ids.len())
                    .map_err(|_| PrismError::Overflow("IVF posting start exceeds u32".into()))?;
                let posting_len = u32::try_from(ids.len())
                    .map_err(|_| PrismError::Overflow("IVF posting length exceeds u32".into()))?;
                all_posting_ids.try_reserve(ids.len()).map_err(|error| {
                    PrismError::Overflow(format!(
                        "cannot grow IVF posting storage to {posting_end} entries: {error}"
                    ))
                })?;
                all_posting_ids.extend_from_slice(&ids);
                cluster_entries.push((tag, posting_start, posting_len));
            }
            all_tag_entries.push(cluster_entries);
        }

        let tag_offset_count = n_clusters
            .checked_add(1)
            .ok_or_else(|| PrismError::Overflow("IVF tag-offset shape overflows usize".into()))?;
        let mut tag_offsets = Vec::new();
        tag_offsets
            .try_reserve_exact(tag_offset_count)
            .map_err(|error| {
                PrismError::Overflow(format!(
                    "cannot allocate {tag_offset_count} IVF tag offsets: {error}"
                ))
            })?;
        let mut tag_index = Vec::new();
        for entries in &all_tag_entries {
            let offset = u32::try_from(tag_index.len())
                .map_err(|_| PrismError::Overflow("IVF tag-index offsets exceed u32".into()))?;
            tag_offsets.push(offset);
            tag_index.try_reserve(entries.len()).map_err(|error| {
                PrismError::Overflow(format!("cannot grow IVF tag index: {error}"))
            })?;
            tag_index.extend_from_slice(entries);
        }
        tag_offsets.push(
            u32::try_from(tag_index.len())
                .map_err(|_| PrismError::Overflow("IVF tag-index offsets exceed u32".into()))?,
        );

        let mut tag_clusters: HashMap<u32, Vec<u16>> = HashMap::new();
        tag_clusters.try_reserve(tag_index.len()).map_err(|error| {
            PrismError::Overflow(format!("cannot allocate IVF tag-cluster map: {error}"))
        })?;
        for ci in 0..n_clusters {
            let start = tag_offsets[ci] as usize;
            let end = tag_offsets[ci + 1] as usize;
            for &(tag, _, _) in &tag_index[start..end] {
                let clusters = tag_clusters.entry(tag).or_default();
                clusters.try_reserve(1).map_err(|error| {
                    PrismError::Overflow(format!(
                        "cannot grow IVF cluster list for tag {tag}: {error}"
                    ))
                })?;
                clusters.push(ci as u16);
            }
        }

        let binary = match &vectors {
            VecStore::U8(values) => BinaryStore::build_from_u8(values, dim)?,
            VecStore::F32(values) => BinaryStore::build_from_f32(values, dim)?,
        };

        Ok(Self {
            vectors,
            original_ids: new_order,
            cluster_starts,
            tag_offsets,
            tag_index,
            posting_ids: all_posting_ids,
            tag_clusters,
            n_tags: base_meta.cols,
            binary,
            dim,
            n_clusters,
        })
    }

    /// Look up local IDs matching a tag within a cluster.
    #[inline]
    fn lookup_tag(&self, cluster: usize, tag: usize) -> &[u32] {
        let Ok(tag) = u32::try_from(tag) else {
            return &[];
        };
        let start = self.tag_offsets[cluster] as usize;
        let end = self.tag_offsets[cluster + 1] as usize;
        let entries = &self.tag_index[start..end];
        match entries.binary_search_by_key(&tag, |&(t, _, _)| t) {
            Ok(idx) => {
                let (_, ps, pl) = entries[idx];
                let start = ps as usize;
                let end = start + pl as usize;
                &self.posting_ids[start..end]
            }
            Err(_) => &[],
        }
    }

    /// Scan local ids in a cluster against the query. The Hamming pre-filter
    /// applies when the candidate count exceeds the rerank budget.
    #[allow(clippy::too_many_arguments)]
    fn scan_cluster(
        &self,
        ci: usize,
        lids: impl ExactSizeIterator<Item = u32>,
        query: &QueryVec,
        q_binary: &[u64],
        ef: usize,
        binary_rerank: usize,
        heap: &mut BinaryHeap<(u64, u32)>,
    ) {
        let dim = self.dim;
        let cluster_base = self.cluster_starts[ci] as usize;
        let rerank_budget = binary_rerank.saturating_mul(ef);

        if rerank_budget > 0 && lids.len() > rerank_budget {
            let mut candidates: Vec<(u64, u32)> = lids
                .map(|lid| {
                    let gid = (cluster_base + lid as usize) as u32;
                    (
                        distance::hamming(q_binary, self.binary.code_unchecked(gid)),
                        lid,
                    )
                })
                .collect();
            let budget = rerank_budget.min(candidates.len());
            candidates.select_nth_unstable_by_key(budget - 1, |&(d, _)| d);
            candidates.truncate(budget);
            for &(_, lid) in &candidates {
                let gid = (cluster_base + lid as usize) as u32;
                let dist = compute_dist(&self.vectors, gid as usize, query, dim);
                let orig_id = self.original_ids[gid as usize];
                heap_insert(heap, dist, orig_id, ef);
            }
        } else {
            for lid in lids {
                let gid = (cluster_base + lid as usize) as u32;
                let dist = compute_dist(&self.vectors, gid as usize, query, dim);
                let orig_id = self.original_ids[gid as usize];
                heap_insert(heap, dist, orig_id, ef);
            }
        }
    }

    /// Multi-query cell batching (MQCB): process queries grouped by geometric
    /// cluster for L3-cache reuse.
    #[allow(clippy::too_many_arguments)]
    pub fn batch_search_mqcb(
        &self,
        queries: &QueryStore,
        nq: usize,
        query_tags: &[Vec<usize>],
        query_top_clusters: &[Vec<usize>],
        k: usize,
        ef: usize,
        n_probe: usize,
        binary_rerank: usize,
    ) -> PrismResult<Vec<Vec<u32>>> {
        let dim = self.dim;
        if query_tags.len() != nq {
            return Err(PrismError::InvalidInput(format!(
                "query tag row count {} must equal nq {nq}",
                query_tags.len()
            )));
        }
        if query_top_clusters.len() != nq {
            return Err(PrismError::InvalidInput(format!(
                "query cluster row count {} must equal nq {nq}",
                query_top_clusters.len()
            )));
        }
        if n_probe == 0 {
            return Err(PrismError::InvalidInput(
                "n_probe must be greater than zero".into(),
            ));
        }
        let ef = ef.max(k);
        let expected_values = nq.checked_mul(dim).ok_or_else(|| {
            PrismError::Overflow("query batch shape exceeds addressable memory".into())
        })?;
        match (queries, &self.vectors) {
            (QueryStore::U8(data), VecStore::U8(_)) => {
                if data.len() != expected_values {
                    return Err(PrismError::InvalidInput(format!(
                        "query data length {} must equal nq * dim ({expected_values})",
                        data.len()
                    )));
                }
            }
            (QueryStore::F32(data), VecStore::F32(_)) => {
                if data.len() != expected_values {
                    return Err(PrismError::InvalidInput(format!(
                        "query data length {} must equal nq * dim ({expected_values})",
                        data.len()
                    )));
                }
                validate_f32_distance_domain(data, dim, "IVF query")?;
            }
            _ => {
                return Err(PrismError::InvalidInput(
                    "query dtype must match the indexed vector dtype".into(),
                ));
            }
        }
        for (qi, tags) in query_tags.iter().enumerate() {
            if let Some(&tag) = tags.iter().find(|&&tag| tag >= self.n_tags) {
                return Err(PrismError::InvalidInput(format!(
                    "query {qi} contains out-of-range tag {tag} for {} tags",
                    self.n_tags
                )));
            }
        }
        for (qi, top_clusters) in query_top_clusters.iter().enumerate() {
            if let Some(&cluster) = top_clusters
                .iter()
                .find(|&&cluster| cluster >= self.n_clusters)
            {
                return Err(PrismError::InvalidInput(format!(
                    "query {qi} contains out-of-range cluster {cluster} for {} clusters",
                    self.n_clusters
                )));
            }
        }
        if k == 0 {
            return Ok(vec![Vec::new(); nq]);
        }

        // Codes are derived with the rotation owned by this exact reordered
        // index, so an external store can never silently prefilter wrong IDs.
        let query_binary: Vec<Vec<u64>> = if binary_rerank > 0 {
            match queries {
                QueryStore::U8(data) => data
                    .par_chunks_exact(dim)
                    .map(|query| self.binary.encode_query_u8(query))
                    .collect::<PrismResult<Vec<_>>>()?,
                QueryStore::F32(data) => data
                    .par_chunks_exact(dim)
                    .map(|query| self.binary.encode_query(query))
                    .collect::<PrismResult<Vec<_>>>()?,
            }
        } else {
            Vec::new()
        };

        let mut cluster_queries: Vec<Vec<usize>> = vec![vec![]; self.n_clusters];
        for (qi, top_clusters) in query_top_clusters.iter().enumerate().take(nq) {
            let mut seen = HashSet::with_capacity(n_probe.min(top_clusters.len()));
            for &ci in top_clusters {
                if !seen.insert(ci) {
                    continue;
                }
                cluster_queries[ci].push(qi);
                if seen.len() >= n_probe {
                    break;
                }
            }
        }

        // One uncontended lock per (cluster, query) update keeps the parallel
        // implementation safe even if a caller supplied duplicate clusters.
        let heaps: Vec<Mutex<BinaryHeap<(u64, u32)>>> =
            (0..nq).map(|_| Mutex::new(BinaryHeap::new())).collect();

        // Sequential cluster iteration for prefetcher-friendly memory access
        for (ci, qi_list) in cluster_queries.iter().enumerate() {
            if qi_list.is_empty() {
                continue;
            }

            qi_list.par_iter().for_each(|&qi| {
                let query = match queries {
                    QueryStore::U8(data) => QueryVec::U8(&data[qi * dim..(qi + 1) * dim]),
                    QueryStore::F32(data) => QueryVec::F32(&data[qi * dim..(qi + 1) * dim]),
                };
                let tags = &query_tags[qi];
                let q_binary = query_binary.get(qi).map_or(&[][..], Vec::as_slice);
                let mut heap = heaps[qi]
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);

                match tags.len() {
                    0 => {
                        let len = self.cluster_starts[ci + 1] - self.cluster_starts[ci];
                        self.scan_cluster(
                            ci,
                            0..len,
                            &query,
                            q_binary,
                            ef,
                            binary_rerank,
                            &mut heap,
                        );
                    }
                    1 => {
                        let matching = self.lookup_tag(ci, tags[0]);
                        self.scan_cluster(
                            ci,
                            matching.iter().copied(),
                            &query,
                            q_binary,
                            ef,
                            binary_rerank,
                            &mut heap,
                        );
                    }
                    _ => {
                        let lists: Vec<&[u32]> =
                            tags.iter().map(|&t| self.lookup_tag(ci, t)).collect();
                        let matching = intersect_postings(lists);
                        self.scan_cluster(
                            ci,
                            matching.iter().copied(),
                            &query,
                            q_binary,
                            ef,
                            binary_rerank,
                            &mut heap,
                        );
                    }
                }
            });
        }

        Ok(heaps
            .into_par_iter()
            .map(|mutex| {
                let heap = mutex
                    .into_inner()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let mut results: Vec<(u64, u32)> = heap.into_vec();
                results.sort_unstable_by_key(|&(d, _)| d);
                results.iter().take(k).map(|&(_, id)| id).collect()
            })
            .collect())
    }
}

/// Bounded max-heap insert via PeekMut (single sift-down).
#[inline]
fn heap_insert(heap: &mut BinaryHeap<(u64, u32)>, dist: u64, id: u32, cap: usize) {
    if heap.len() < cap {
        heap.push((dist, id));
    } else if let Some(mut top) = heap.peek_mut() {
        if dist < top.0 {
            *top = (dist, id);
        }
    }
}

/// Sorted k-way intersection of posting lists, smallest list first so the
/// accumulator only shrinks.
fn intersect_postings(mut lists: Vec<&[u32]>) -> Vec<u32> {
    lists.sort_unstable_by_key(|l| l.len());
    let mut acc: Vec<u32> = lists[0].to_vec();
    for list in &lists[1..] {
        if acc.is_empty() {
            break;
        }
        let mut out = Vec::with_capacity(acc.len().min(list.len()));
        let (mut i, mut j) = (0, 0);
        while i < acc.len() && j < list.len() {
            match acc[i].cmp(&list[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    out.push(acc[i]);
                    i += 1;
                    j += 1;
                }
            }
        }
        acc = out;
    }
    acc
}

/// Sorted intersection of two sorted u16 slices.
pub fn sorted_intersect_u16(a: &[u16], b: &[u16]) -> Vec<u16> {
    let mut result = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                result.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    result
}

/// K-means clustering. Returns (assignments, centroids as VecStore matching input type).
pub fn kmeans(
    base: &VecStore,
    n: usize,
    dim: usize,
    c: usize,
    iters: usize,
) -> PrismResult<(Vec<u16>, VecStore)> {
    if n == 0 {
        return Err(PrismError::InvalidInput(
            "k-means requires at least one vector".into(),
        ));
    }
    if dim == 0 {
        return Err(PrismError::InvalidInput(
            "k-means dimension must be greater than zero".into(),
        ));
    }
    if n > u32::MAX as usize {
        return Err(PrismError::Overflow(
            "k-means point count exceeds the u32 identifier space".into(),
        ));
    }
    if !(1..=n).contains(&c) {
        return Err(PrismError::InvalidInput(
            "k-means cluster count must be between 1 and the point count".into(),
        ));
    }
    if c > u16::MAX as usize + 1 {
        return Err(PrismError::Overflow(
            "k-means cluster count exceeds the u16 assignment space".into(),
        ));
    }
    if iters == 0 {
        return Err(PrismError::InvalidInput(
            "k-means iteration count must be greater than zero".into(),
        ));
    }
    let expected_values = n.checked_mul(dim).ok_or_else(|| {
        PrismError::Overflow("k-means vector shape exceeds addressable memory".into())
    })?;
    if base.len() != expected_values {
        return Err(PrismError::InvalidInput(format!(
            "k-means vector data length {} must equal n * dimension ({expected_values})",
            base.len()
        )));
    }
    if let VecStore::F32(values) = base {
        validate_f32_distance_domain(values, dim, "k-means vector")?;
    }

    let mut rng = StdRng::seed_from_u64(42);
    let mut centroid_ids: Vec<usize> = (0..n).collect();
    centroid_ids.shuffle(&mut rng);
    centroid_ids.truncate(c);

    let centroid_values = c.checked_mul(dim).ok_or_else(|| {
        PrismError::Overflow("k-means centroid shape exceeds addressable memory".into())
    })?;
    let mut centroids_f32 = vec![0.0f32; centroid_values];
    match base {
        VecStore::U8(data) => {
            for (ci, &vid) in centroid_ids.iter().enumerate() {
                for d in 0..dim {
                    centroids_f32[ci * dim + d] = data[vid * dim + d] as f32;
                }
            }
        }
        VecStore::F32(data) => {
            for (ci, &vid) in centroid_ids.iter().enumerate() {
                centroids_f32[ci * dim..(ci + 1) * dim]
                    .copy_from_slice(&data[vid * dim..(vid + 1) * dim]);
            }
        }
    }

    let mut assignments = vec![0u16; n];

    for _ in 0..iters {
        let new_assignments: Vec<u16> = match base {
            VecStore::U8(data) => {
                let centroids_u8: Vec<u8> = centroids_f32
                    .iter()
                    .map(|&x| x.round().clamp(0.0, 255.0) as u8)
                    .collect();
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let v = &data[i * dim..(i + 1) * dim];
                        let mut best_c = 0u16;
                        let mut best_d = u64::MAX;
                        for ci in 0..c {
                            let cent = &centroids_u8[ci * dim..(ci + 1) * dim];
                            let d = distance::l2_sq8(v, cent);
                            if d < best_d {
                                best_d = d;
                                best_c = ci as u16;
                            }
                        }
                        best_c
                    })
                    .collect()
            }
            VecStore::F32(data) => (0..n)
                .into_par_iter()
                .map(|i| {
                    let v = &data[i * dim..(i + 1) * dim];
                    let mut best_c = 0u16;
                    let mut best_d = f32::INFINITY;
                    for ci in 0..c {
                        let cent = &centroids_f32[ci * dim..(ci + 1) * dim];
                        let d = distance::l2_squared(v, cent);
                        if d < best_d {
                            best_d = d;
                            best_c = ci as u16;
                        }
                    }
                    best_c
                })
                .collect(),
        };
        assignments = new_assignments;

        // Centroid update accumulates in f64 to avoid f32 cancellation.
        let mut sums = vec![0.0f64; c * dim];
        let mut counts = vec![0u32; c];
        match base {
            VecStore::U8(data) => {
                for i in 0..n {
                    let ci = assignments[i] as usize;
                    counts[ci] = counts[ci].checked_add(1).ok_or_else(|| {
                        PrismError::Overflow("k-means cluster size exceeds u32".into())
                    })?;
                    for d in 0..dim {
                        sums[ci * dim + d] += data[i * dim + d] as f64;
                    }
                }
            }
            VecStore::F32(data) => {
                for i in 0..n {
                    let ci = assignments[i] as usize;
                    counts[ci] = counts[ci].checked_add(1).ok_or_else(|| {
                        PrismError::Overflow("k-means cluster size exceeds u32".into())
                    })?;
                    for d in 0..dim {
                        sums[ci * dim + d] += data[i * dim + d] as f64;
                    }
                }
            }
        }
        for ci in 0..c {
            if counts[ci] > 0 {
                let inv = 1.0 / counts[ci] as f64;
                for d in 0..dim {
                    centroids_f32[ci * dim + d] = (sums[ci * dim + d] * inv) as f32;
                }
            }
        }

        // Empty-cluster repair inspired by Faiss, but this variant moves an
        // actual member instead of perturbing a copied centroid:
        // https://github.com/facebookresearch/faiss/blob/main/faiss/impl/ClusteringHelpers.cpp
        for ci in 0..c {
            if counts[ci] > 0 {
                continue;
            }
            let donor = (0..c).max_by_key(|&d| counts[d]).unwrap();
            if counts[donor] <= 1 {
                break;
            }
            let members: Vec<usize> = (0..n)
                .filter(|&i| assignments[i] as usize == donor)
                .collect();
            let p = members[rng.gen_range(0..members.len())];
            match base {
                VecStore::U8(data) => {
                    for d in 0..dim {
                        centroids_f32[ci * dim + d] = data[p * dim + d] as f32;
                    }
                }
                VecStore::F32(data) => {
                    centroids_f32[ci * dim..(ci + 1) * dim]
                        .copy_from_slice(&data[p * dim..(p + 1) * dim]);
                }
            }
            assignments[p] = ci as u16;
            counts[donor] -= 1;
            counts[ci] = 1;
        }
    }

    let centroids = match base {
        VecStore::U8(_) => VecStore::U8(
            centroids_f32
                .iter()
                .map(|&x| x.round().clamp(0.0, 255.0) as u8)
                .collect(),
        ),
        VecStore::F32(_) => VecStore::F32(centroids_f32),
    };

    Ok((assignments, centroids))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 6 points, 4 tags, 2 hand-assigned clusters. Tag sets per point:
    /// 0:{0,1,2} 1:{0,1} 2:{0,2} 3:{1,2} 4:{0,1,2} 5:{3}.
    fn fixture() -> IvfIndex {
        let points: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.2, 0.0],
            vec![0.3, 0.0],
            vec![5.0, 5.0],
            vec![5.1, 5.0],
        ];
        let tag_sets: Vec<Vec<i32>> = vec![
            vec![0, 0, 1, 2],
            vec![0, 1],
            vec![0, 2],
            vec![1, 2],
            vec![0, 1, 2],
            vec![3],
        ];
        let flat: Vec<f32> = points.iter().flatten().copied().collect();
        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        for tags in &tag_sets {
            indices.extend_from_slice(tags);
            indptr.push(indices.len() as i64);
        }
        let meta = SpMat {
            rows: points.len(),
            cols: 4,
            indptr,
            indices,
        };
        let assignments: Vec<u16> = vec![0, 0, 0, 0, 1, 1];
        let base = VecStore::F32(flat.clone());
        IvfIndex::build(&base, &meta, &assignments, points.len(), 2, 2).unwrap()
    }

    fn run_query(
        index: &IvfIndex,
        query: &[f32],
        tags: Vec<usize>,
        k: usize,
        binary_rerank: usize,
    ) -> Vec<u32> {
        let mut results = index
            .batch_search_mqcb(
                &QueryStore::F32(query),
                1,
                &[tags],
                &[vec![0, 1]],
                k,
                10,
                2,
                binary_rerank,
            )
            .unwrap();
        results.pop().unwrap()
    }

    #[test]
    fn batch_zero_tags_scans_whole_clusters() {
        let index = fixture();
        let mut ids = run_query(&index, &[5.05, 5.0], Vec::new(), 2, 0);
        ids.sort_unstable();
        assert_eq!(ids, vec![4, 5]);
    }

    #[test]
    fn batch_three_tags_enforces_full_conjunction() {
        let index = fixture();
        let ids = run_query(&index, &[0.05, 0.0], vec![0, 1, 2], 4, 0);
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        // Only points 0 and 4 carry all three tags; point 1 matches just {0,1}
        // and must not leak through a first-two-tags-only intersection.
        assert_eq!(sorted, vec![0, 4]);
    }

    #[test]
    fn duplicate_metadata_and_cluster_ids_do_not_duplicate_results() {
        let index = fixture();
        let query = [0.0f32, 0.0];
        let results = index
            .batch_search_mqcb(
                &QueryStore::F32(&query),
                1,
                &[vec![0]],
                &[vec![0, 0, 1, 1]],
                6,
                6,
                4,
                0,
            )
            .unwrap();
        let mut unique = results[0].clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(results[0].len(), unique.len());
    }

    #[test]
    fn ivf_results_use_original_ids_after_reordering() {
        // Cluster assignments deliberately interleave insertion-order IDs, so
        // internal ID 0 belongs to original point 1 after cluster reordering.
        let base = VecStore::F32(vec![
            -1.0, -2.0, // original 0, cluster 1
            1.0, 2.0, // original 1, cluster 0 and exact nearest
            5.0, 5.0, // original 2, cluster 1
            -1.0, -2.0, // original 3, cluster 0 and binary opposite
        ]);
        let meta = SpMat {
            rows: 4,
            cols: 1,
            indptr: vec![0, 1, 2, 3, 4],
            indices: vec![0, 0, 0, 0],
        };
        let index = IvfIndex::build(&base, &meta, &[1, 0, 1, 0], 4, 2, 2).unwrap();
        assert_eq!(index.original_ids, vec![1, 3, 0, 2]);
        let ids = run_query(&index, &[1.0, 2.0], vec![0], 1, 1);

        assert_eq!(ids, vec![1]);
    }

    #[test]
    fn binary_prefilter_executes_on_large_posting() {
        let index = fixture();
        let query = [0.0f32, 0.0];
        let results = index
            .batch_search_mqcb(
                &QueryStore::F32(&query),
                1,
                &[vec![0]],
                &[vec![0, 1]],
                1,
                1,
                2,
                1,
            )
            .unwrap();
        assert_eq!(results[0].len(), 1);
        assert!([0, 1, 2, 4].contains(&results[0][0]));
    }

    #[test]
    fn kmeans_reseeds_empty_clusters() {
        // 60 identical points + 4 outliers, 8 clusters: without repair most
        // centroids never win an assignment and stay empty forever.
        let n = 64;
        let mut flat = vec![0.0f32; n * 2];
        for (i, off) in [(60, 50.0f32), (61, -50.0), (62, 100.0), (63, -100.0)] {
            flat[i * 2] = off;
            flat[i * 2 + 1] = off;
        }
        let (assignments, _) = kmeans(&VecStore::F32(flat), n, 2, 8, 3).unwrap();
        let mut seen = [false; 8];
        for &a in &assignments {
            seen[a as usize] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "every cluster must keep at least one member, got {assignments:?}"
        );
    }

    #[test]
    fn csr_and_ivf_build_reject_malformed_public_inputs() {
        assert!(matches!(
            SpMat::new(1, 1, vec![1, 1], vec![]),
            Err(PrismError::InvalidInput(_))
        ));
        assert!(matches!(
            SpMat::new(1, 1, vec![0, 1], vec![-1]),
            Err(PrismError::InvalidInput(_))
        ));

        let metadata = SpMat::new(1, 1, vec![0, 1], vec![0]).unwrap();
        assert_eq!(metadata.rows(), 1);
        assert_eq!(metadata.cols(), 1);
        assert_eq!(metadata.indptr(), &[0, 1]);
        assert_eq!(metadata.indices(), &[0]);
        assert!(matches!(
            IvfIndex::build(&VecStore::F32(vec![0.0]), &metadata, &[0], 1, 2, 1),
            Err(PrismError::InvalidInput(_))
        ));
        assert!(matches!(
            IvfIndex::build(&VecStore::F32(vec![0.0]), &metadata, &[1], 1, 1, 1),
            Err(PrismError::InvalidInput(_))
        ));
        assert!(matches!(
            kmeans(&VecStore::F32(vec![0.0]), 1, 1, 1, 0),
            Err(PrismError::InvalidInput(_))
        ));
    }

    #[test]
    fn sparse_tag_space_never_allocates_by_declared_column_count() {
        assert!(matches!(
            SpMat::new(0, usize::MAX, vec![0], vec![]),
            Err(PrismError::Overflow(_))
        ));

        let max_supported_cols = i32::MAX as usize + 1;
        let metadata = SpMat::new(1, max_supported_cols, vec![0, 1], vec![i32::MAX]).unwrap();
        let index = IvfIndex::build(&VecStore::F32(vec![0.0]), &metadata, &[0], 1, 1, 1).unwrap();
        assert_eq!(index.n_tags(), max_supported_cols);
        assert_eq!(index.clusters_for_tag(i32::MAX as usize), Some(&[0][..]));
        assert_eq!(index.clusters_for_tag(0), None);

        let invalid_metadata = SpMat {
            rows: 1,
            cols: usize::MAX,
            indptr: vec![0, 0],
            indices: vec![],
        };
        assert!(matches!(
            IvfIndex::build(&VecStore::F32(vec![0.0]), &invalid_metadata, &[0], 1, 1, 1,),
            Err(PrismError::Overflow(_))
        ));
    }

    #[test]
    fn ivf_rejects_extreme_finite_base_and_query_coordinates() {
        let metadata = SpMat::new(1, 1, vec![0, 1], vec![0]).unwrap();
        assert!(matches!(
            IvfIndex::build(&VecStore::F32(vec![f32::MAX]), &metadata, &[0], 1, 1, 1,),
            Err(PrismError::InvalidInput(_))
        ));
        assert!(matches!(
            kmeans(&VecStore::F32(vec![-f32::MAX]), 1, 1, 1, 1),
            Err(PrismError::InvalidInput(_))
        ));

        let index = fixture();
        assert!(matches!(
            index.batch_search_mqcb(
                &QueryStore::F32(&[f32::MAX, -f32::MAX]),
                1,
                &[vec![]],
                &[vec![0]],
                1,
                1,
                1,
                1,
            ),
            Err(PrismError::InvalidInput(_))
        ));
    }

    #[test]
    fn ivf_search_rejects_shape_dtype_and_cluster_mismatches() {
        let index = fixture();
        let query = [0.0f32, 0.0];

        assert!(matches!(
            index.batch_search_mqcb(
                &QueryStore::U8(&[0, 0]),
                1,
                &[vec![]],
                &[vec![0]],
                1,
                1,
                1,
                0,
            ),
            Err(PrismError::InvalidInput(_))
        ));
        assert!(matches!(
            index.batch_search_mqcb(
                &QueryStore::F32(&query),
                1,
                &[vec![]],
                &[vec![index.n_clusters()]],
                1,
                1,
                1,
                0,
            ),
            Err(PrismError::InvalidInput(_))
        ));
        assert!(matches!(
            index.batch_search_mqcb(
                &QueryStore::F32(&query),
                1,
                &[vec![]],
                &[vec![0]],
                1,
                1,
                0,
                0,
            ),
            Err(PrismError::InvalidInput(_))
        ));
        assert!(matches!(
            index.batch_search_mqcb(
                &QueryStore::F32(&query),
                1,
                &[vec![index.n_tags()]],
                &[vec![0]],
                1,
                1,
                1,
                0,
            ),
            Err(PrismError::InvalidInput(_))
        ));
    }
}
