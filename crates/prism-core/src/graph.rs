use super::error::{PrismError, PrismResult};

/// CSR (Compressed Sparse Row) graph for neighbor storage.
///
/// For n nodes, `offsets` has n+1 entries.
/// Neighbors of node i are `neighbors[offsets[i]..offsets[i+1]]`.
pub struct Graph {
    pub(crate) offsets: Vec<u32>,
    pub(crate) neighbors: Vec<u32>,
    pub(crate) n: usize,
}

impl Graph {
    /// Build from adjacency lists. Each entry in `adj` is the neighbor list for that node.
    pub fn from_adj(adj: &[Vec<u32>]) -> PrismResult<Self> {
        let n = adj.len();
        validate_node_count(n)?;
        let offsets_len = n.checked_add(1).ok_or_else(|| {
            PrismError::Overflow("graph offset count exceeds addressable memory".into())
        })?;
        let edge_count = adj.iter().try_fold(0usize, |total, list| {
            total.checked_add(list.len()).ok_or_else(|| {
                PrismError::Overflow("graph edge count exceeds addressable memory".into())
            })
        })?;
        if edge_count > u32::MAX as usize {
            return Err(PrismError::Overflow(
                "graph edge count exceeds the u32 CSR offset space".into(),
            ));
        }

        let mut offsets = Vec::new();
        offsets.try_reserve_exact(offsets_len).map_err(|error| {
            PrismError::Overflow(format!("cannot allocate graph offsets: {error}"))
        })?;
        let mut neighbors = Vec::new();
        neighbors.try_reserve_exact(edge_count).map_err(|error| {
            PrismError::Overflow(format!("cannot allocate graph neighbors: {error}"))
        })?;
        let mut offset = 0u32;
        for (node, list) in adj.iter().enumerate() {
            if let Some(&neighbor) = list.iter().find(|&&neighbor| neighbor as usize >= n) {
                return Err(PrismError::InvalidInput(format!(
                    "graph node {node} has out-of-range neighbor {neighbor} for {n} nodes"
                )));
            }
            offsets.push(offset);
            neighbors.extend_from_slice(list);
            let list_len = u32::try_from(list.len()).map_err(|_| {
                PrismError::Overflow("graph adjacency list exceeds the u32 offset space".into())
            })?;
            offset = offset
                .checked_add(list_len)
                .ok_or_else(|| PrismError::Overflow("graph edge offsets exceed u32".into()))?;
        }
        offsets.push(offset);
        Ok(Self {
            offsets,
            neighbors,
            n,
        })
    }

    /// Reassemble one persisted CSR graph after validating all offset and
    /// neighbor bounds before any search can index through them. Pass this
    /// component with the remaining validated parts to
    /// [`crate::PrismIndex::from_parts`] to reassemble a complete index.
    pub fn from_parts(offsets: Vec<u32>, neighbors: Vec<u32>, n: usize) -> PrismResult<Self> {
        validate_node_count(n)?;
        let expected_offsets = n.checked_add(1).ok_or_else(|| {
            PrismError::Overflow("graph offset count exceeds addressable memory".into())
        })?;
        if offsets.len() != expected_offsets {
            return Err(PrismError::InvalidFormat(format!(
                "graph has {} offsets; expected {expected_offsets}",
                offsets.len()
            )));
        }
        if neighbors.len() > u32::MAX as usize {
            return Err(PrismError::Overflow(
                "graph edge count exceeds the u32 CSR offset space".into(),
            ));
        }
        if offsets.first().copied() != Some(0) {
            return Err(PrismError::InvalidFormat(
                "graph offsets must start at zero".into(),
            ));
        }
        let edge_count = u32::try_from(neighbors.len()).map_err(|_| {
            PrismError::Overflow("graph edge count exceeds the u32 CSR offset space".into())
        })?;
        for (index, pair) in offsets.windows(2).enumerate() {
            if pair[0] > pair[1] || pair[1] > edge_count {
                return Err(PrismError::InvalidFormat(format!(
                    "graph offsets at node {index} are not monotone and within neighbor storage"
                )));
            }
        }
        if offsets.last().copied() != Some(edge_count) {
            return Err(PrismError::InvalidFormat(format!(
                "graph final offset must equal neighbor count {}",
                neighbors.len()
            )));
        }
        if let Some(&neighbor) = neighbors.iter().find(|&&neighbor| neighbor as usize >= n) {
            return Err(PrismError::InvalidFormat(format!(
                "graph contains out-of-range neighbor {neighbor} for {n} nodes"
            )));
        }

        Ok(Self {
            offsets,
            neighbors,
            n,
        })
    }

    /// Empty graph with n nodes and no edges.
    pub fn empty(n: usize) -> PrismResult<Self> {
        validate_node_count(n)?;
        let offsets_len = n.checked_add(1).ok_or_else(|| {
            PrismError::Overflow("graph offset count exceeds addressable memory".into())
        })?;
        Ok(Self {
            offsets: vec![0; offsets_len],
            neighbors: Vec::new(),
            n,
        })
    }

    /// Number of graph nodes.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the graph contains no nodes.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Immutable CSR offsets.
    pub fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    /// Immutable flat CSR neighbor storage.
    pub fn neighbor_ids(&self) -> &[u32] {
        &self.neighbors
    }

    /// Degree of node i.
    #[inline]
    pub fn degree(&self, i: u32) -> Option<usize> {
        ((i as usize) < self.n).then(|| self.degree_unchecked(i))
    }

    /// Degree of a previously validated node ID.
    #[inline]
    pub(crate) fn degree_unchecked(&self, i: u32) -> usize {
        let i = i as usize;
        (self.offsets[i + 1] - self.offsets[i]) as usize
    }

    /// Neighbors of node i.
    #[inline]
    pub fn neighbors(&self, i: u32) -> Option<&[u32]> {
        ((i as usize) < self.n).then(|| self.neighbors_unchecked(i))
    }

    /// Neighbors of a previously validated node ID.
    #[inline]
    pub(crate) fn neighbors_unchecked(&self, i: u32) -> &[u32] {
        let i = i as usize;
        let start = self.offsets[i] as usize;
        let end = self.offsets[i + 1] as usize;
        &self.neighbors[start..end]
    }

    /// Total number of edges (directed).
    pub fn num_edges(&self) -> usize {
        self.neighbors.len()
    }
}

fn validate_node_count(n: usize) -> PrismResult<()> {
    if n > u32::MAX as usize {
        return Err(PrismError::Overflow(
            "graph node count exceeds the u32 identifier space".into(),
        ));
    }
    Ok(())
}

/// Mutable adjacency list builder that converts to CSR.
pub struct AdjBuilder {
    adj: Vec<Vec<u32>>,
}

impl AdjBuilder {
    pub fn new(n: usize) -> PrismResult<Self> {
        validate_node_count(n)?;
        let mut adj = Vec::new();
        adj.try_reserve_exact(n).map_err(|error| {
            PrismError::Overflow(format!("cannot allocate graph builder: {error}"))
        })?;
        adj.resize_with(n, Vec::new);
        Ok(Self { adj })
    }

    /// Add a directed edge from `src` to `dst`.
    #[inline]
    pub fn add_edge(&mut self, src: u32, dst: u32) -> PrismResult<()> {
        self.validate_endpoint(src)?;
        self.validate_endpoint(dst)?;
        self.adj[src as usize].push(dst);
        Ok(())
    }

    /// Add bidirectional edge.
    #[inline]
    pub fn add_undirected(&mut self, a: u32, b: u32) -> PrismResult<()> {
        self.validate_endpoint(a)?;
        self.validate_endpoint(b)?;
        self.adj[a as usize].push(b);
        self.adj[b as usize].push(a);
        Ok(())
    }

    /// Get current neighbors.
    pub fn neighbors(&self, i: u32) -> Option<&[u32]> {
        self.adj.get(i as usize).map(Vec::as_slice)
    }

    /// Total directed edges currently stored.
    pub fn total_edges(&self) -> usize {
        self.adj.iter().map(|v| v.len()).sum()
    }

    /// Create a read-only, deduplicated CSR graph snapshot without consuming the builder.
    pub fn snapshot(&self) -> PrismResult<Graph> {
        // Clone one row at a time: cloning the whole Vec<Vec<_>> duplicates every
        // edge at once, making the local-graph snapshot a peak memory event.
        freeze_adjacency_rows(
            self.adj.iter().map(|list| {
                let mut copied = Vec::new();
                copied.try_reserve_exact(list.len()).map_err(|error| {
                    PrismError::Overflow(format!("cannot allocate graph snapshot row: {error}"))
                })?;
                copied.extend_from_slice(list);
                Ok(copied)
            }),
            self.adj.len(),
        )
    }

    /// Freeze into CSR graph, deduplicating edges.
    pub fn build(self) -> PrismResult<Graph> {
        // Move rows out so allocations release progressively instead of holding
        // the whole mutable graph beside a second full edge copy.
        let n = self.adj.len();
        freeze_adjacency_rows(self.adj.into_iter().map(Ok), n)
    }

    fn validate_endpoint(&self, node: u32) -> PrismResult<()> {
        if node as usize >= self.adj.len() {
            return Err(PrismError::InvalidInput(format!(
                "graph node {node} is outside the builder's {}-node range",
                self.adj.len()
            )));
        }
        Ok(())
    }
}

/// Sort, deduplicate, and freeze adjacency rows without materializing a second
/// complete adjacency-list graph. The iterator form lets snapshots copy only
/// one row at a time and lets consuming builds move rows directly.
fn freeze_adjacency_rows(
    rows: impl IntoIterator<Item = PrismResult<Vec<u32>>>,
    n: usize,
) -> PrismResult<Graph> {
    validate_node_count(n)?;
    let offsets_len = n.checked_add(1).ok_or_else(|| {
        PrismError::Overflow("graph offset count exceeds addressable memory".into())
    })?;
    let mut offsets = Vec::new();
    offsets
        .try_reserve_exact(offsets_len)
        .map_err(|error| PrismError::Overflow(format!("cannot allocate graph offsets: {error}")))?;
    let mut neighbors = Vec::new();
    offsets.push(0);

    let mut row_count = 0usize;
    for row in rows {
        let mut row = row?;
        row.sort_unstable();
        row.dedup();
        if let Some(&neighbor) = row.iter().find(|&&neighbor| neighbor as usize >= n) {
            return Err(PrismError::InvalidInput(format!(
                "graph node {row_count} has out-of-range neighbor {neighbor} for {n} nodes"
            )));
        }
        neighbors.try_reserve(row.len()).map_err(|error| {
            PrismError::Overflow(format!("cannot allocate graph neighbors: {error}"))
        })?;
        neighbors.extend(row);
        let offset = u32::try_from(neighbors.len()).map_err(|_| {
            PrismError::Overflow("graph edge count exceeds the u32 CSR offset space".into())
        })?;
        offsets.push(offset);
        row_count += 1;
    }

    if row_count != n {
        return Err(PrismError::InvalidFormat(format!(
            "graph builder produced {row_count} rows; expected {n}"
        )));
    }
    Ok(Graph {
        offsets,
        neighbors,
        n,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_from_adj() {
        let adj = vec![vec![1, 2], vec![0], vec![0, 1]];
        let g = Graph::from_adj(&adj).unwrap();
        assert_eq!(g.n, 3);
        assert_eq!(g.degree(0), Some(2));
        assert_eq!(g.degree(1), Some(1));
        assert_eq!(g.degree(2), Some(2));
        assert_eq!(g.degree(3), None);
        assert_eq!(g.neighbors(0), Some(&[1, 2][..]));
        assert_eq!(g.neighbors(1), Some(&[0][..]));
        assert_eq!(g.neighbors(2), Some(&[0, 1][..]));
        assert_eq!(g.neighbors(3), None);
    }

    #[test]
    fn persisted_graph_rejects_bad_offsets_and_neighbors() {
        assert!(matches!(
            Graph::from_parts(vec![0, 2], vec![0], 1),
            Err(PrismError::InvalidFormat(_))
        ));
        assert!(matches!(
            Graph::from_parts(vec![0, 1], vec![1], 1),
            Err(PrismError::InvalidFormat(_))
        ));
        assert!(matches!(
            Graph::from_parts(vec![1, 1], vec![0], 1),
            Err(PrismError::InvalidFormat(_))
        ));
    }

    #[test]
    fn graph_parts_round_trip_through_immutable_accessors() {
        let graph = Graph::from_parts(vec![0, 1, 2], vec![1, 0], 2).unwrap();
        assert_eq!(graph.len(), 2);
        assert!(!graph.is_empty());
        assert_eq!(graph.offsets(), &[0, 1, 2]);
        assert_eq!(graph.neighbor_ids(), &[1, 0]);
    }

    #[test]
    fn test_adj_builder() {
        let mut builder = AdjBuilder::new(3).unwrap();
        builder.add_undirected(0, 1).unwrap();
        builder.add_undirected(0, 2).unwrap();
        assert!(builder.add_edge(0, 3).is_err());
        let g = builder.build().unwrap();
        assert_eq!(g.neighbors(0), Some(&[1, 2][..]));
        assert_eq!(g.neighbors(1), Some(&[0][..]));
        assert_eq!(g.neighbors(2), Some(&[0][..]));
    }

    #[test]
    fn snapshot_matches_build_after_deduplication() {
        let mut builder = AdjBuilder::new(3).unwrap();
        builder.add_edge(0, 2).unwrap();
        builder.add_edge(0, 1).unwrap();
        builder.add_edge(0, 2).unwrap();

        let snapshot = builder.snapshot().unwrap();
        let graph = builder.build().unwrap();

        assert_eq!(snapshot.offsets(), graph.offsets());
        assert_eq!(snapshot.neighbor_ids(), graph.neighbor_ids());
        assert_eq!(snapshot.neighbors(0), Some(&[1, 2][..]));
    }
}
