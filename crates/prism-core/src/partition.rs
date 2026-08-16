use super::error::{PrismError, PrismResult};
use super::point::PointStore;
use std::collections::{HashMap, HashSet};

/// A single leaf cell in the Attribute Partition Tree.
#[derive(Clone, Debug)]
pub struct Cell {
    /// Attribute values that define this cell: `values[j]` = value for attribute j.
    pub(crate) values: Vec<u32>,
    /// Point ids belonging to this cell.
    pub(crate) point_ids: Vec<u32>,
}

impl Cell {
    /// Reassemble a cell for validation by [`PartitionTree::from_parts`].
    pub fn from_parts(values: Vec<u32>, point_ids: Vec<u32>) -> Self {
        Self { values, point_ids }
    }

    /// Full categorical tuple represented by this cell.
    pub fn values(&self) -> &[u32] {
        &self.values
    }

    /// Internal point identifiers contained in this cell.
    pub fn point_ids(&self) -> &[u32] {
        &self.point_ids
    }
}

/// Attribute-tuple partition catalog.
///
/// Despite the historical type name, this is a flat catalog of populated
/// full-tuple cells plus per-attribute posting lists, not a recursive tree.
/// Every point in a cell has the same scalar categorical value for every
/// attribute.
pub struct PartitionTree {
    /// All leaf cells.
    pub(crate) cells: Vec<Cell>,
    /// Attribute split order (permutation of [0..k]).
    pub(crate) split_order: Vec<usize>,
    /// Number of attribute dimensions.
    pub(crate) k: usize,
    /// Attribute value -> sorted compatible cell indices.
    cell_postings: Vec<HashMap<u32, Vec<usize>>>,
}

impl PartitionTree {
    /// Build the partition tree from a PointStore.
    /// Split order: attributes with the most distinct values first (a
    /// cardinality heuristic).
    pub fn build(store: &PointStore) -> Self {
        let k = store.k();
        let n = store.len;

        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by_key(|&b| std::cmp::Reverse(store.cardinality_unchecked(b)));

        let mut groups: HashMap<Vec<u32>, Vec<u32>> = HashMap::new();
        for i in 0..n {
            let key: Vec<u32> = (0..k).map(|j| store.attr_unchecked(i as u32, j)).collect();
            groups.entry(key).or_default().push(i as u32);
        }

        let mut cells: Vec<Cell> = groups
            .into_iter()
            .map(|(values, point_ids)| Cell { values, point_ids })
            .collect();
        // HashMap iteration order is random per process, so sort for stable cell
        // ordering. Randomized graph phases use the build seed instead.
        cells.sort_unstable_by(|a, b| a.values.cmp(&b.values));

        let cell_postings = build_cell_postings(&cells, k);

        Self {
            cells,
            split_order: order,
            k,
            cell_postings,
        }
    }

    /// Find all cells compatible with a filter.
    /// A cell is compatible if for every constrained attribute j,
    /// the cell's value on j is in the allowed set.
    pub fn filter_cells(&self, constraints: &[(usize, Vec<u32>)]) -> PrismResult<Vec<usize>> {
        if let Some((attribute, _)) = constraints.iter().find(|(j, _)| *j >= self.k) {
            return Err(PrismError::InvalidInput(format!(
                "attribute index {attribute} is out of range for {} attributes",
                self.k
            )));
        }
        Ok(self.filter_cells_unchecked(constraints))
    }

    /// Filter cells after every attribute index has been validated.
    pub(crate) fn filter_cells_unchecked(&self, constraints: &[(usize, Vec<u32>)]) -> Vec<usize> {
        if constraints.is_empty() {
            return (0..self.cells.len()).collect();
        }

        let mut matches_per_constraint: Vec<Vec<usize>> = constraints
            .iter()
            .map(|(j, allowed)| {
                debug_assert!(*j < self.k);
                let mut matching = Vec::new();
                for value in allowed {
                    if let Some(posting) = self.cell_postings[*j].get(value) {
                        matching.extend_from_slice(posting);
                    }
                }
                matching.sort_unstable();
                matching.dedup();
                matching
            })
            .collect();

        matches_per_constraint.sort_unstable_by_key(Vec::len);
        let mut result = matches_per_constraint.remove(0);
        for matching in matches_per_constraint {
            result = intersect_sorted(&result, &matching);
            if result.is_empty() {
                break;
            }
        }
        result
    }

    /// Total number of points across given cell indices.
    pub fn count_points(&self, cell_indices: &[usize]) -> PrismResult<usize> {
        if let Some(&cell) = cell_indices.iter().find(|&&i| i >= self.cells.len()) {
            return Err(PrismError::InvalidInput(format!(
                "cell index {cell} is out of range for {} cells",
                self.cells.len()
            )));
        }
        self.count_points_unchecked(cell_indices)
            .ok_or_else(|| PrismError::Overflow("selected point count overflowed usize".into()))
    }

    /// Count points after every cell index has been validated.
    pub(crate) fn count_points_unchecked(&self, cell_indices: &[usize]) -> Option<usize> {
        cell_indices.iter().try_fold(0usize, |total, &i| {
            total.checked_add(self.cells[i].point_ids.len())
        })
    }

    /// Get all point ids in the given cell indices.
    pub fn collect_points(&self, cell_indices: &[usize]) -> PrismResult<Vec<u32>> {
        let point_count = self.count_points(cell_indices)?;
        let mut pts = Vec::new();
        pts.try_reserve_exact(point_count).map_err(|error| {
            PrismError::Overflow(format!(
                "cannot allocate {point_count} selected point identifiers: {error}"
            ))
        })?;
        self.collect_points_unchecked(cell_indices, &mut pts);
        Ok(pts)
    }

    /// Collect points after every cell index and aggregate size has been
    /// validated by the caller.
    pub(crate) fn collect_points_unchecked(&self, cell_indices: &[usize], points: &mut Vec<u32>) {
        for &i in cell_indices {
            points.extend_from_slice(&self.cells[i].point_ids);
        }
    }

    /// Find which cell a point belongs to. Returns cell index.
    pub fn cell_of(&self, store: &PointStore, point_id: u32) -> PrismResult<Option<usize>> {
        if store.k() != self.k {
            return Err(PrismError::InvalidInput(format!(
                "point-store attribute count {} must equal partition schema {}",
                store.k(),
                self.k
            )));
        }
        if point_id as usize >= store.len() {
            return Ok(None);
        }
        Ok(self.cell_of_unchecked(store, point_id))
    }

    /// Resolve a cell after the store schema and point identifier have been
    /// validated.
    pub(crate) fn cell_of_unchecked(&self, store: &PointStore, point_id: u32) -> Option<usize> {
        debug_assert_eq!(store.k(), self.k);
        debug_assert!((point_id as usize) < store.len());
        let key: Vec<u32> = (0..self.k)
            .map(|j| store.attr_unchecked(point_id, j))
            .collect();
        self.cells.iter().position(|c| c.values == key)
    }

    /// Reassemble one persisted, reordered partition component after validating
    /// every schema and point-membership invariant used by filtered search.
    ///
    /// Persisted cells must cover internal point IDs `0..point_count` exactly
    /// once, in contiguous cell ranges. PRISM's local graph search relies on
    /// that layout when translating a global point ID to a cell-local bitset.
    /// Pass this component with the remaining validated parts to
    /// [`crate::PrismIndex::from_parts`] to reassemble a complete index.
    pub fn from_parts(
        cells: Vec<Cell>,
        split_order: Vec<usize>,
        k: usize,
        point_count: usize,
    ) -> PrismResult<Self> {
        if point_count > u32::MAX as usize {
            return Err(PrismError::Overflow(
                "partition point count exceeds the u32 identifier space".into(),
            ));
        }
        if split_order.len() != k {
            return Err(PrismError::InvalidFormat(format!(
                "partition split-order length {} must equal attribute count {k}",
                split_order.len()
            )));
        }
        let mut seen_order = vec![false; k];
        for &attribute in &split_order {
            if attribute >= k || seen_order[attribute] {
                return Err(PrismError::InvalidFormat(
                    "partition split order must be a permutation of 0..k".into(),
                ));
            }
            seen_order[attribute] = true;
        }

        if point_count == 0 && !cells.is_empty() {
            return Err(PrismError::InvalidFormat(
                "an empty point set cannot contain partition cells".into(),
            ));
        }

        let mut tuples = HashSet::with_capacity(cells.len());
        let mut expected_point = 0usize;
        for (cell_index, cell) in cells.iter().enumerate() {
            if cell.values.len() != k {
                return Err(PrismError::InvalidFormat(format!(
                    "partition cell {cell_index} has {} values; expected {k}",
                    cell.values.len()
                )));
            }
            if !tuples.insert(cell.values.clone()) {
                return Err(PrismError::InvalidFormat(format!(
                    "partition cell {cell_index} duplicates a full attribute tuple"
                )));
            }
            if cell.point_ids.is_empty() {
                return Err(PrismError::InvalidFormat(format!(
                    "partition cell {cell_index} is empty"
                )));
            }
            for &point_id in &cell.point_ids {
                if expected_point >= point_count {
                    return Err(PrismError::InvalidFormat(format!(
                        "partition cells contain more than the declared {point_count} points"
                    )));
                }
                let expected_id = u32::try_from(expected_point).map_err(|_| {
                    PrismError::Overflow(
                        "partition point identifier exceeds the u32 identifier space".into(),
                    )
                })?;
                if point_id != expected_id {
                    return Err(PrismError::InvalidFormat(format!(
                        "partition cells must cover contiguous point IDs; expected {expected_id}, got {point_id}"
                    )));
                }
                expected_point += 1;
            }
        }
        if expected_point != point_count {
            return Err(PrismError::InvalidFormat(format!(
                "partition cells cover {expected_point} points; expected {point_count}"
            )));
        }

        let cell_postings = build_cell_postings(&cells, k);
        Ok(Self {
            cells,
            split_order,
            k,
            cell_postings,
        })
    }

    /// Immutable cells in catalog order.
    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    /// Attribute split order used when the catalog was built.
    pub fn split_order(&self) -> &[usize] {
        &self.split_order
    }

    /// Number of scalar attribute dimensions.
    pub fn num_attributes(&self) -> usize {
        self.k
    }
}

fn intersect_sorted(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut result = Vec::with_capacity(a.len().min(b.len()));
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

fn build_cell_postings(cells: &[Cell], k: usize) -> Vec<HashMap<u32, Vec<usize>>> {
    let mut postings: Vec<HashMap<u32, Vec<usize>>> = vec![HashMap::new(); k];
    for (cell_idx, cell) in cells.iter().enumerate() {
        for (j, &value) in cell.values.iter().enumerate() {
            postings[j].entry(value).or_default().push(cell_idx);
        }
    }
    postings
}

#[cfg(test)]
mod tests {
    use super::super::point::PointStore;
    use super::*;

    #[test]
    fn test_partition_tree() {
        // 6 points, 2 attributes: color(3 values), size(2 values)
        let vectors = vec![0.0f32; 6 * 2];
        let attrs = vec![
            vec![0, 0, 1, 1, 2, 2], // color
            vec![0, 1, 0, 1, 0, 1], // size
        ];
        let store = PointStore::from_parts(vectors, 2, attrs).unwrap();
        let tree = PartitionTree::build(&store);
        assert_eq!(tree.cells.len(), 6); // 3*2 = 6 distinct combos

        let cells = tree.filter_cells(&[(0, vec![0])]).unwrap();
        let pts = tree.collect_points(&cells).unwrap();
        assert_eq!(pts.len(), 2);

        let cells = tree.filter_cells(&[(0, vec![1]), (1, vec![0])]).unwrap();
        let pts = tree.collect_points(&cells).unwrap();
        assert_eq!(pts.len(), 1);

        // IN within one attribute is a union; constraints across attributes
        // are intersected through the cell posting lists.
        let cells = tree.filter_cells(&[(0, vec![0, 2]), (1, vec![1])]).unwrap();
        let pts = tree.collect_points(&cells).unwrap();
        assert_eq!(pts.len(), 2);
    }

    #[test]
    fn persisted_tree_rebuilds_postings_after_full_validation() {
        let cells = vec![
            Cell::from_parts(vec![0, 1], vec![0, 1]),
            Cell::from_parts(vec![2, 1], vec![2]),
        ];
        let tree = PartitionTree::from_parts(cells, vec![0, 1], 2, 3).unwrap();

        assert_eq!(tree.num_attributes(), 2);
        assert_eq!(tree.split_order(), &[0, 1]);
        assert_eq!(tree.cells()[0].values(), &[0, 1]);
        assert_eq!(tree.cells()[0].point_ids(), &[0, 1]);
        assert_eq!(tree.filter_cells(&[(0, vec![2])]).unwrap(), vec![1]);
    }

    #[test]
    fn persisted_tree_rejects_noncontiguous_or_duplicate_membership() {
        let noncontiguous = vec![
            Cell::from_parts(vec![0], vec![0]),
            Cell::from_parts(vec![1], vec![2]),
        ];
        assert!(matches!(
            PartitionTree::from_parts(noncontiguous, vec![0], 1, 2),
            Err(PrismError::InvalidFormat(_))
        ));

        let duplicate_tuple = vec![
            Cell::from_parts(vec![0], vec![0]),
            Cell::from_parts(vec![0], vec![1]),
        ];
        assert!(matches!(
            PartitionTree::from_parts(duplicate_tuple, vec![0], 1, 2),
            Err(PrismError::InvalidFormat(_))
        ));
    }

    #[test]
    fn public_partition_queries_reject_invalid_indices_and_schema() {
        let store =
            PointStore::from_parts(vec![0.0, 1.0], 1, vec![vec![0, 1], vec![2, 3]]).unwrap();
        let tree = PartitionTree::build(&store);

        assert!(matches!(
            tree.filter_cells(&[(2, vec![0])]),
            Err(PrismError::InvalidInput(_))
        ));
        assert!(matches!(
            tree.count_points(&[tree.cells().len()]),
            Err(PrismError::InvalidInput(_))
        ));
        assert!(matches!(
            tree.collect_points(&[tree.cells().len()]),
            Err(PrismError::InvalidInput(_))
        ));

        let wrong_schema = PointStore::from_parts(vec![0.0, 1.0], 1, vec![vec![0, 1]]).unwrap();
        assert!(matches!(
            tree.cell_of(&wrong_schema, 0),
            Err(PrismError::InvalidInput(_))
        ));
        assert_eq!(tree.cell_of(&store, u32::MAX).unwrap(), None);
    }
}
