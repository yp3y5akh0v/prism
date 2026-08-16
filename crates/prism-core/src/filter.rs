use super::error::{PrismError, PrismResult};
use super::point::PointStore;

/// A conjunctive filter: `(attr_index, allowed_values)` pairs.
/// Point passes iff for every constrained attribute, its value is in the allowed set.
#[derive(Clone, Debug)]
pub struct Filter {
    /// Each entry: (attribute_index, set of allowed values)
    constraints: Vec<(usize, Vec<u32>)>,
}

impl Filter {
    /// Create from a list of (attr_index, allowed_values).
    pub fn new(mut constraints: Vec<(usize, Vec<u32>)>) -> Self {
        // Canonicalize for deterministic planning and logarithmic membership.
        // Repeated constraints on one attribute keep AND semantics by intersecting.
        for (_, allowed) in &mut constraints {
            allowed.sort_unstable();
            allowed.dedup();
        }
        constraints.sort_unstable_by_key(|(j, _)| *j);

        let mut normalized: Vec<(usize, Vec<u32>)> = Vec::with_capacity(constraints.len());
        for (j, allowed) in constraints {
            if let Some((last_j, last_allowed)) = normalized.last_mut() {
                if *last_j == j {
                    *last_allowed = sorted_intersection(last_allowed, &allowed);
                    continue;
                }
            }
            normalized.push((j, allowed));
        }
        Self {
            constraints: normalized,
        }
    }

    /// Unconstrained filter (matches everything).
    pub fn none() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    /// Single equality filter: attribute `j` must equal `val`.
    pub fn eq(j: usize, val: u32) -> Self {
        Self {
            constraints: vec![(j, vec![val])],
        }
    }

    /// Number of constrained attributes (strength).
    pub fn strength(&self) -> usize {
        self.constraints.len()
    }

    /// Check if point `id` passes this filter.
    #[inline]
    pub fn matches(&self, store: &PointStore, id: u32) -> bool {
        if id as usize >= store.len() {
            return false;
        }
        for &(j, ref allowed) in &self.constraints {
            let Some(val) = store.attr(id, j) else {
                return false;
            };
            if allowed.binary_search(&val).is_err() {
                return false;
            }
        }
        true
    }

    /// Get the constraints.
    pub fn constraints(&self) -> &[(usize, Vec<u32>)] {
        &self.constraints
    }

    /// Validate this filter against an attribute schema.
    pub fn validate(&self, num_attributes: usize) -> PrismResult<()> {
        if let Some((j, _)) = self.constraints.iter().find(|(j, _)| *j >= num_attributes) {
            return Err(PrismError::InvalidInput(format!(
                "attribute index {j} is out of range for {num_attributes} attributes"
            )));
        }
        Ok(())
    }
}

fn sorted_intersection(a: &[u32], b: &[u32]) -> Vec<u32> {
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

#[cfg(test)]
mod tests {
    use super::super::point::PointStore;
    use super::*;

    #[test]
    fn test_filter_matches() {
        let store =
            PointStore::from_parts(vec![0.0; 6], 2, vec![vec![0, 1, 2], vec![10, 20, 30]]).unwrap();
        let f = Filter::new(vec![(0, vec![0, 1]), (1, vec![20])]);
        assert!(!f.matches(&store, 0)); // attr0=0 ok, attr1=10 fail
        assert!(f.matches(&store, 1)); // attr0=1 ok, attr1=20 ok
        assert!(!f.matches(&store, 2)); // attr0=2 fail
    }

    #[test]
    fn test_filter_none() {
        let store = PointStore::from_parts(vec![0.0; 3], 3, vec![vec![5]]).unwrap();
        let f = Filter::none();
        assert!(f.matches(&store, 0));
        assert!(!f.matches(&store, 1));
    }

    #[test]
    fn test_filter_strength() {
        let f = Filter::new(vec![(0, vec![1]), (2, vec![3, 4])]);
        assert_eq!(f.strength(), 2);
    }

    #[test]
    fn filter_canonicalizes_values_and_intersects_duplicate_attributes() {
        let f = Filter::new(vec![(0, vec![3, 2, 2, 1]), (0, vec![4, 2, 3])]);
        assert_eq!(f.constraints(), &[(0, vec![2, 3])]);
        assert_eq!(f.strength(), 1);
    }
}
