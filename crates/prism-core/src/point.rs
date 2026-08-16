use super::error::{PrismError, PrismResult};

/// Largest absolute coordinate accepted by the public f32-distance contract.
///
/// If both operands obey this bound, a full-dimensional squared L2 distance
/// is at most `f32::MAX / 2` and an absolute dot product is at most
/// `f32::MAX / 8`. Keeping a margin matters because the public ranking APIs
/// expose f32 distances even when their internal accumulation uses f64.
pub(crate) fn f32_distance_component_limit(dim: usize) -> f64 {
    debug_assert!(dim > 0);
    ((f32::MAX as f64) / (8.0 * dim as f64)).sqrt()
}

/// Validate coordinates before they can reach any f32-valued ranking path.
pub(crate) fn validate_f32_distance_domain(
    values: &[f32],
    dim: usize,
    context: &str,
) -> PrismResult<()> {
    if dim == 0 {
        return Err(PrismError::InvalidInput(format!(
            "{context} dimension must be greater than zero"
        )));
    }
    let limit = f32_distance_component_limit(dim);
    if let Some((index, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || f64::from(value.abs()) > limit)
    {
        return Err(PrismError::InvalidInput(format!(
            "{context} value at flat index {index} must be finite with absolute magnitude at most {limit:e} for dimension {dim}, got {value}"
        )));
    }
    Ok(())
}

/// Flat f32 storage for vectors + per-point attribute metadata.
///
/// Vectors are stored in a contiguous `Vec<f32>` with stride = `dim`.
/// Attributes are stored as `k` arrays of `u32`, one per attribute dimension.
pub struct PointStore {
    /// Contiguous vector data: point i is at `vectors[i*dim..(i+1)*dim]`
    pub(crate) vectors: Vec<f32>,
    /// Number of dimensions per vector
    pub(crate) dim: usize,
    /// Number of points
    pub(crate) len: usize,
    /// Attribute values: `attrs[j][i]` = value of attribute j for point i
    pub(crate) attrs: Vec<Vec<u32>>,
}

impl PointStore {
    pub fn new(dim: usize, k: usize) -> PrismResult<Self> {
        if dim == 0 {
            return Err(PrismError::InvalidInput(
                "vector dimension must be greater than zero".into(),
            ));
        }
        let mut attrs = Vec::new();
        attrs.try_reserve_exact(k).map_err(|error| {
            PrismError::Overflow(format!("cannot allocate {k} attribute columns: {error}"))
        })?;
        attrs.resize_with(k, Vec::new);
        Ok(Self {
            vectors: Vec::new(),
            dim,
            len: 0,
            attrs,
        })
    }

    /// Validate borrowed row-major vectors and attribute columns without
    /// taking ownership or allocating a second copy.
    pub(crate) fn validate_parts(
        vectors: &[f32],
        dim: usize,
        attrs: &[Vec<u32>],
    ) -> PrismResult<usize> {
        if dim == 0 {
            return Err(PrismError::InvalidInput(
                "vector dimension must be greater than zero".into(),
            ));
        }
        if vectors.len() % dim != 0 {
            return Err(PrismError::InvalidInput(format!(
                "vector data length {} must be divisible by dimension {dim}",
                vectors.len()
            )));
        }
        let len = vectors.len() / dim;
        if len > u32::MAX as usize {
            return Err(PrismError::Overflow(
                "point count exceeds the u32 identifier space".into(),
            ));
        }
        validate_f32_distance_domain(vectors, dim, "vector")?;
        for (j, attribute) in attrs.iter().enumerate() {
            if attribute.len() != len {
                return Err(PrismError::InvalidInput(format!(
                    "attribute column {j} length {} must equal point count {len}",
                    attribute.len()
                )));
            }
        }
        Ok(len)
    }

    /// Build from pre-allocated vectors and attributes.
    pub fn from_parts(vectors: Vec<f32>, dim: usize, attrs: Vec<Vec<u32>>) -> PrismResult<Self> {
        let len = Self::validate_parts(&vectors, dim, &attrs)?;
        Ok(Self {
            vectors,
            dim,
            len,
            attrs,
        })
    }

    /// Number of attribute dimensions.
    pub fn k(&self) -> usize {
        self.attrs.len()
    }

    /// Contiguous vector storage in row-major point order.
    pub fn vectors(&self) -> &[f32] {
        &self.vectors
    }

    /// Number of dimensions per vector.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of stored points.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the store contains no points.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Immutable attribute columns in schema order.
    pub fn attributes(&self) -> &[Vec<u32>] {
        &self.attrs
    }

    /// Immutable values for one attribute dimension.
    pub fn attribute(&self, j: usize) -> Option<&[u32]> {
        self.attrs.get(j).map(Vec::as_slice)
    }

    /// Get the vector slice for point `id`.
    #[inline]
    pub fn vector(&self, id: u32) -> Option<&[f32]> {
        if id as usize >= self.len {
            return None;
        }
        Some(self.vector_unchecked(id))
    }

    /// Get a vector by a previously validated internal point ID.
    #[inline]
    pub(crate) fn vector_unchecked(&self, id: u32) -> &[f32] {
        let start = id as usize * self.dim;
        &self.vectors[start..start + self.dim]
    }

    /// Get attribute value for point `id` on dimension `j`.
    #[inline]
    pub fn attr(&self, id: u32, j: usize) -> Option<u32> {
        self.attrs
            .get(j)
            .and_then(|column| column.get(id as usize))
            .copied()
    }

    /// Get an attribute by previously validated point and schema IDs.
    #[inline]
    pub(crate) fn attr_unchecked(&self, id: u32, j: usize) -> u32 {
        self.attrs[j][id as usize]
    }

    /// Append a single point. Returns its id.
    pub fn push(&mut self, vector: &[f32], attr_values: &[u32]) -> PrismResult<u32> {
        if vector.len() != self.dim {
            return Err(PrismError::InvalidInput(format!(
                "vector length {} must equal store dimension {}",
                vector.len(),
                self.dim
            )));
        }
        if attr_values.len() != self.attrs.len() {
            return Err(PrismError::InvalidInput(format!(
                "attribute count {} must equal store schema {}",
                attr_values.len(),
                self.attrs.len()
            )));
        }
        validate_f32_distance_domain(vector, self.dim, "vector")?;
        if self.len >= u32::MAX as usize {
            return Err(PrismError::Overflow(
                "point count exceeds the u32 identifier space".into(),
            ));
        }
        self.vectors.try_reserve_exact(self.dim).map_err(|error| {
            PrismError::Overflow(format!("cannot grow vector storage: {error}"))
        })?;
        for column in &mut self.attrs {
            column.try_reserve_exact(1).map_err(|error| {
                PrismError::Overflow(format!("cannot grow attribute storage: {error}"))
            })?;
        }
        let id = self.len as u32;
        self.vectors.extend_from_slice(vector);
        for (j, &val) in attr_values.iter().enumerate() {
            self.attrs[j].push(val);
        }
        self.len += 1;
        Ok(id)
    }

    /// Number of distinct values for attribute dimension `j`.
    pub fn cardinality(&self, j: usize) -> Option<usize> {
        self.attrs.get(j).map(|_| self.cardinality_unchecked(j))
    }

    /// Number of distinct values for a previously validated attribute index.
    pub(crate) fn cardinality_unchecked(&self, j: usize) -> usize {
        let mut seen = std::collections::HashSet::new();
        for &v in &self.attrs[j] {
            seen.insert(v);
        }
        seen.len()
    }
}

impl Drop for PointStore {
    fn drop(&mut self) {
        // Clear this owned vector allocation before releasing it. This does not
        // erase caller copies, allocator/OS copies, persistence, or backups.
        use zeroize::Zeroize;
        self.vectors.zeroize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_store_basic() {
        let mut store = PointStore::new(3, 2).unwrap();
        let id0 = store.push(&[1.0, 2.0, 3.0], &[0, 1]).unwrap();
        let id1 = store.push(&[4.0, 5.0, 6.0], &[1, 0]).unwrap();
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(store.len, 2);
        assert_eq!(store.len(), 2);
        assert_eq!(store.dim(), 3);
        assert!(!store.is_empty());
        assert_eq!(store.vectors(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(store.attribute(0), Some(&[0, 1][..]));
        assert_eq!(store.vector(0), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(store.vector(1), Some(&[4.0, 5.0, 6.0][..]));
        assert_eq!(store.vector(2), None);
        assert_eq!(store.attr(0, 0), Some(0));
        assert_eq!(store.attr(0, 1), Some(1));
        assert_eq!(store.attr(1, 0), Some(1));
        assert_eq!(store.attr(1, 1), Some(0));
        assert_eq!(store.attr(2, 0), None);
        assert_eq!(store.attr(0, 2), None);
    }

    #[test]
    fn test_from_parts() {
        let vectors = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let attrs = vec![vec![0, 1], vec![1, 0]];
        let store = PointStore::from_parts(vectors, 3, attrs).unwrap();
        assert_eq!(store.len, 2);
        assert_eq!(store.k(), 2);
        assert_eq!(store.cardinality(0), Some(2));
        assert_eq!(store.cardinality(2), None);
    }

    #[test]
    fn malformed_store_inputs_return_errors() {
        assert!(matches!(
            PointStore::new(0, 1),
            Err(PrismError::InvalidInput(_))
        ));
        assert!(matches!(
            PointStore::from_parts(vec![1.0], 2, vec![Vec::new()]),
            Err(PrismError::InvalidInput(_))
        ));
        assert!(matches!(
            PointStore::from_parts(vec![f32::NAN], 1, vec![vec![0]]),
            Err(PrismError::InvalidInput(_))
        ));

        let mut store = PointStore::new(2, 1).unwrap();
        assert!(store.push(&[1.0], &[0]).is_err());
        assert!(store.push(&[1.0, 2.0], &[]).is_err());
        assert!(store.push(&[1.0, f32::INFINITY], &[0]).is_err());
        assert!(store.is_empty());
    }

    #[test]
    fn distance_domain_rejects_extreme_finite_coordinates() {
        assert!(matches!(
            PointStore::from_parts(vec![f32::MAX], 1, vec![vec![0]]),
            Err(PrismError::InvalidInput(_))
        ));

        let mut store = PointStore::new(2, 1).unwrap();
        assert!(matches!(
            store.push(&[f32::MAX, -f32::MAX], &[0]),
            Err(PrismError::InvalidInput(_))
        ));
        assert!(store.is_empty());
    }

    #[test]
    fn distance_domain_accepts_values_at_a_safe_margin() {
        let limit = f32_distance_component_limit(2) as f32;
        let store = PointStore::from_parts(vec![limit, -limit, -limit, limit], 2, vec![vec![0, 0]])
            .unwrap();
        let a = store.vector(0).unwrap();
        let b = store.vector(1).unwrap();
        let l2: f64 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| {
                let delta = f64::from(x) - f64::from(y);
                delta * delta
            })
            .sum();
        assert!(l2.is_finite() && l2 <= f32::MAX as f64);
    }
}
