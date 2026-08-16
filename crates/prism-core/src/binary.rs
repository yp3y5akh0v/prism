use super::error::{PrismError, PrismResult};
use super::point::{validate_f32_distance_domain, PointStore};
use rayon::prelude::*;

/// Binary code store for Hamming-distance prefiltering.
///
/// Uses a seeded blockwise randomized Walsh-Hadamard transform before sign
/// extraction. This is a structured
/// [SimHash-like](https://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/p380-charikar.pdf)
/// sketch, not classic independent-random-hyperplane SimHash; see also the
/// [randomized-Hadamard transform](https://doi.org/10.1137/060673096).
pub struct BinaryStore {
    codes: Vec<u64>,
    code_words: usize,
    signs: Vec<f32>,
    block_size: usize,
}

impl Drop for BinaryStore {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.codes.zeroize();
        self.signs.zeroize();
    }
}

impl BinaryStore {
    /// Reassemble one persisted binary-code component after validation. Signs
    /// are accepted explicitly rather than re-derived from the seed, so a
    /// future seed change cannot silently desynchronize codes from queries.
    /// Pass this component with the remaining validated parts to
    /// [`crate::PrismIndex::from_parts`] to reassemble a complete index.
    pub fn from_parts(
        codes: Vec<u64>,
        code_words: usize,
        signs: Vec<f32>,
        block_size: usize,
    ) -> PrismResult<Self> {
        let dim = signs.len();
        if dim == 0 {
            return Err(PrismError::InvalidFormat(
                "binary-code dimension must be greater than zero".into(),
            ));
        }
        let expected_words = dim.div_ceil(64);
        if code_words != expected_words {
            return Err(PrismError::InvalidFormat(format!(
                "binary code_words {code_words} must equal ceil({dim}/64) = {expected_words}"
            )));
        }
        if codes.len() % code_words != 0 {
            return Err(PrismError::InvalidFormat(format!(
                "binary code length {} must be divisible by code_words {code_words}",
                codes.len()
            )));
        }
        let point_count = codes.len() / code_words;
        if point_count > u32::MAX as usize {
            return Err(PrismError::Overflow(
                "binary-code point count exceeds the u32 identifier space".into(),
            ));
        }
        if block_size == 0 || !block_size.is_power_of_two() || dim % block_size != 0 {
            return Err(PrismError::InvalidFormat(format!(
                "binary block size {block_size} must be a nonzero power of two dividing dimension {dim}"
            )));
        }
        if let Some((dimension, sign)) = signs
            .iter()
            .enumerate()
            .find(|(_, sign)| **sign != -1.0 && **sign != 1.0)
        {
            return Err(PrismError::InvalidFormat(format!(
                "binary sign for dimension {dimension} must be -1 or 1, got {sign}"
            )));
        }
        let used_bits = dim % 64;
        if used_bits != 0 {
            let used_mask = (1u64 << used_bits) - 1;
            if codes
                .chunks_exact(code_words)
                .any(|code| code[code_words - 1] & !used_mask != 0)
            {
                return Err(PrismError::InvalidFormat(
                    "binary code sets bits beyond the vector dimension".into(),
                ));
            }
        }

        Ok(Self {
            codes,
            code_words,
            signs,
            block_size,
        })
    }

    pub fn codes(&self) -> &[u64] {
        &self.codes
    }

    pub fn signs(&self) -> &[f32] {
        &self.signs
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Vector dimensionality encoded by each binary code.
    pub fn dim(&self) -> usize {
        self.signs.len()
    }

    /// Number of stored point codes.
    pub fn len(&self) -> usize {
        self.codes.len() / self.code_words
    }

    /// Whether the store contains no point codes.
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Build seeded randomized-Hadamard sign sketches in power-of-two blocks.
    pub fn build(store: &PointStore) -> PrismResult<Self> {
        Self::build_from_f32(store.vectors(), store.dim())
    }

    /// Build codes for a validated row-major f32 buffer. This is used by IVF
    /// after it has reordered vectors into internal identifier order.
    pub(crate) fn build_from_f32(vectors: &[f32], dim: usize) -> PrismResult<Self> {
        let n = validate_raw_shape(vectors.len(), dim, "binary-code vector")?;
        validate_f32_distance_domain(vectors, dim, "binary-code vector")?;
        Self::build_rows(vectors, n, dim)
    }

    /// Build codes for a validated row-major byte buffer in IVF internal order.
    pub(crate) fn build_from_u8(vectors: &[u8], dim: usize) -> PrismResult<Self> {
        let n = validate_raw_shape(vectors.len(), dim, "binary-code byte vector")?;
        Self::build_rows(vectors, n, dim)
    }

    fn build_rows<T>(vectors: &[T], n: usize, dim: usize) -> PrismResult<Self>
    where
        T: Copy + Into<f64> + Sync,
    {
        let code_words = dim.div_ceil(64);
        let block_size = largest_pow2_factor(dim);
        let signs = seeded_signs(dim)?;

        let code_count = n
            .checked_mul(code_words)
            .ok_or_else(|| PrismError::Overflow("binary-code shape overflows usize".into()))?;
        let mut codes = Vec::new();
        codes.try_reserve_exact(code_count).map_err(|error| {
            PrismError::Overflow(format!(
                "cannot allocate {code_count} binary-code words: {error}"
            ))
        })?;
        codes.resize(code_count, 0u64);
        codes
            .par_chunks_mut(code_words)
            .enumerate()
            .try_for_each(|(i, chunk)| {
                let start = i * dim;
                encode_vector(&vectors[start..start + dim], &signs, block_size, chunk)
            })?;

        Ok(Self {
            codes,
            code_words,
            signs,
            block_size,
        })
    }

    /// A store with signs but no codes, for configs that never consult the
    /// binary pre-filter (`binary_rerank == 0`). `encode_query` stays valid;
    /// `code()` must not be reached (every caller is gated on the rerank
    /// factor), so the per-point encoding pass and its memory are skipped.
    pub fn empty(dim: usize) -> PrismResult<Self> {
        if dim == 0 {
            return Err(PrismError::InvalidInput(
                "binary-code dimension must be greater than zero".into(),
            ));
        }
        Ok(Self {
            codes: Vec::new(),
            code_words: dim.div_ceil(64),
            signs: seeded_signs(dim)?,
            block_size: largest_pow2_factor(dim),
        })
    }

    /// Get the binary code (packed u64 words) for point id.
    #[inline]
    pub fn code(&self, id: u32) -> Option<&[u64]> {
        ((id as usize) < self.len()).then(|| self.code_unchecked(id))
    }

    /// Get a code by a previously validated internal point ID.
    #[inline]
    pub(crate) fn code_unchecked(&self, id: u32) -> &[u64] {
        let start = id as usize * self.code_words;
        &self.codes[start..start + self.code_words]
    }

    /// Number of u64 words per binary code.
    #[inline]
    pub fn code_words(&self) -> usize {
        self.code_words
    }

    /// Encode a query vector to binary code using the same HD rotation.
    pub fn encode_query(&self, query: &[f32]) -> PrismResult<Vec<u64>> {
        if query.len() != self.signs.len() {
            return Err(PrismError::InvalidInput(format!(
                "query length {} must equal binary-code dimension {}",
                query.len(),
                self.signs.len()
            )));
        }
        validate_f32_distance_domain(query, self.dim(), "query")?;
        let mut code = allocate_query_code(self.code_words)?;
        encode_vector(query, &self.signs, self.block_size, &mut code)?;
        Ok(code)
    }

    /// Encode a byte query for an IVF index that owns byte vectors.
    pub(crate) fn encode_query_u8(&self, query: &[u8]) -> PrismResult<Vec<u64>> {
        if query.len() != self.dim() {
            return Err(PrismError::InvalidInput(format!(
                "query length {} must equal binary-code dimension {}",
                query.len(),
                self.dim()
            )));
        }
        let mut code = allocate_query_code(self.code_words)?;
        encode_vector(query, &self.signs, self.block_size, &mut code)?;
        Ok(code)
    }
}

/// Seed-fixed random sign flips shared by build and query encoding.
fn seeded_signs(dim: usize) -> PrismResult<Vec<f32>> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x505249534D);
    let mut signs = Vec::new();
    signs.try_reserve_exact(dim).map_err(|error| {
        PrismError::Overflow(format!("cannot allocate {dim} binary signs: {error}"))
    })?;
    signs.extend((0..dim).map(|_| if rng.gen_bool(0.5) { 1.0 } else { -1.0 }));
    Ok(signs)
}

/// Apply HD rotation (sign flip + WHT) and extract signs into packed u64 code.
fn encode_vector<T>(
    vector: &[T],
    signs: &[f32],
    block_size: usize,
    out: &mut [u64],
) -> PrismResult<()>
where
    T: Copy + Into<f64>,
{
    let dim = vector.len();
    debug_assert_eq!(signs.len(), dim);
    debug_assert_eq!(out.len(), dim.div_ceil(64));
    debug_assert!(block_size.is_power_of_two() && dim % block_size == 0);
    let mut buf = Vec::new();
    buf.try_reserve_exact(dim).map_err(|error| {
        PrismError::Overflow(format!(
            "cannot allocate {dim}-dimension Walsh-Hadamard workspace: {error}"
        ))
    })?;
    buf.extend(
        vector
            .iter()
            .enumerate()
            .map(|(d, &value)| value.into() * f64::from(signs[d])),
    );
    for start in (0..dim).step_by(block_size) {
        walsh_hadamard(&mut buf[start..start + block_size]);
    }
    for d in 0..dim {
        if buf[d] >= 0.0 {
            out[d / 64] |= 1u64 << (d % 64);
        }
    }
    Ok(())
}

/// In-place Walsh-Hadamard transform on a slice of length 2^k.
/// Not normalized (irrelevant for sign extraction).
fn walsh_hadamard(data: &mut [f64]) {
    let n = data.len();
    debug_assert!(n.is_power_of_two());
    if n <= 1 {
        return;
    }
    let mut half = 1;
    while half < n {
        let step = half * 2;
        for i in (0..n).step_by(step) {
            for j in 0..half {
                let a = data[i + j];
                let b = data[i + j + half];
                data[i + j] = a + b;
                data[i + j + half] = a - b;
            }
        }
        half = step;
    }
}

/// Largest power-of-2 factor of n (i.e., 2^(trailing zeros of n)).
fn largest_pow2_factor(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    1 << n.trailing_zeros()
}

fn validate_raw_shape(len: usize, dim: usize, context: &str) -> PrismResult<usize> {
    if dim == 0 {
        return Err(PrismError::InvalidInput(format!(
            "{context} dimension must be greater than zero"
        )));
    }
    if len % dim != 0 {
        return Err(PrismError::InvalidInput(format!(
            "{context} length {len} must be divisible by dimension {dim}"
        )));
    }
    let n = len / dim;
    if n > u32::MAX as usize {
        return Err(PrismError::Overflow(format!(
            "{context} point count exceeds the u32 identifier space"
        )));
    }
    Ok(n)
}

fn allocate_query_code(code_words: usize) -> PrismResult<Vec<u64>> {
    let mut code = Vec::new();
    code.try_reserve_exact(code_words).map_err(|error| {
        PrismError::Overflow(format!(
            "cannot allocate {code_words} binary query words: {error}"
        ))
    })?;
    code.resize(code_words, 0u64);
    Ok(code)
}

#[cfg(test)]
mod tests {
    use super::super::point::PointStore;
    use super::*;

    #[test]
    fn test_walsh_hadamard_identity() {
        let mut data = vec![1.0, 0.0, 0.0, 0.0];
        walsh_hadamard(&mut data);
        assert_eq!(data, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_walsh_hadamard_butterfly() {
        let mut data = vec![1.0, 1.0];
        walsh_hadamard(&mut data);
        assert_eq!(data, vec![2.0, 0.0]);

        let mut data = vec![1.0, -1.0];
        walsh_hadamard(&mut data);
        assert_eq!(data, vec![0.0, 2.0]);
    }

    #[test]
    fn test_largest_pow2_factor() {
        assert_eq!(largest_pow2_factor(384), 128);
        assert_eq!(largest_pow2_factor(128), 128);
        assert_eq!(largest_pow2_factor(256), 256);
        assert_eq!(largest_pow2_factor(12), 4);
        assert_eq!(largest_pow2_factor(1), 1);
    }

    #[test]
    fn test_binary_query_encoding() {
        let dim = 128;
        let p0: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut vecs = Vec::with_capacity(dim);
        vecs.extend_from_slice(&p0);

        let store = PointStore::from_parts(vecs, dim, vec![vec![0]]).unwrap();
        let binary = BinaryStore::build(&store).unwrap();

        let q = binary.encode_query(&p0).unwrap();
        let c0 = binary.code(0).unwrap();
        assert_eq!(q, c0, "query encoding must match point encoding");
    }

    #[test]
    fn test_hamming_distance_ordering() {
        use super::super::distance;
        let dim = 128;
        let p0: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) / dim as f32).collect();
        let p1: Vec<f32> = p0.iter().map(|&v| v + 0.001).collect();
        let p2: Vec<f32> = p0.iter().map(|&v| -v).collect();

        let mut vecs = Vec::with_capacity(3 * dim);
        vecs.extend_from_slice(&p0);
        vecs.extend_from_slice(&p1);
        vecs.extend_from_slice(&p2);

        let store = PointStore::from_parts(vecs, dim, vec![vec![0, 0, 0]]).unwrap();
        let binary = BinaryStore::build(&store).unwrap();
        let q = binary.encode_query(&p0).unwrap();

        let d0 = distance::hamming(&q, binary.code(0).unwrap());
        let d1 = distance::hamming(&q, binary.code(1).unwrap());
        let d2 = distance::hamming(&q, binary.code(2).unwrap());

        assert_eq!(d0, 0, "same vector must have 0 Hamming distance");
        assert!(
            d1 < d2,
            "close vector (d={d1}) must have smaller Hamming than opposite (d={d2})"
        );
    }

    #[test]
    fn test_binary_code_words() {
        let store = PointStore::from_parts(vec![0.0; 128], 128, vec![vec![0]]).unwrap();
        let binary = BinaryStore::build(&store).unwrap();
        assert_eq!(binary.code_words(), 2);

        let store = PointStore::from_parts(vec![0.0; 384], 384, vec![vec![0]]).unwrap();
        let binary = BinaryStore::build(&store).unwrap();
        assert_eq!(binary.code_words(), 6);
    }

    #[test]
    fn persisted_binary_rejects_invalid_shapes_signs_and_padding() {
        assert!(matches!(
            BinaryStore::from_parts(vec![0], 2, vec![1.0; 65], 1),
            Err(PrismError::InvalidFormat(_))
        ));
        assert!(matches!(
            BinaryStore::from_parts(vec![0], 1, vec![1.0, 0.5], 2),
            Err(PrismError::InvalidFormat(_))
        ));
        assert!(matches!(
            BinaryStore::from_parts(vec![0], 1, vec![1.0; 6], 4),
            Err(PrismError::InvalidFormat(_))
        ));
        assert!(matches!(
            BinaryStore::from_parts(vec![u64::MAX], 1, vec![1.0; 5], 1),
            Err(PrismError::InvalidFormat(_))
        ));
    }

    #[test]
    fn persisted_binary_valid_parts_round_trip() {
        let binary = BinaryStore::from_parts(vec![0b1_1111], 1, vec![1.0; 5], 1).unwrap();
        assert_eq!(binary.dim(), 5);
        assert_eq!(binary.len(), 1);
        assert!(!binary.is_empty());
        assert_eq!(binary.code(0), Some(&[0b1_1111][..]));
        assert_eq!(binary.code(1), None);
    }

    #[test]
    fn binary_rejects_extreme_finite_query_and_raw_vectors() {
        let binary = BinaryStore::empty(1).unwrap();
        assert!(matches!(
            binary.encode_query(&[f32::MAX]),
            Err(PrismError::InvalidInput(_))
        ));
        assert!(matches!(
            BinaryStore::build_from_f32(&[f32::MAX], 1),
            Err(PrismError::InvalidInput(_))
        ));
        assert!(matches!(
            BinaryStore::empty(usize::MAX),
            Err(PrismError::Overflow(_))
        ));
    }

    #[test]
    fn f64_walsh_workspace_handles_safe_large_coordinates() {
        let limit = (super::super::point::f32_distance_component_limit(4) * 0.99) as f32;
        let vectors = [limit, limit, limit, limit];
        let binary = BinaryStore::build_from_f32(&vectors, 4).unwrap();
        let query = binary.encode_query(&vectors).unwrap();
        assert_eq!(query, binary.code(0).unwrap());
    }
}
