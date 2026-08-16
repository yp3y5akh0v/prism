use super::error::{PrismError, PrismResult};
use super::point::{f32_distance_component_limit, validate_f32_distance_domain, PointStore};
use zeroize::Zeroize;

/// Human-readable diagnostic for the SQ8 candidate-distance implementation
/// selected by this build and CPU.
///
/// The scalar-reference feature deliberately takes precedence over runtime
/// CPU detection so benchmark output can prove that the controlled reference
/// path actually ran.
pub fn sq8_candidate_backend_name() -> &'static str {
    #[cfg(feature = "force-scalar-sq8")]
    {
        "scalar reference (forced feature)"
    }

    #[cfg(all(not(feature = "force-scalar-sq8"), target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            "AVX2/FMA (runtime detected)"
        } else {
            "scalar runtime fallback"
        }
    }

    #[cfg(all(not(feature = "force-scalar-sq8"), not(target_arch = "x86_64")))]
    {
        "scalar runtime fallback"
    }
}

/// Per-dimension scalar-quantized vector store with one byte per coordinate.
///
/// The code payload is one quarter of an f32 vector payload; the complete
/// store also carries per-dimension minimum and scale metadata. When every
/// coordinate is an integer in `0..=255`, construction preserves those byte
/// values exactly; other supported f32 inputs use per-dimension scaling.
pub struct SQ8Store {
    codes: Vec<u8>,
    mins: Vec<f32>,
    scales: Vec<f32>,
    dim: usize,
}

impl Drop for SQ8Store {
    fn drop(&mut self) {
        // Clear these owned accelerator allocations before releasing them.
        // This does not erase caller copies, persistence, or system-level copies.
        self.codes.zeroize();
        self.mins.zeroize();
        self.scales.zeroize();
    }
}

impl SQ8Store {
    /// Reassemble one persisted SQ8 component after validating its shape,
    /// metadata, and reconstructed numeric domain. Pass this component with
    /// the remaining validated parts to [`crate::PrismIndex::from_parts`] to
    /// reassemble a complete index.
    pub fn from_parts(
        codes: Vec<u8>,
        mins: Vec<f32>,
        scales: Vec<f32>,
        dim: usize,
    ) -> PrismResult<Self> {
        if dim == 0 {
            return Err(PrismError::InvalidFormat(
                "SQ8 dimension must be greater than zero".into(),
            ));
        }
        if mins.len() != dim || scales.len() != dim {
            return Err(PrismError::InvalidFormat(format!(
                "SQ8 mins/scales lengths ({}/{}) must equal dimension {dim}",
                mins.len(),
                scales.len()
            )));
        }
        if codes.len() % dim != 0 {
            return Err(PrismError::InvalidFormat(format!(
                "SQ8 code length {} must be divisible by dimension {dim}",
                codes.len()
            )));
        }
        let point_count = codes.len() / dim;
        if point_count > u32::MAX as usize {
            return Err(PrismError::Overflow(
                "SQ8 point count exceeds the u32 identifier space".into(),
            ));
        }
        if let Some((dimension, _)) = mins.iter().enumerate().find(|(_, min)| !min.is_finite()) {
            return Err(PrismError::InvalidFormat(format!(
                "SQ8 minimum for dimension {dimension} is not finite"
            )));
        }
        if let Some((dimension, scale)) = scales
            .iter()
            .enumerate()
            .find(|(_, scale)| !scale.is_finite() || **scale <= 0.0)
        {
            return Err(PrismError::InvalidFormat(format!(
                "SQ8 scale for dimension {dimension} must be finite and positive, got {scale}"
            )));
        }
        let component_limit = f32_distance_component_limit(dim);
        for (flat_index, &code) in codes.iter().enumerate() {
            let dimension = flat_index % dim;
            let reconstructed =
                f64::from(mins[dimension]) + f64::from(code) * f64::from(scales[dimension]);
            let search_reconstruction = f32::from(code).mul_add(scales[dimension], mins[dimension]);
            if !reconstructed.is_finite()
                || reconstructed.abs() > component_limit
                || !search_reconstruction.is_finite()
                || f64::from(search_reconstruction).abs() > component_limit
            {
                return Err(PrismError::InvalidFormat(format!(
                    "SQ8 reconstructed value at flat index {flat_index} must remain finite with absolute magnitude at most {component_limit:e} for dimension {dim}, got {reconstructed}"
                )));
            }
        }

        Ok(Self {
            codes,
            mins,
            scales,
            dim,
        })
    }

    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    pub fn mins(&self) -> &[f32] {
        &self.mins
    }

    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of stored point codes.
    pub fn len(&self) -> usize {
        self.codes.len() / self.dim
    }

    /// Whether the store contains no point codes.
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Build SQ8 codes. Uses identity quantization for integer `0..=255` data.
    pub fn build(store: &PointStore) -> Self {
        let n = store.len;
        let dim = store.dim;

        let all_integer_byte = (0..n).all(|i| {
            store
                .vector_unchecked(i as u32)
                .iter()
                .all(|&v| (0.0..=255.0).contains(&v) && v == v.round())
        });

        if all_integer_byte {
            let mut codes = vec![0u8; n * dim];
            for i in 0..n {
                let vec = store.vector_unchecked(i as u32);
                let off = i * dim;
                for d in 0..dim {
                    codes[off + d] = vec[d] as u8;
                }
            }
            return Self {
                codes,
                mins: vec![0.0; dim],
                scales: vec![1.0; dim],
                dim,
            };
        }

        let sample_n = n.min(10_000);

        let (mins, maxs) = if sample_n >= 200 {
            let mut mins = vec![0.0f32; dim];
            let mut maxs = vec![0.0f32; dim];
            for d in 0..dim {
                // Spread ids across the full range (a floored stride covers
                // only a prefix when n is not a multiple of sample_n).
                let mut sample: Vec<f32> = (0..sample_n)
                    .map(|s| {
                        let idx = ((s as u64 * n as u64) / sample_n as u64) as usize;
                        store.vector_unchecked(idx.min(n - 1) as u32)[d]
                    })
                    .collect();
                sample.sort_unstable_by(f32::total_cmp);
                let lo = sample_n / 200;
                let hi = sample_n.saturating_sub(1 + sample_n / 200);
                mins[d] = sample[lo];
                maxs[d] = sample[hi.max(lo + 1).min(sample_n - 1)];
            }
            (mins, maxs)
        } else {
            let mut mins = vec![f32::MAX; dim];
            let mut maxs = vec![f32::MIN; dim];
            for i in 0..n {
                let vec = store.vector_unchecked(i as u32);
                for d in 0..dim {
                    mins[d] = mins[d].min(vec[d]);
                    maxs[d] = maxs[d].max(vec[d]);
                }
            }
            (mins, maxs)
        };

        let scales: Vec<f32> = mins
            .iter()
            .zip(maxs.iter())
            .map(|(&mn, &mx)| {
                // Subtract in f64: two finite f32 extrema can have an
                // infinite f32 difference even though range / 255 is finite.
                let range = mx as f64 - mn as f64;
                if range > 0.0 {
                    (range / 255.0) as f32
                } else {
                    1.0
                }
            })
            .collect();

        let mut codes = vec![0u8; n * dim];
        for i in 0..n {
            let vec = store.vector_unchecked(i as u32);
            let off = i * dim;
            for d in 0..dim {
                let val = (f64::from(vec[d]) - f64::from(mins[d])) / f64::from(scales[d]);
                codes[off + d] = val.round().clamp(0.0, 255.0) as u8;
            }
        }

        Self {
            codes,
            mins,
            scales,
            dim,
        }
    }

    /// Get the quantized code for point id.
    #[inline]
    pub fn code(&self, id: u32) -> Option<&[u8]> {
        ((id as usize) < self.len()).then(|| self.code_unchecked(id))
    }

    /// Get a code by a previously validated internal point ID.
    #[inline]
    pub(crate) fn code_unchecked(&self, id: u32) -> &[u8] {
        let start = id as usize * self.dim;
        &self.codes[start..start + self.dim]
    }

    /// Quantize a f32 query vector to u8.
    pub fn quantize_query(&self, query: &[f32]) -> PrismResult<Vec<u8>> {
        if query.len() != self.dim {
            return Err(PrismError::InvalidInput(format!(
                "query length {} must equal SQ8 dimension {}",
                query.len(),
                self.dim
            )));
        }
        validate_f32_distance_domain(query, self.dim, "query")?;
        Ok(query
            .iter()
            .enumerate()
            .map(|(d, &v)| {
                let val = (f64::from(v) - f64::from(self.mins[d])) / f64::from(self.scales[d]);
                val.round().clamp(0.0, 255.0) as u8
            })
            .collect())
    }

    /// Scale-aware asymmetric L2 distance from a full-precision query to a
    /// stored code. Comparing raw code bytes is incorrect when dimensions have
    /// different quantization scales; reconstructing the stored value keeps
    /// candidate ordering in the original coordinate system.
    #[inline]
    pub(crate) fn asymmetric_l2(&self, query: &[f32], id: u32) -> f32 {
        assert_eq!(
            query.len(),
            self.dim,
            "query length must equal the SQ8 dimension"
        );
        let code = self.code_unchecked(id);

        #[cfg(all(target_arch = "x86_64", not(feature = "force-scalar-sq8")))]
        {
            // Widening eight codes at a time still reconstructs each dimension with
            // its own min/scale, so this stays the scalar path's scale-aware
            // metric rather than raw byte L2.
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { asymmetric_l2_avx2(query, code, &self.mins, &self.scales) };
            }
        }

        asymmetric_l2_scalar(query, code, &self.mins, &self.scales)
    }

    /// Scale-aware L2 distance between two stored codes.
    #[inline]
    pub(crate) fn code_l2(&self, a: u32, b: u32) -> f32 {
        let distance: f64 = self
            .code_unchecked(a)
            .iter()
            .zip(self.code_unchecked(b).iter())
            .zip(self.scales.iter())
            .map(|((&x, &y), &scale)| {
                let delta = (f64::from(x) - f64::from(y)) * f64::from(scale);
                delta * delta
            })
            .sum();
        debug_assert!(distance.is_finite() && distance <= f32::MAX as f64);
        distance as f32
    }
}

#[inline]
fn asymmetric_l2_scalar(query: &[f32], code: &[u8], mins: &[f32], scales: &[f32]) -> f32 {
    let distance = query
        .iter()
        .zip(code.iter())
        .zip(mins.iter().zip(scales.iter()))
        .map(|((&q, &c), (&min, &scale))| {
            let reconstructed = f32::from(c).mul_add(scale, min);
            let delta = q - reconstructed;
            delta * delta
        })
        .sum::<f32>();
    // PointStore and persisted-part validation bound every component so the
    // worst possible supported squared-L2 sum is below f32::MAX / 2.
    debug_assert!(distance.is_finite());
    distance
}

#[cfg(all(target_arch = "x86_64", not(feature = "force-scalar-sq8")))]
#[target_feature(enable = "avx2,fma")]
unsafe fn asymmetric_l2_avx2(query: &[f32], code: &[u8], mins: &[f32], scales: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let chunks = query.len() / 8;
    let mut sum = _mm256_setzero_ps();
    let query_ptr = query.as_ptr();
    let code_ptr = code.as_ptr();
    let min_ptr = mins.as_ptr();
    let scale_ptr = scales.as_ptr();

    for chunk in 0..chunks {
        let offset = chunk * 8;
        let packed = _mm_loadl_epi64(code_ptr.add(offset).cast::<__m128i>());
        let code_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(packed));
        let query_f32 = _mm256_loadu_ps(query_ptr.add(offset));
        let min_f32 = _mm256_loadu_ps(min_ptr.add(offset));
        let scale_f32 = _mm256_loadu_ps(scale_ptr.add(offset));
        let reconstructed = _mm256_fmadd_ps(code_f32, scale_f32, min_f32);
        let delta = _mm256_sub_ps(query_f32, reconstructed);
        sum = _mm256_fmadd_ps(delta, delta, sum);
    }

    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
    let mut distance = lanes.into_iter().sum::<f32>();

    for dimension in (chunks * 8)..query.len() {
        let reconstructed = f32::from(code[dimension]).mul_add(scales[dimension], mins[dimension]);
        let delta = query[dimension] - reconstructed;
        distance += delta * delta;
    }

    debug_assert!(distance.is_finite());
    distance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sq8_roundtrip() {
        let store = PointStore::from_parts(
            vec![0.0, 0.0, 0.0, 255.0, 255.0, 255.0],
            3,
            vec![vec![0, 0]],
        )
        .unwrap();
        let sq8 = SQ8Store::build(&store);
        assert_eq!(sq8.code(0), Some(&[0, 0, 0][..]));
        assert_eq!(sq8.code(1), Some(&[255, 255, 255][..]));
    }

    #[test]
    fn test_sq8_midpoint() {
        let store = PointStore::from_parts(
            vec![0.0, 0.0, 255.0, 255.0, 128.0, 128.0],
            2,
            vec![vec![0, 0, 0]],
        )
        .unwrap();
        let sq8 = SQ8Store::build(&store);
        assert_eq!(sq8.code(2).unwrap()[0], 128);
    }

    #[test]
    fn test_sq8_identity_quantization() {
        let store = PointStore::from_parts(
            vec![10.0, 200.0, 50.0, 150.0, 0.0, 255.0],
            2,
            vec![vec![0, 0, 0]],
        )
        .unwrap();
        let sq8 = SQ8Store::build(&store);
        assert_eq!(sq8.code(0), Some(&[10, 200][..]));
        assert_eq!(sq8.code(1), Some(&[50, 150][..]));
        assert_eq!(sq8.code(2), Some(&[0, 255][..]));
        assert_eq!(sq8.mins, vec![0.0, 0.0]);
        assert_eq!(sq8.scales, vec![1.0, 1.0]);
    }

    #[test]
    fn test_sq8_non_identity_for_float_data() {
        let store =
            PointStore::from_parts(vec![0.0, 0.0, 1000.0, 500.5], 2, vec![vec![0, 0]]).unwrap();
        let sq8 = SQ8Store::build(&store);
        assert_eq!(sq8.code(0), Some(&[0, 0][..]));
        assert_eq!(sq8.code(1), Some(&[255, 255][..]));
    }

    #[test]
    fn test_sq8_sampling_covers_tail_when_n_not_multiple_of_sample() {
        // n = 12,500 (> 10k sample cap, not a multiple of it): the first 10k
        // points sit in [0, 1], the last 2.5k at 100.5. A prefix-only sample
        // would estimate the range from the old points alone.
        let n = 12_500;
        let mut vectors = Vec::with_capacity(n);
        for i in 0..n {
            if i < 10_000 {
                vectors.push((i % 100) as f32 / 100.0 + 0.25);
            } else {
                vectors.push(100.5);
            }
        }
        let store = PointStore::from_parts(vectors, 1, vec![vec![0; n]]).unwrap();
        let sq8 = SQ8Store::build(&store);
        assert!(
            sq8.scales[0] > 0.3,
            "scale {} must reflect the tail range ~[0,100], not the prefix [0,1]",
            sq8.scales[0]
        );
        assert_eq!(sq8.code((n - 1) as u32).unwrap()[0], 255);
    }

    #[test]
    fn test_sq8_distance_ranking() {
        use super::super::distance;
        let store = PointStore::from_parts(
            vec![0.0, 0.0, 100.0, 100.0, 200.0, 200.0],
            2,
            vec![vec![0, 0, 0]],
        )
        .unwrap();
        let sq8 = SQ8Store::build(&store);
        let q = sq8.quantize_query(&[90.0, 90.0]).unwrap();
        let d0 = distance::l2_sq8(&q, sq8.code(0).unwrap());
        let d1 = distance::l2_sq8(&q, sq8.code(1).unwrap());
        let d2 = distance::l2_sq8(&q, sq8.code(2).unwrap());
        assert!(d1 < d0, "point 1 should be closer than point 0");
        assert!(d1 < d2, "point 1 should be closer than point 2");
    }

    #[test]
    fn scale_aware_distance_preserves_anisotropic_l2_order() {
        // Dimension 0 spans 1000 units while dimension 1 spans one. Raw code
        // L2 reverses A/B because it treats a byte in both dimensions equally.
        let store = PointStore::from_parts(
            vec![0.0, 0.0, 100.0, 0.0, 0.0, 1.0, 1000.0, 0.0],
            2,
            vec![vec![0; 4]],
        )
        .unwrap();
        let sq8 = SQ8Store::build(&store);
        let query = [0.0, 0.0];

        let query_code = sq8.quantize_query(&query).unwrap();
        let raw_a = super::super::distance::l2_sq8(&query_code, sq8.code(1).unwrap());
        let raw_b = super::super::distance::l2_sq8(&query_code, sq8.code(2).unwrap());
        assert!(raw_a < raw_b, "fixture must expose the raw-code reversal");

        let scaled_a = sq8.asymmetric_l2(&query, 1);
        let scaled_b = sq8.asymmetric_l2(&query, 2);
        assert!(
            scaled_b < scaled_a,
            "original-space nearest neighbor must survive SQ8 ranking"
        );

        let stored_a = sq8.code_l2(0, 1);
        let stored_b = sq8.code_l2(0, 2);
        assert!(
            stored_b < stored_a,
            "stored-code construction distance must use per-dimension scales"
        );
    }

    #[test]
    fn persisted_sq8_rejects_inconsistent_or_nonfinite_parts() {
        assert!(matches!(
            SQ8Store::from_parts(vec![1, 2, 3], vec![0.0, 0.0], vec![1.0, 1.0], 2),
            Err(PrismError::InvalidFormat(_))
        ));
        assert!(matches!(
            SQ8Store::from_parts(vec![1, 2], vec![0.0, 0.0], vec![1.0, 0.0], 2),
            Err(PrismError::InvalidFormat(_))
        ));
        assert!(matches!(
            SQ8Store::from_parts(vec![1], vec![f32::NAN], vec![1.0], 1),
            Err(PrismError::InvalidFormat(_))
        ));
    }

    #[test]
    fn persisted_sq8_valid_parts_round_trip() {
        let sq8 =
            SQ8Store::from_parts(vec![1, 2, 3, 4], vec![-1.0, 2.0], vec![0.5, 1.0], 2).unwrap();
        assert_eq!(sq8.len(), 2);
        assert!(!sq8.is_empty());
        assert_eq!(sq8.code(1), Some(&[3, 4][..]));
        assert_eq!(sq8.code(2), None);
    }

    #[test]
    fn sq8_rejects_extreme_queries_and_persisted_reconstructions() {
        let store = PointStore::from_parts(vec![0.0, 1.0], 1, vec![vec![0, 0]]).unwrap();
        let sq8 = SQ8Store::build(&store);
        assert!(matches!(
            sq8.quantize_query(&[f32::MAX]),
            Err(PrismError::InvalidInput(_))
        ));
        assert!(matches!(
            SQ8Store::from_parts(vec![255], vec![f32::MAX], vec![f32::MAX], 1),
            Err(PrismError::InvalidFormat(_))
        ));
    }

    #[test]
    fn sq8_safe_domain_distances_remain_finite() {
        let limit = (f32_distance_component_limit(2) * 0.99) as f32;
        let store = PointStore::from_parts(vec![limit, -limit, -limit, limit], 2, vec![vec![0, 0]])
            .unwrap();
        let sq8 = SQ8Store::build(&store);
        let query = [limit, -limit];
        assert!(sq8.asymmetric_l2(&query, 1).is_finite());
        assert!(sq8.code_l2(0, 1).is_finite());
    }

    #[test]
    fn dispatched_asymmetric_distance_matches_the_scalar_scale_aware_path() {
        for &dim in &[1usize, 7, 8, 9, 31, 128, 129] {
            let n = 17;
            let vectors: Vec<f32> = (0..n * dim)
                .map(|flat| {
                    let point = flat / dim;
                    let dimension = flat % dim;
                    ((point * 37 + dimension * 19) % 251) as f32 / 17.0
                        + (dimension as f32 * 0.013).sin()
                })
                .collect();
            let store = PointStore::from_parts(vectors, dim, vec![vec![0; n]]).unwrap();
            let sq8 = SQ8Store::build(&store);
            let query: Vec<f32> = (0..dim)
                .map(|dimension| {
                    ((dimension * 43 + 11) % 239) as f32 / 19.0 - (dimension as f32 * 0.021).cos()
                })
                .collect();

            for id in 0..n as u32 {
                let scalar =
                    asymmetric_l2_scalar(&query, sq8.code_unchecked(id), sq8.mins(), sq8.scales());
                let dispatched = sq8.asymmetric_l2(&query, id);
                let tolerance = scalar.abs().max(1.0) * 2.0e-5;
                assert!(
                    (dispatched - scalar).abs() <= tolerance,
                    "dimension={dim}, id={id}, scalar={scalar}, dispatched={dispatched}"
                );
            }
        }
    }
}
