//! Distance kernels.
//!
//! Every function here asserts that its inputs have equal length and panics
//! otherwise; the assertion message names the offending pair.

/// Distance metric.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    L2,
    InnerProduct,
    Cosine,
}

/// L2 squared distance between two vectors.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "L2 vectors must have equal lengths");

    #[cfg(target_arch = "x86_64")]
    {
        // FMA and AVX2 are separate CPUID features, so `_mm256_fmadd_ps` needs
        // both checked before entering the target-feature function.
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { l2_squared_avx2(a, b) };
        }
        // The horizontal reduction uses `_mm_movehdup_ps`, which is SSE3.
        if is_x86_feature_detected!("sse3") {
            return unsafe { l2_squared_sse(a, b) };
        }
    }

    l2_squared_scalar(a, b)
}

/// Inner product (negative for distance: higher IP = closer).
#[inline]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "inner-product vectors must have equal lengths"
    );

    // A finite f32 product can overflow before cancellation (MAX*MAX + MAX*-MAX),
    // turning a finite dot product into `inf + -inf = NaN`. f64 accumulation holds
    // every supported product; IP indexes scan exhaustively, so exactness wins.
    inner_product_scalar(a, b)
}

/// Cosine distance: `1 - (a . b) / (||a|| * ||b||)`. Returns 1.0 if either
/// vector has zero norm (degenerate case).
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine vectors must have equal lengths");
    // f64 accumulation stops near-identical vectors rounding to a similarity above
    // 1.0. The resulting negative distance is not cosmetic: cross-cell selection
    // raises distances to a configurable power, where it becomes NaN.
    let dot = dot_product_f64(a, b);
    let na = cosine_norm_squared(a);
    let nb = cosine_norm_squared(b);
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        1.0
    } else {
        (1.0 - (dot / denom).clamp(-1.0, 1.0)) as f32
    }
}

/// Euclidean norm accumulated with the same f64 contract as [`cosine`].
///
/// Exact cosine scans cache this value for stored rows and compute it once for
/// each query. Normalized f32 vectors are not assumed to have a norm of exactly
/// one: doing so would change distances around close ties after normalization
/// rounding.
#[inline]
pub(crate) fn cosine_norm(vector: &[f32]) -> f64 {
    cosine_norm_squared(vector).sqrt()
}

#[inline]
pub(crate) fn cosine_norm_squared(vector: &[f32]) -> f64 {
    dot_product_f64(vector, vector)
}

/// Cosine distance with both norms previously computed by [`cosine_norm`].
///
/// This is algebraically and numerically identical to [`cosine`] for the same
/// vectors: dot-product accumulation, zero handling, clamping, and the final
/// f32 conversion retain the public distance contract. Only invariant norm
/// work is removed from the per-query row scan.
#[inline]
pub(crate) fn cosine_with_norms(query: &[f32], row: &[f32], query_norm: f64, row_norm: f64) -> f32 {
    assert_eq!(
        query.len(),
        row.len(),
        "cosine vectors must have equal lengths"
    );

    let dot = dot_product_f64(query, row);

    let denom = query_norm * row_norm;
    if denom == 0.0 {
        1.0
    } else {
        (1.0 - (dot / denom).clamp(-1.0, 1.0)) as f32
    }
}

#[inline]
fn dot_product_f64(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_product_f64_avx2(a, b) };
        }
    }

    dot_product_f64_scalar(a, b)
}

#[inline]
fn dot_product_f64_scalar(a: &[f32], b: &[f32]) -> f64 {
    // Mirror the AVX2 four-lane accumulation and reduction tree exactly. f32
    // products are exact in f64, so separate multiply/add keeps the precision
    // contract without needing hardware FMA.
    let chunks = a.len() / 4;
    let mut lane0 = 0.0f64;
    let mut lane1 = 0.0f64;
    let mut lane2 = 0.0f64;
    let mut lane3 = 0.0f64;
    for chunk in 0..chunks {
        let offset = chunk * 4;
        lane0 += f64::from(a[offset]) * f64::from(b[offset]);
        lane1 += f64::from(a[offset + 1]) * f64::from(b[offset + 1]);
        lane2 += f64::from(a[offset + 2]) * f64::from(b[offset + 2]);
        lane3 += f64::from(a[offset + 3]) * f64::from(b[offset + 3]);
    }
    let mut total = (lane0 + lane2) + (lane1 + lane3);
    for index in chunks * 4..a.len() {
        total += f64::from(a[index]) * f64::from(b[index]);
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_f64_avx2(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;

    let mut acc = _mm256_setzero_pd();
    let chunks = a.len() / 4;
    let ap = a.as_ptr();
    let bp = b.as_ptr();

    for chunk in 0..chunks {
        let offset = chunk * 4;
        let left = _mm256_cvtps_pd(_mm_loadu_ps(ap.add(offset)));
        let right = _mm256_cvtps_pd(_mm_loadu_ps(bp.add(offset)));
        acc = _mm256_add_pd(acc, _mm256_mul_pd(left, right));
    }

    let low = _mm256_castpd256_pd128(acc);
    let high = _mm256_extractf128_pd(acc, 1);
    let pair = _mm_add_pd(low, high);
    let mut total = _mm_cvtsd_f64(_mm_add_sd(pair, _mm_unpackhi_pd(pair, pair)));

    for index in chunks * 4..a.len() {
        total += f64::from(*a.get_unchecked(index)) * f64::from(*b.get_unchecked(index));
    }
    total
}

/// Compute distance using the given metric.
#[inline]
pub fn distance(a: &[f32], b: &[f32], metric: Metric) -> f32 {
    match metric {
        Metric::L2 => l2_squared(a, b),
        Metric::InnerProduct => -inner_product(a, b),
        Metric::Cosine => cosine(a, b),
    }
}

/// Total-order u32 key for an f32: `a < b` iff `ord_key(a) < ord_key(b)`,
/// for any sign mix. Lets exact f32 distances flow through the u32
/// candidate heaps used for SQ8 ranking.
#[inline]
pub fn ord_key(x: f32) -> u32 {
    let b = x.to_bits();
    if b & 0x8000_0000 == 0 {
        b | 0x8000_0000
    } else {
        !b
    }
}

/// Recover the `f32` encoded by [`ord_key`].
#[inline]
pub(crate) fn from_ord_key(key: u32) -> f32 {
    let bits = if key & 0x8000_0000 != 0 {
        key & 0x7fff_ffff
    } else {
        !key
    };
    f32::from_bits(bits)
}

/// L2-normalize each `dim`-stride row in place; zero rows are left unchanged.
///
/// # Panics
///
/// Panics when `dim` is zero or `data.len()` is not divisible by `dim`.
pub fn normalize_rows(data: &mut [f32], dim: usize) {
    assert!(dim > 0, "normalization dimension must be greater than zero");
    assert_eq!(
        data.len() % dim,
        0,
        "normalized data length must be divisible by the dimension"
    );
    for row in data.chunks_mut(dim) {
        let norm = row.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
        if norm > 0.0 {
            // Casting the reciprocal to f32 first overflows on a subnormal row and
            // `0 * inf` yields NaN, so divide in f64 and cast only the result.
            for x in row {
                *x = (f64::from(*x) / norm) as f32;
            }
        }
    }
}

/// L2-normalized copy of a single vector.
pub fn normalized(v: &[f32]) -> Vec<f32> {
    if v.is_empty() {
        return Vec::new();
    }
    let mut out = v.to_vec();
    normalize_rows(&mut out, v.len());
    out
}

/// L2 squared distance between two SQ8 (u8) vectors.
#[inline]
pub fn l2_sq8(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len(), "SQ8 vectors must have equal lengths");

    #[cfg(target_arch = "x86_64")]
    {
        // AVX2 accumulates in u32 lanes, so it is safe only while the worst sum
        // (255^2 per dimension) fits; larger vectors take the u64 scalar path.
        const MAX_U32_DIM: usize = u32::MAX as usize / (255 * 255);
        if a.len() <= MAX_U32_DIM && is_x86_feature_detected!("avx2") {
            return unsafe { l2_sq8_avx2(a, b) } as u64;
        }
    }

    l2_sq8_scalar(a, b)
}

fn l2_sq8_scalar(a: &[u8], b: &[u8]) -> u64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x as i32 - y as i32;
            (d * d) as u64
        })
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_sq8_avx2(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 32;
    let remainder = n % 32;

    let mut acc = _mm256_setzero_si256();
    let ap = a.as_ptr();
    let bp = b.as_ptr();

    for i in 0..chunks {
        let va = _mm256_loadu_si256(ap.add(i * 32) as *const __m256i);
        let vb = _mm256_loadu_si256(bp.add(i * 32) as *const __m256i);

        let a_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(va));
        let b_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(vb));
        let diff_lo = _mm256_sub_epi16(a_lo, b_lo);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(diff_lo, diff_lo));

        let a_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(va, 1));
        let b_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(vb, 1));
        let diff_hi = _mm256_sub_epi16(a_hi, b_hi);
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(diff_hi, diff_hi));
    }

    let hi = _mm256_extracti128_si256(acc, 1);
    let lo = _mm256_castsi256_si128(acc);
    let sum128 = _mm_add_epi32(lo, hi);
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, hi64);
    let hi32 = _mm_shuffle_epi32(sum64, 1);
    let sum32 = _mm_add_epi32(sum64, hi32);
    let mut total = _mm_cvtsi128_si32(sum32) as u32;

    let offset = chunks * 32;
    for i in 0..remainder {
        let d = a[offset + i] as i32 - b[offset + i] as i32;
        total += (d * d) as u32;
    }

    total
}

/// Hamming distance between binary codes packed as u64 words (XOR + POPCNT).
///
/// # Panics
///
/// Panics when the codes have different word lengths.
#[inline]
pub fn hamming(a: &[u64], b: &[u64]) -> u64 {
    assert_eq!(a.len(), b.len(), "binary codes must have equal lengths");
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| u64::from((x ^ y).count_ones()))
        .sum()
}

fn l2_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn inner_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum::<f64>() as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 8;
    let remainder = n % 8;

    let mut sum = _mm256_setzero_ps();

    let ap = a.as_ptr();
    let bp = b.as_ptr();

    for i in 0..chunks {
        let va = _mm256_loadu_ps(ap.add(i * 8));
        let vb = _mm256_loadu_ps(bp.add(i * 8));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);

    let offset = chunks * 8;
    for i in 0..remainder {
        let d = a[offset + i] - b[offset + i];
        total += d * d;
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse3")]
unsafe fn l2_squared_sse(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum = _mm_setzero_ps();
    let ap = a.as_ptr();
    let bp = b.as_ptr();

    for i in 0..chunks {
        let va = _mm_loadu_ps(ap.add(i * 4));
        let vb = _mm_loadu_ps(bp.add(i * 4));
        let diff = _mm_sub_ps(va, vb);
        let sq = _mm_mul_ps(diff, diff);
        sum = _mm_add_ps(sum, sq);
    }

    let shuf = _mm_movehdup_ps(sum);
    let sums = _mm_add_ps(sum, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let result = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(result);

    let offset = chunks * 4;
    for i in 0..remainder {
        let d = a[offset + i] - b[offset + i];
        total += d * d;
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_squared() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let d = l2_squared(&a, &b);
        assert!((d - 27.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "L2 vectors must have equal lengths")]
    fn l2_rejects_mismatched_lengths_in_all_profiles() {
        let _ = l2_squared(&[1.0; 16], &[1.0; 8]);
    }

    #[test]
    #[should_panic(expected = "inner-product vectors must have equal lengths")]
    fn inner_product_rejects_mismatched_lengths_in_all_profiles() {
        let _ = inner_product(&[1.0; 16], &[1.0; 8]);
    }

    #[test]
    #[should_panic(expected = "SQ8 vectors must have equal lengths")]
    fn sq8_rejects_mismatched_lengths_in_all_profiles() {
        let _ = l2_sq8(&[1; 32], &[1; 16]);
    }

    #[test]
    fn test_l2_squared_large() {
        let dim = 128;
        let a: Vec<f32> = (0..dim).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32) + 1.0).collect();
        let d = l2_squared(&a, &b);
        assert!((d - dim as f32).abs() < 1e-3); // each diff=1, so sum = dim
    }

    #[test]
    fn test_inner_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let ip = inner_product(&a, &b);
        assert!((ip - 32.0).abs() < 1e-6);
    }

    #[test]
    fn inner_product_preserves_finite_cancellation_after_f32_products_would_overflow() {
        let a = [f32::MAX, f32::MAX];
        let b = [f32::MAX, -f32::MAX];
        let ip = inner_product(&a, &b);
        assert_eq!(ip, 0.0);
        assert!(ip.is_finite());
    }

    #[test]
    fn test_distance_metric() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((distance(&a, &b, Metric::L2) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_sq8() {
        let a: Vec<u8> = vec![10, 20, 30, 40];
        let b: Vec<u8> = vec![11, 22, 27, 45];
        // (1)^2 + (2)^2 + (3)^2 + (5)^2 = 1 + 4 + 9 + 25 = 39
        assert_eq!(l2_sq8(&a, &b), 39);
    }

    #[test]
    fn test_hamming() {
        let distance: u64 = hamming(&[0b1010], &[0b1001]);
        assert_eq!(distance, 2);
        assert_eq!(hamming(&[0, 0], &[0, 0]), 0);
        assert_eq!(hamming(&[u64::MAX], &[0]), 64);
        assert_eq!(hamming(&[u64::MAX, 0], &[0, u64::MAX]), 128);
    }

    #[test]
    fn ord_key_is_monotone_across_signs() {
        let vals = [-1e9f32, -100.0, -1.5, -0.0, 0.0, 1e-10, 3.0, 1e9];
        for w in vals.windows(2) {
            assert!(
                ord_key(w[0]) <= ord_key(w[1]),
                "ord_key({}) > ord_key({})",
                w[0],
                w[1]
            );
        }
        assert!(ord_key(-1.0) < ord_key(1.0));
        for value in vals {
            assert_eq!(from_ord_key(ord_key(value)).to_bits(), value.to_bits());
        }
    }

    #[test]
    fn normalize_rows_unit_norms_and_zero_rows() {
        let mut data = vec![3.0, 4.0, 0.0, 0.0, -2.0, 0.0];
        normalize_rows(&mut data, 2);
        assert!((data[0] - 0.6).abs() < 1e-6);
        assert!((data[1] - 0.8).abs() < 1e-6);
        assert_eq!(&data[2..4], &[0.0, 0.0]);
        assert!((data[4] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalization_handles_subnormal_and_empty_vectors_without_nonfinite_values() {
        let smallest = f32::from_bits(1);
        let mut row = [smallest, 0.0];
        normalize_rows(&mut row, 2);
        assert_eq!(row, [1.0, 0.0]);
        assert!(row.iter().all(|value| value.is_finite()));

        assert!(normalized(&[]).is_empty());
    }

    #[test]
    fn test_l2_sq8_large() {
        let dim = 128;
        let a: Vec<u8> = (0..dim).map(|i| i as u8).collect();
        let b: Vec<u8> = (0..dim).map(|i| (i as u8).wrapping_add(1)).collect();
        // Every pair differs by exactly 1 (a[127]=127, b[127]=128), so the
        // squared sum is dim = 128.
        assert_eq!(l2_sq8(&a, &b), 128);
    }

    #[test]
    fn l2_sq8_does_not_overflow_u32_boundary() {
        let dim = u32::MAX as usize / (255 * 255) + 1;
        let a = vec![0u8; dim];
        let b = vec![255u8; dim];
        let expected = dim as u64 * 255u64 * 255u64;
        assert!(expected > u32::MAX as u64);
        assert_eq!(l2_sq8(&a, &b), expected);
    }

    #[test]
    fn cosine_orthogonal_is_one() {
        assert!((cosine(&[1.0, 0.0], &[0.0, 1.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_parallel_is_zero() {
        assert!(cosine(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).abs() < 1e-6);
    }

    #[test]
    fn cosine_is_clamped_for_near_identical_vectors() {
        let a: Vec<f32> = (0..1536)
            .map(|i| (i as f32 * 0.017).sin() + 0.25 * (i as f32 * 0.031).cos())
            .collect();
        let mut b = a.clone();
        b[517] = f32::from_bits(b[517].to_bits() + 1);

        let self_distance = cosine(&a, &a);
        let near_distance = cosine(&a, &b);
        assert_eq!(self_distance, 0.0);
        assert!((0.0..=2.0).contains(&near_distance));
    }

    #[test]
    fn cached_cosine_norms_are_bit_exact_for_edge_cases() {
        fn assert_same(query: &[f32], row: &[f32]) {
            let expected = cosine(query, row);
            let actual = cosine_with_norms(query, row, cosine_norm(query), cosine_norm(row));
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "cached query norm changed cosine({query:?}, {row:?}): {actual} != {expected}"
            );
        }

        let smallest = f32::from_bits(1);
        let largest_search_domain_value = 1.0e18f32;
        for (query, row) in [
            (vec![0.0, 0.0, 0.0], vec![1.0, -2.0, 3.0]),
            (vec![1.0, -2.0, 3.0], vec![0.0, 0.0, 0.0]),
            (vec![smallest, 0.0, 0.0], vec![smallest, 0.0, 0.0]),
            (
                vec![
                    largest_search_domain_value,
                    -largest_search_domain_value,
                    1.0,
                ],
                vec![
                    -largest_search_domain_value,
                    largest_search_domain_value,
                    -1.0,
                ],
            ),
            (vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]),
            (vec![1.0, 2.0, 3.0], vec![-1.0, -2.0, -3.0]),
        ] {
            assert_same(&query, &row);
        }
    }

    #[test]
    fn cached_cosine_norms_preserve_near_tie_ordering_and_distances() {
        let dim = 64;
        let query: Vec<f32> = (0..dim)
            .map(|i| (i as f32 * 0.071).sin() + 0.3 * (i as f32 * 0.113).cos())
            .collect();
        let query = normalized(&query);

        let epsilon = 0.001f32;
        let mut rows = Vec::new();
        for perturbation in [
            epsilon,
            f32::from_bits(epsilon.to_bits() + 1),
            f32::from_bits(epsilon.to_bits() + 2),
            -epsilon,
        ] {
            let mut row = query.clone();
            row[17] += perturbation;
            rows.push(normalized(&row));
        }
        rows.push(vec![0.0; dim]);

        let query_norm = cosine_norm(&query);
        let mut public: Vec<_> = rows
            .iter()
            .enumerate()
            .map(|(id, row)| (cosine(&query, row), id))
            .collect();
        let mut cached: Vec<_> = rows
            .iter()
            .enumerate()
            .map(|(id, row)| {
                (
                    cosine_with_norms(&query, row, query_norm, cosine_norm(row)),
                    id,
                )
            })
            .collect();
        public.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
        cached.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));

        assert_eq!(
            cached
                .iter()
                .map(|(dist, id)| (dist.to_bits(), *id))
                .collect::<Vec<_>>(),
            public
                .iter()
                .map(|(dist, id)| (dist.to_bits(), *id))
                .collect::<Vec<_>>()
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_dot_matches_portable_reduction_bits() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        for len in [0, 1, 3, 31, 32, 33, 63, 64, 127, 1024, 1027] {
            let mut left: Vec<f32> = (0..len)
                .map(|index| (index as f32 * 0.017).sin() + 0.25 * (index as f32 * 0.031).cos())
                .collect();
            let mut right: Vec<f32> = (0..len)
                .map(|index| (index as f32 * 0.023).cos() - 0.4 * (index as f32 * 0.011).sin())
                .collect();
            if len >= 3 {
                left[0] = 1.0e18;
                left[1] = 1.0e18;
                right[0] = 1.0e18;
                right[1] = -1.0e18;
                left[len - 1] = f32::from_bits(1);
                right[len - 1] = -f32::from_bits(1);
            }

            let portable = dot_product_f64_scalar(&left, &right);
            let dispatched = unsafe { dot_product_f64_avx2(&left, &right) };
            assert_eq!(
                dispatched.to_bits(),
                portable.to_bits(),
                "AVX2 dot changed the fixed f64 reduction for length {len}"
            );
        }
    }

    #[test]
    fn cosine_antiparallel_is_two() {
        assert!((cosine(&[1.0, 0.0], &[-1.0, 0.0]) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_vector_returns_one() {
        assert_eq!(cosine(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]), 1.0);
    }

    #[test]
    fn metric_dispatch_cosine() {
        let d = distance(&[1.0, 0.0], &[0.0, 1.0], Metric::Cosine);
        assert!((d - 1.0).abs() < 1e-6);
    }
}
