use super::error::{PrismError, PrismResult};
use super::point::PointStore;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{self, BufReader, Read};
use std::path::Path;

/// Load vectors from .fvecs format (used by SIFT1M, GIST1M, etc.).
///
/// Format: each vector is preceded by a 4-byte little-endian int (dimension),
/// followed by `dim` little-endian f32 values.
pub fn load_fvecs(path: &Path) -> io::Result<(Vec<f32>, usize)> {
    let file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let dim = read_positive_dimension(&mut reader, "fvecs")?;
    let (n, value_count) = vector_layout(file_len, dim, 4, "fvecs")?;
    let mut vectors = reserved_vec(value_count, "fvecs vector values")?;

    for row in 0..n {
        if row > 0 {
            validate_row_dimension(&mut reader, dim, row, "fvecs")?;
        }
        for column in 0..dim {
            let value = reader.read_f32::<LittleEndian>()?;
            if !value.is_finite() {
                return Err(invalid_data(format!(
                    "fvecs row {row}, column {column} is not finite"
                )));
            }
            vectors.push(value);
        }
    }
    ensure_eof(&mut reader, "fvecs")?;
    Ok((vectors, dim))
}

/// Load vectors from .bvecs format (unsigned byte vectors).
pub fn load_bvecs(path: &Path) -> io::Result<(Vec<f32>, usize)> {
    let file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let dim = read_positive_dimension(&mut reader, "bvecs")?;
    let (n, value_count) = vector_layout(file_len, dim, 1, "bvecs")?;
    let mut vectors = reserved_vec(value_count, "bvecs vector values")?;
    let mut row_values = reserved_vec(dim, "bvecs row")?;
    row_values.resize(dim, 0u8);

    for row in 0..n {
        if row > 0 {
            validate_row_dimension(&mut reader, dim, row, "bvecs")?;
        }
        reader.read_exact(&mut row_values)?;
        vectors.extend(row_values.iter().map(|&value| value as f32));
    }
    ensure_eof(&mut reader, "bvecs")?;
    Ok((vectors, dim))
}

/// Load integer vectors from .ivecs format (ground truth indices).
pub fn load_ivecs(path: &Path) -> io::Result<Vec<Vec<u32>>> {
    let file = std::fs::File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut reader = BufReader::new(file);
    let k = read_positive_dimension(&mut reader, "ivecs")?;
    let (n, _) = vector_layout(file_len, k, 4, "ivecs")?;
    let mut result = reserved_vec(n, "ivecs rows")?;

    for row in 0..n {
        if row > 0 {
            validate_row_dimension(&mut reader, k, row, "ivecs")?;
        }
        let mut ids = reserved_vec(k, "ivecs row")?;
        for column in 0..k {
            let id = reader.read_i32::<LittleEndian>()?;
            if id < 0 {
                return Err(invalid_data(format!(
                    "ivecs row {row}, column {column} has negative identifier {id}"
                )));
            }
            ids.push(id as u32);
        }
        result.push(ids);
    }
    ensure_eof(&mut reader, "ivecs")?;
    Ok(result)
}

/// Build a PointStore from loaded vectors and deterministic synthetic attributes.
///
/// Generates `cardinalities.len()` attribute dimensions. Attribute `j` uses
/// `(point_id / stride_j) % cardinalities[j]`; all combinations are populated
/// only when the point count is at least their checked Cartesian-product size.
pub fn build_store_with_synthetic_attrs(
    vectors: Vec<f32>,
    dim: usize,
    cardinalities: &[usize],
) -> PrismResult<PointStore> {
    if dim == 0 {
        return Err(PrismError::InvalidInput(
            "synthetic-store dimension must be greater than zero".into(),
        ));
    }
    if vectors.len() % dim != 0 {
        return Err(PrismError::InvalidInput(format!(
            "synthetic-store vector length {} must be divisible by dimension {dim}",
            vectors.len()
        )));
    }
    let n = vectors.len() / dim;
    let mut attrs = Vec::new();
    attrs
        .try_reserve_exact(cardinalities.len())
        .map_err(|error| {
            PrismError::Overflow(format!("cannot allocate synthetic attributes: {error}"))
        })?;

    let mut stride = 1usize;
    for (j, &cardinality) in cardinalities.iter().enumerate() {
        if cardinality == 0 {
            return Err(PrismError::InvalidInput(format!(
                "synthetic attribute {j} cardinality must be greater than zero"
            )));
        }
        if cardinality as u128 > u32::MAX as u128 + 1 {
            return Err(PrismError::Overflow(format!(
                "synthetic attribute {j} cardinality exceeds the u32 value space"
            )));
        }

        let mut column = Vec::new();
        column.try_reserve_exact(n).map_err(|error| {
            PrismError::Overflow(format!("cannot allocate synthetic attribute {j}: {error}"))
        })?;
        column.extend((0..n).map(|point| ((point / stride) % cardinality) as u32));
        attrs.push(column);

        if j + 1 < cardinalities.len() {
            stride = stride.checked_mul(cardinality).ok_or_else(|| {
                PrismError::Overflow(format!(
                    "synthetic attribute stride overflows before dimension {}",
                    j + 1
                ))
            })?;
        }
    }

    PointStore::from_parts(vectors, dim, attrs)
}

fn read_positive_dimension(reader: &mut impl Read, format: &str) -> io::Result<usize> {
    let dimension = reader.read_i32::<LittleEndian>()?;
    if dimension <= 0 {
        return Err(invalid_data(format!(
            "{format} dimension must be positive, got {dimension}"
        )));
    }
    Ok(dimension as usize)
}

fn validate_row_dimension(
    reader: &mut impl Read,
    expected: usize,
    row: usize,
    format: &str,
) -> io::Result<()> {
    let dimension = reader.read_i32::<LittleEndian>()?;
    if dimension <= 0 || dimension as usize != expected {
        return Err(invalid_data(format!(
            "{format} row {row} dimension {dimension} does not match first-row dimension {expected}"
        )));
    }
    Ok(())
}

fn vector_layout(
    file_len: u64,
    dimension: usize,
    bytes_per_value: u64,
    format: &str,
) -> io::Result<(usize, usize)> {
    let payload = (dimension as u64)
        .checked_mul(bytes_per_value)
        .ok_or_else(|| invalid_data(format!("{format} row size overflows")))?;
    let stride = 4u64
        .checked_add(payload)
        .ok_or_else(|| invalid_data(format!("{format} row stride overflows")))?;
    if file_len % stride != 0 {
        return Err(invalid_data(format!(
            "{format} file length {file_len} is not an exact multiple of row stride {stride}"
        )));
    }
    let rows = usize::try_from(file_len / stride)
        .map_err(|_| invalid_data(format!("{format} row count exceeds addressable memory")))?;
    if rows == 0 {
        return Err(invalid_data(format!(
            "{format} file contains no complete rows"
        )));
    }
    let values = rows
        .checked_mul(dimension)
        .ok_or_else(|| invalid_data(format!("{format} value count overflows")))?;
    Ok((rows, values))
}

fn reserved_vec<T>(capacity: usize, what: &str) -> io::Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|error| invalid_data(format!("cannot allocate {what}: {error}")))?;
    Ok(values)
}

fn ensure_eof(reader: &mut impl Read, format: &str) -> io::Result<()> {
    let mut trailing = [0u8; 1];
    if reader.read(&mut trailing)? != 0 {
        return Err(invalid_data(format!(
            "{format} contains trailing bytes after the declared rows"
        )));
    }
    Ok(())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    fn temporary_file(extension: &str, bytes: &[u8]) -> std::path::PathBuf {
        let serial = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "prism-io-test-{}-{serial}.{extension}",
            std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn test_build_store_with_synthetic_attrs() {
        let vectors = vec![0.0f32; 100 * 4];
        let store = build_store_with_synthetic_attrs(vectors, 4, &[10, 5, 3]).unwrap();
        assert_eq!(store.len(), 100);
        assert_eq!(store.k(), 3);
        assert!(store.cardinality(0).unwrap() <= 10);
        assert!(store.cardinality(1).unwrap() <= 5);
        assert!(store.cardinality(2).unwrap() <= 3);
    }

    #[test]
    fn synthetic_attributes_reject_zero_and_overflowing_cardinalities() {
        assert!(matches!(
            build_store_with_synthetic_attrs(vec![0.0], 1, &[0]),
            Err(PrismError::InvalidInput(_))
        ));
        if usize::BITS >= 64 {
            let huge = u32::MAX as usize + 1;
            assert!(matches!(
                build_store_with_synthetic_attrs(vec![0.0], 1, &[huge, huge, 2]),
                Err(PrismError::Overflow(_))
            ));
        }
    }

    #[test]
    fn vector_loaders_reject_negative_mismatched_and_trailing_dimensions() {
        let negative = temporary_file("fvecs", &(-1i32).to_le_bytes());
        assert_eq!(
            load_fvecs(&negative).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        std::fs::remove_file(negative).unwrap();

        let mut mismatched = Vec::new();
        mismatched.extend_from_slice(&2i32.to_le_bytes());
        mismatched.extend_from_slice(&1.0f32.to_le_bytes());
        mismatched.extend_from_slice(&2.0f32.to_le_bytes());
        mismatched.extend_from_slice(&3i32.to_le_bytes());
        mismatched.extend_from_slice(&3.0f32.to_le_bytes());
        mismatched.extend_from_slice(&4.0f32.to_le_bytes());
        let mismatched = temporary_file("fvecs", &mismatched);
        assert_eq!(
            load_fvecs(&mismatched).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        std::fs::remove_file(mismatched).unwrap();

        let mut trailing = Vec::new();
        trailing.extend_from_slice(&1i32.to_le_bytes());
        trailing.push(7);
        trailing.push(8);
        let trailing = temporary_file("bvecs", &trailing);
        assert_eq!(
            load_bvecs(&trailing).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        std::fs::remove_file(trailing).unwrap();
    }

    #[test]
    fn vector_loaders_reject_nonfinite_values_and_negative_ids() {
        let mut fvecs = Vec::new();
        fvecs.extend_from_slice(&1i32.to_le_bytes());
        fvecs.extend_from_slice(&f32::NAN.to_le_bytes());
        let fvecs = temporary_file("fvecs", &fvecs);
        assert_eq!(
            load_fvecs(&fvecs).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        std::fs::remove_file(fvecs).unwrap();

        let mut ivecs = Vec::new();
        ivecs.extend_from_slice(&1i32.to_le_bytes());
        ivecs.extend_from_slice(&(-1i32).to_le_bytes());
        let ivecs = temporary_file("ivecs", &ivecs);
        assert_eq!(
            load_ivecs(&ivecs).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        std::fs::remove_file(ivecs).unwrap();
    }
}
