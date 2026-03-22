use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use rayon::prelude::*;

use ::prism_ann::binary::BinaryStore;
use ::prism_ann::construct::{PrismConfig, PrismIndex};
use ::prism_ann::distance;
use ::prism_ann::distance::Metric;
use ::prism_ann::filter::Filter;
use ::prism_ann::ivf::{self, IvfIndex, SpMat};
use ::prism_ann::point::PointStore;

use std::collections::HashMap;

fn parse_metric(s: &str) -> PyResult<Metric> {
    match s {
        "l2" => Ok(Metric::L2),
        "ip" => Ok(Metric::InnerProduct),
        _ => Err(PyValueError::new_err(format!(
            "unknown metric '{}', expected 'l2' or 'ip'", s
        ))),
    }
}

fn parse_filter(dict: Option<HashMap<String, Vec<u32>>>, num_attrs: usize) -> PyResult<Filter> {
    let Some(dict) = dict else {
        return Ok(Filter::none());
    };
    if dict.is_empty() {
        return Ok(Filter::none());
    }
    let mut constraints = Vec::with_capacity(dict.len());
    for (key, values) in dict {
        let j: usize = key
            .strip_prefix("attr_")
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| PyValueError::new_err(format!(
                "invalid filter key '{}', expected 'attr_N'", key
            )))?;
        if j >= num_attrs {
            return Err(PyValueError::new_err(format!(
                "attribute index {} out of range (num_attributes={})", j, num_attrs
            )));
        }
        if values.is_empty() {
            return Err(PyValueError::new_err(format!(
                "filter for attr_{} has empty value list", j
            )));
        }
        constraints.push((j, values));
    }
    Ok(Filter::new(constraints))
}

#[pyclass(name = "Index")]
struct PyIndex {
    pending_vectors: Vec<f32>,
    pending_attrs: Vec<Vec<u32>>,
    dim: usize,
    num_attrs: usize,
    num_points: usize,
    config: PrismConfig,
    inner: Option<PrismIndex>,
}

#[pymethods]
impl PyIndex {
    #[new]
    #[pyo3(signature = (
        dim,
        num_attributes,
        m_local = 16,
        m_greedy = 12,
        m_random = 4,
        t = 2,
        alpha = 1.0,
        vamana_alpha = 1.0,
        beam_width = 120,
        metric = "l2",
        sigma_high = 0.10,
        sigma_low = 0.001,
        beta = 3.0,
        epsilon = 0.2,
        binary_rerank = 4,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        num_attributes: usize,
        m_local: usize,
        m_greedy: usize,
        m_random: usize,
        t: usize,
        alpha: f32,
        vamana_alpha: f32,
        beam_width: usize,
        metric: &str,
        sigma_high: f32,
        sigma_low: f32,
        beta: f32,
        epsilon: f32,
        binary_rerank: usize,
    ) -> PyResult<Self> {
        let metric = parse_metric(metric)?;
        let config = PrismConfig {
            m_local,
            m_greedy,
            m_random,
            t,
            alpha,
            vamana_alpha,
            beam_width,
            metric,
            sigma_high,
            sigma_low,
            beta,
            epsilon,
            binary_rerank,
        };
        Ok(Self {
            pending_vectors: Vec::new(),
            pending_attrs: (0..num_attributes).map(|_| Vec::new()).collect(),
            dim,
            num_attrs: num_attributes,
            num_points: 0,
            config,
            inner: None,
        })
    }

    /// Add vectors and attributes to the index (before building).
    ///
    /// vectors: numpy array of shape (n, dim), dtype float32
    /// attributes: numpy array of shape (n, num_attributes), dtype uint32
    fn add(
        &mut self,
        vectors: PyReadonlyArray2<f32>,
        attributes: PyReadonlyArray2<u32>,
    ) -> PyResult<()> {
        if self.inner.is_some() {
            return Err(PyValueError::new_err("cannot add after build()"));
        }
        let vshape = vectors.shape();
        let ashape = attributes.shape();
        if vshape[1] != self.dim {
            return Err(PyValueError::new_err(format!(
                "expected dim={}, got {}", self.dim, vshape[1]
            )));
        }
        if ashape[1] != self.num_attrs {
            return Err(PyValueError::new_err(format!(
                "expected {} attributes, got {}", self.num_attrs, ashape[1]
            )));
        }
        if vshape[0] != ashape[0] {
            return Err(PyValueError::new_err(
                "vectors and attributes must have same number of rows"
            ));
        }
        let n = vshape[0];
        let v_array = vectors.as_array();
        let a_array = attributes.as_array();

        // Append vectors (flatten row-major)
        for i in 0..n {
            self.pending_vectors.extend_from_slice(v_array.row(i).as_slice().unwrap());
        }
        // Append attributes (transpose: row-major → column-major)
        for i in 0..n {
            for j in 0..self.num_attrs {
                self.pending_attrs[j].push(a_array[[i, j]]);
            }
        }
        self.num_points += n;
        Ok(())
    }

    /// Build the PRISM index from accumulated data.
    fn build(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.inner.is_some() {
            return Err(PyValueError::new_err("index already built"));
        }
        if self.num_points == 0 {
            return Err(PyValueError::new_err("no data added; call add() first"));
        }
        let vectors = std::mem::take(&mut self.pending_vectors);
        let attrs = std::mem::take(&mut self.pending_attrs);
        let dim = self.dim;
        let config = self.config.clone();

        let index = py.allow_threads(move || {
            let store = PointStore::from_parts(vectors, dim, attrs);
            PrismIndex::build(store, config)
        });
        self.inner = Some(index);
        Ok(())
    }

    /// Search for k nearest neighbors.
    ///
    /// Returns (ids, distances) as 1D numpy arrays.
    #[pyo3(signature = (query, k = 10, ef = 200, filter = None))]
    #[allow(clippy::type_complexity)]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
        k: usize,
        ef: usize,
        filter: Option<HashMap<String, Vec<u32>>>,
    ) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>)> {
        let index = self.inner.as_ref()
            .ok_or_else(|| PyValueError::new_err("index not built; call build() first"))?;
        let q = query.as_array();
        let q_slice = q.as_slice().ok_or_else(|| {
            PyValueError::new_err("query must be C-contiguous")
        })?;
        if q_slice.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "query dim {} != index dim {}", q_slice.len(), self.dim
            )));
        }
        let f = parse_filter(filter, self.num_attrs)?;
        let results = index.search(q_slice, &f, k, ef);

        let ids: Vec<u32> = results.iter()
            .map(|r| index.original_ids[r.id as usize])
            .collect();
        let dists: Vec<f32> = results.iter().map(|r| r.dist).collect();

        Ok((
            PyArray1::from_vec(py, ids),
            PyArray1::from_vec(py, dists),
        ))
    }

    /// Batch search: multiple queries in parallel.
    ///
    /// Returns (ids, distances) as 2D numpy arrays of shape (nq, k).
    /// `filter`: single filter applied to all queries.
    /// `filters`: list of per-query filters (one dict per query, or None for no filter).
    #[pyo3(signature = (queries, k = 10, ef = 200, filter = None, filters = None))]
    #[allow(clippy::type_complexity)]
    fn batch_search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        ef: usize,
        filter: Option<HashMap<String, Vec<u32>>>,
        filters: Option<Vec<Option<HashMap<String, Vec<u32>>>>>,
    ) -> PyResult<(Bound<'py, PyArray2<u32>>, Bound<'py, PyArray2<f32>>)> {
        let index = self.inner.as_ref()
            .ok_or_else(|| PyValueError::new_err("index not built; call build() first"))?;
        let shape = queries.shape();
        let nq = shape[0];
        if shape[1] != self.dim {
            return Err(PyValueError::new_err(format!(
                "query dim {} != index dim {}", shape[1], self.dim
            )));
        }

        // Parse filters: per-query list takes priority over single broadcast filter
        let parsed_filters: Vec<Filter> = if let Some(filter_list) = filters {
            if filter_list.len() != nq {
                return Err(PyValueError::new_err(format!(
                    "filters length {} != query count {}", filter_list.len(), nq
                )));
            }
            filter_list.into_iter()
                .map(|f| parse_filter(f, self.num_attrs))
                .collect::<PyResult<Vec<_>>>()?
        } else {
            let f = parse_filter(filter, self.num_attrs)?;
            vec![f; nq]
        };

        // Copy query data to release GIL
        let q_array = queries.as_array();
        let q_data: Vec<f32> = q_array.iter().copied().collect();

        let all_results = py.allow_threads(|| {
            let results = index.batch_search(&q_data, &parsed_filters, nq, k, ef);
            results.iter()
                .map(|query_results| {
                    query_results.iter()
                        .map(|r| (index.original_ids[r.id as usize], r.dist))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        });

        // Pack into (nq, k) arrays with padding
        let mut ids = vec![u32::MAX; nq * k];
        let mut dists = vec![f32::INFINITY; nq * k];
        for (i, row) in all_results.iter().enumerate() {
            for (j, &(id, dist)) in row.iter().enumerate().take(k) {
                ids[i * k + j] = id;
                dists[i * k + j] = dist;
            }
        }

        let ids_arr = Array2::from_shape_vec((nq, k), ids).unwrap();
        let dists_arr = Array2::from_shape_vec((nq, k), dists).unwrap();

        Ok((
            PyArray2::from_owned_array(py, ids_arr),
            PyArray2::from_owned_array(py, dists_arr),
        ))
    }

    #[getter]
    fn dim(&self) -> usize { self.dim }

    #[getter]
    fn num_points(&self) -> usize { self.num_points }

    #[getter]
    fn num_attributes(&self) -> usize { self.num_attrs }

    #[getter]
    fn is_built(&self) -> bool { self.inner.is_some() }
}

// ---------------------------------------------------------------------------
// IVF² Index (YFCC-style: uint8 vectors + tag metadata)
// ---------------------------------------------------------------------------

#[pyclass(name = "IvfIndex")]
struct PyIvfIndex {
    ivf: Option<IvfIndex>,
    binary: Option<BinaryStore>,
    centroids_u8: Vec<u8>,
    dim: usize,
    n: usize,
    n_clusters: usize,
    kmeans_iters: usize,
}

#[pymethods]
impl PyIvfIndex {
    #[new]
    #[pyo3(signature = (n_clusters = 4000, kmeans_iters = 5))]
    fn new(n_clusters: usize, kmeans_iters: usize) -> Self {
        Self {
            ivf: None,
            binary: None,
            centroids_u8: Vec::new(),
            dim: 0,
            n: 0,
            n_clusters,
            kmeans_iters,
        }
    }

    /// Build IVF² index from uint8 vectors and CSR tag metadata.
    ///
    /// vectors: numpy (n, dim) uint8
    /// meta_indptr: numpy (n+1,) int64 — CSR row pointers
    /// meta_indices: numpy (nnz,) int32 — CSR column indices (tag IDs)
    /// n_tags: vocabulary size
    fn build(
        &mut self,
        py: Python<'_>,
        vectors: PyReadonlyArray2<u8>,
        meta_indptr: PyReadonlyArray1<i64>,
        meta_indices: PyReadonlyArray1<i32>,
        n_tags: usize,
    ) -> PyResult<()> {
        if self.ivf.is_some() {
            return Err(PyValueError::new_err("index already built"));
        }
        let shape = vectors.shape();
        let n = shape[0];
        let dim = shape[1];

        let base_u8: Vec<u8> = vectors.as_array().iter().copied().collect();
        let indptr: Vec<i64> = meta_indptr.as_array().iter().copied().collect();
        let indices: Vec<i32> = meta_indices.as_array().iter().copied().collect();
        let n_clusters = self.n_clusters;
        let kmeans_iters = self.kmeans_iters;

        let (built_ivf, built_binary, built_centroids) = py.allow_threads(move || {
            let meta = SpMat { rows: n, cols: n_tags, indptr, indices };
            let (assignments, centroids_u8) = ivf::kmeans(&base_u8, n, dim, n_clusters, kmeans_iters);
            let ivf = IvfIndex::build(&base_u8, &meta, &assignments, n, dim, n_clusters);

            let base_f32: Vec<f32> = ivf.vectors_u8.iter().map(|&v| v as f32).collect();
            let store = PointStore::from_parts(base_f32, dim, vec![vec![0u32; n]]);
            let binary = BinaryStore::build(&store);

            (ivf, binary, centroids_u8)
        });

        self.ivf = Some(built_ivf);
        self.binary = Some(built_binary);
        self.centroids_u8 = built_centroids;
        self.dim = dim;
        self.n = n;
        Ok(())
    }

    /// Batch filtered search. Returns (nq, k) uint32 array of original IDs.
    ///
    /// queries: numpy (nq, dim) uint8
    /// filter_indptr: numpy (nq+1,) int64 — CSR row pointers
    /// filter_indices: numpy (nnz,) int32 — CSR column indices (required tag IDs)
    #[pyo3(signature = (queries, filter_indptr, filter_indices, k = 10, ef = 50, nprobe = 60, binary_rerank = 0))]
    #[allow(clippy::too_many_arguments)]
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<u8>,
        filter_indptr: PyReadonlyArray1<i64>,
        filter_indices: PyReadonlyArray1<i32>,
        k: usize,
        ef: usize,
        nprobe: usize,
        binary_rerank: usize,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let ivf = self.ivf.as_ref()
            .ok_or_else(|| PyValueError::new_err("index not built; call build() first"))?;
        let binary = self.binary.as_ref().unwrap();

        let shape = queries.shape();
        let nq = shape[0];
        let dim = shape[1];
        if dim != self.dim {
            return Err(PyValueError::new_err(format!(
                "query dim {} != index dim {}", dim, self.dim
            )));
        }

        let queries_u8: Vec<u8> = queries.as_array().iter().copied().collect();
        let indptr: Vec<i64> = filter_indptr.as_array().iter().copied().collect();
        let indices: Vec<i32> = filter_indices.as_array().iter().copied().collect();
        let centroids = self.centroids_u8.clone();

        let results = py.allow_threads(|| {
            // Parse query tags from CSR
            let query_tags: Vec<Vec<usize>> = (0..nq)
                .map(|qi| {
                    let start = indptr[qi] as usize;
                    let end = indptr[qi + 1] as usize;
                    indices[start..end].iter().map(|&t| t as usize).collect()
                })
                .collect();

            // Precompute query binary codes
            let query_binary: Vec<Vec<u64>> = (0..nq)
                .into_par_iter()
                .map(|qi| {
                    let q_u8 = &queries_u8[qi * dim..(qi + 1) * dim];
                    let q_f32: Vec<f32> = q_u8.iter().map(|&v| v as f32).collect();
                    binary.encode_query(&q_f32)
                })
                .collect();

            // Precompute filtered cluster assignments
            let query_top_clusters: Vec<Vec<usize>> = (0..nq)
                .into_par_iter()
                .map(|qi| {
                    let q_u8 = &queries_u8[qi * dim..(qi + 1) * dim];
                    let tags = &query_tags[qi];

                    let candidates: std::borrow::Cow<[u16]> = if tags.len() == 1 {
                        let t = tags[0];
                        if t < ivf.tag_clusters.len() {
                            std::borrow::Cow::Borrowed(&ivf.tag_clusters[t])
                        } else {
                            std::borrow::Cow::Owned(vec![])
                        }
                    } else if tags.len() >= 2 {
                        let t0 = tags[0];
                        let t1 = tags[1];
                        let a = if t0 < ivf.tag_clusters.len() { &ivf.tag_clusters[t0][..] } else { &[] };
                        let b = if t1 < ivf.tag_clusters.len() { &ivf.tag_clusters[t1][..] } else { &[] };
                        std::borrow::Cow::Owned(ivf::sorted_intersect_u16(a, b))
                    } else {
                        std::borrow::Cow::Owned(vec![])
                    };

                    if candidates.is_empty() {
                        return vec![];
                    }

                    let mut cluster_dists: Vec<(usize, u32)> = candidates.iter()
                        .map(|&ci| {
                            let ci = ci as usize;
                            let cent = &centroids[ci * dim..(ci + 1) * dim];
                            (ci, distance::l2_sq8(q_u8, cent))
                        })
                        .collect();

                    let np = nprobe.min(cluster_dists.len());
                    cluster_dists.select_nth_unstable_by_key(np - 1, |&(_, d)| d);
                    cluster_dists.truncate(np);
                    cluster_dists.sort_unstable_by_key(|&(_, d)| d);
                    cluster_dists.iter().map(|&(ci, _)| ci).collect()
                })
                .collect();

            ivf.batch_search_mqcb(
                &queries_u8, nq, &query_tags, &query_binary, &query_top_clusters,
                binary, k, ef, nprobe, binary_rerank,
            )
        });

        // Pack into (nq, k) array with padding
        let mut ids = vec![u32::MAX; nq * k];
        for (i, row) in results.iter().enumerate() {
            for (j, &id) in row.iter().enumerate().take(k) {
                ids[i * k + j] = id;
            }
        }

        let arr = Array2::from_shape_vec((nq, k), ids).unwrap();
        Ok(PyArray2::from_owned_array(py, arr))
    }

    #[getter]
    fn dim(&self) -> usize { self.dim }

    #[getter]
    fn num_points(&self) -> usize { self.n }

    #[getter]
    fn is_built(&self) -> bool { self.ivf.is_some() }
}

#[pymodule]
fn prism_ann(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIndex>()?;
    m.add_class::<PyIvfIndex>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_filter_valid() {
        let mut dict = HashMap::new();
        dict.insert("attr_0".to_string(), vec![1, 2]);
        dict.insert("attr_2".to_string(), vec![5]);
        let filter = parse_filter(Some(dict), 3).unwrap();
        assert_eq!(filter.strength(), 2);
    }

    #[test]
    fn test_parse_filter_none() {
        let filter = parse_filter(None, 3).unwrap();
        assert_eq!(filter.strength(), 0);
    }

    #[test]
    fn test_parse_filter_empty() {
        let filter = parse_filter(Some(HashMap::new()), 3).unwrap();
        assert_eq!(filter.strength(), 0);
    }

    #[test]
    fn test_parse_filter_invalid_key() {
        let mut dict = HashMap::new();
        dict.insert("color".to_string(), vec![1]);
        assert!(parse_filter(Some(dict), 3).is_err());
    }

    #[test]
    fn test_parse_filter_out_of_range() {
        let mut dict = HashMap::new();
        dict.insert("attr_5".to_string(), vec![1]);
        assert!(parse_filter(Some(dict), 3).is_err());
    }

    #[test]
    fn test_parse_metric() {
        assert!(matches!(parse_metric("l2").unwrap(), Metric::L2));
        assert!(matches!(parse_metric("ip").unwrap(), Metric::InnerProduct));
        assert!(parse_metric("cosine").is_err());
    }
}
