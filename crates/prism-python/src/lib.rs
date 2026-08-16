use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use ::prism_ann::construct::{PrismConfig, PrismIndex};
use ::prism_ann::distance;
use ::prism_ann::distance::Metric;
use ::prism_ann::filter::Filter;
use ::prism_ann::ivf::{self, IvfIndex, SpMat};
use ::prism_ann::point::PointStore;
use ::prism_ann::PrismError;

use std::collections::HashMap;

/// One Python filter: attribute name to its allowed values.
type PyFilter = HashMap<String, Vec<u32>>;

/// Per-query filters; a `None` slot leaves that query unfiltered.
type PerQueryFilters = Vec<Option<PyFilter>>;

/// `(ids, distances)` returned to Python for a single query.
type QueryOutput<'py> = PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<f32>>)>;

/// `(ids, distances)` returned to Python for a batch of queries.
type BatchOutput<'py> = PyResult<(Bound<'py, PyArray2<u32>>, Bound<'py, PyArray2<f32>>)>;

fn prism_error(error: PrismError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn checked_output_len(nq: usize, k: usize) -> PyResult<usize> {
    if nq > isize::MAX as usize || k > isize::MAX as usize {
        return Err(PyValueError::new_err(
            "output dimensions exceed the platform array-index space",
        ));
    }
    let elements = nq
        .checked_mul(k)
        .ok_or_else(|| PyValueError::new_err("output shape nq * k overflows usize"))?;
    let bytes = elements
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| PyValueError::new_err("output byte size overflows usize"))?;
    if bytes > isize::MAX as usize {
        return Err(PyValueError::new_err(
            "output array exceeds the platform allocation limit",
        ));
    }
    Ok(elements)
}

fn padded_vec<T: Clone>(len: usize, value: T, name: &str) -> PyResult<Vec<T>> {
    let mut output = Vec::new();
    output.try_reserve_exact(len).map_err(|error| {
        PyValueError::new_err(format!("cannot allocate {name} output: {error}"))
    })?;
    output.resize(len, value);
    Ok(output)
}

fn original_id(index: &PrismIndex, internal_id: u32) -> PyResult<u32> {
    index.original_id(internal_id).ok_or_else(|| {
        PyValueError::new_err(format!(
            "search returned out-of-range internal ID {internal_id}"
        ))
    })
}

fn parse_metric(s: &str) -> PyResult<Metric> {
    match s {
        "l2" => Ok(Metric::L2),
        "ip" | "inner_product" => Ok(Metric::InnerProduct),
        "cosine" => Ok(Metric::Cosine),
        _ => Err(PyValueError::new_err(format!(
            "unknown metric '{}', expected 'l2', 'ip', or 'cosine'",
            s
        ))),
    }
}

fn validate_csr(
    indptr: &[i64],
    indices: &[i32],
    rows: usize,
    cols: usize,
    name: &str,
) -> PyResult<()> {
    let expected = rows
        .checked_add(1)
        .ok_or_else(|| PyValueError::new_err(format!("{name} row count is too large")))?;
    if indptr.len() != expected {
        return Err(PyValueError::new_err(format!(
            "{name} indptr length {} must equal rows + 1 ({expected})",
            indptr.len()
        )));
    }
    if indptr.first().copied() != Some(0) {
        return Err(PyValueError::new_err(format!(
            "{name} indptr must start at zero"
        )));
    }
    let nnz = indices.len() as i64;
    if indptr
        .windows(2)
        .any(|pair| pair[0] < 0 || pair[0] > pair[1] || pair[1] > nnz)
    {
        return Err(PyValueError::new_err(format!(
            "{name} indptr must be nonnegative, monotone, and within indices"
        )));
    }
    if indptr.last().copied() != Some(nnz) {
        return Err(PyValueError::new_err(format!(
            "{name} indptr must end at the indices length"
        )));
    }
    if let Some(&tag) = indices.iter().find(|&&tag| tag < 0 || tag as usize >= cols) {
        return Err(PyValueError::new_err(format!(
            "{name} tag {tag} is outside the valid range 0..{cols}"
        )));
    }
    Ok(())
}

fn parse_filter(dict: Option<PyFilter>, num_attrs: usize) -> PyResult<Filter> {
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
            .ok_or_else(|| {
                PyValueError::new_err(format!("invalid filter key '{}', expected 'attr_N'", key))
            })?;
        if j >= num_attrs {
            return Err(PyValueError::new_err(format!(
                "attribute index {} out of range (num_attributes={})",
                j, num_attrs
            )));
        }
        if values.is_empty() {
            return Err(PyValueError::new_err(format!(
                "filter for attr_{} has empty value list",
                j
            )));
        }
        constraints.push((j, values));
    }
    Ok(Filter::new(constraints))
}

/// Mutable builder and searchable PRISM index.
///
/// `scan_threshold` (default 20,000): eligible population at or below which
/// HIGH/MID queries score exactly; above it, search traverses the graph.
/// Unfiltered HIGH measures index size instead. `0` disables the route, so set
/// it and `multi_cell_scan_threshold` to 0 to force traversal. An index that
/// fits entirely stores a zero-edge graph, since no query can then need one.
///
/// `multi_cell_scan_threshold` (default 500,000): the same limit for a filter
/// spanning several cells. Not applied to unfiltered or one-cell queries.
///
/// `graph_expansion` (default 3, must exceed 0): traversal uses at most
/// `min(searched_population, ef * graph_expansion)` candidates; local-cell
/// search narrows back to `ef`. Expansion 1 is an ablation.
///
/// `cross_cell_exact_ranking_limit` (default 4,096, max 8,192): cell count at
/// or below which the greedy phase enumerates every foreign medoid; above it,
/// a bounded pool. "Exact" covers medoid enumeration, not neighbor ranking.
///
/// `build_seed`: seeds the randomized construction phases. The same data,
/// config and seed reproduce them within one numeric/backend environment.
///
/// The thresholds are workload-informed defaults rather than universal
/// crossovers, no setting guarantees a recall or latency ratio, and reproducible
/// builds are not cross-CPU byte identity.
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
        m_local = 48,
        m_greedy = 12,
        m_random = 4,
        t = 2,
        alpha = 1.0,
        vamana_alpha = 1.0,
        beam_width = 128,
        cross_cell_exact_ranking_limit = 4096,
        scan_threshold = 20000,
        multi_cell_scan_threshold = 500000,
        graph_expansion = 3,
        build_seed = 5787769093251747406,
        metric = "l2",
        sigma_high = 0.10,
        sigma_low = 0.001,
        beta = 3.0,
        epsilon = 0.2,
        binary_rerank = 0,
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
        cross_cell_exact_ranking_limit: usize,
        scan_threshold: usize,
        multi_cell_scan_threshold: usize,
        graph_expansion: usize,
        build_seed: u64,
        metric: &str,
        sigma_high: f32,
        sigma_low: f32,
        beta: f32,
        epsilon: f32,
        binary_rerank: usize,
    ) -> PyResult<Self> {
        if dim == 0 {
            return Err(PyValueError::new_err("dim must be greater than zero"));
        }
        let metric = parse_metric(metric)?;
        let config = PrismConfig {
            m_local,
            m_greedy,
            m_random,
            t,
            alpha,
            vamana_alpha,
            beam_width,
            cross_cell_exact_ranking_limit,
            scan_threshold,
            multi_cell_scan_threshold,
            graph_expansion,
            build_seed,
            metric,
            sigma_high,
            sigma_low,
            beta,
            epsilon,
            binary_rerank,
        };
        if let Err(error) = config.validate() {
            return Err(PyValueError::new_err(format!(
                "invalid PRISM configuration: {error}"
            )));
        }
        let mut pending_attrs = Vec::new();
        pending_attrs
            .try_reserve_exact(num_attributes)
            .map_err(|error| {
                PyValueError::new_err(format!("cannot allocate attribute schema: {error}"))
            })?;
        pending_attrs.resize_with(num_attributes, Vec::new);
        Ok(Self {
            pending_vectors: Vec::new(),
            pending_attrs,
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
                "expected dim={}, got {}",
                self.dim, vshape[1]
            )));
        }
        if ashape[1] != self.num_attrs {
            return Err(PyValueError::new_err(format!(
                "expected {} attributes, got {}",
                self.num_attrs, ashape[1]
            )));
        }
        if vshape[0] != ashape[0] {
            return Err(PyValueError::new_err(
                "vectors and attributes must have same number of rows",
            ));
        }
        let n = vshape[0];
        let v_array = vectors.as_array();
        let a_array = attributes.as_array();
        if v_array.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "vectors must contain only finite values",
            ));
        }

        let new_point_count = self.num_points.checked_add(n).ok_or_else(|| {
            PyValueError::new_err("total point count exceeds the platform integer space")
        })?;
        if new_point_count > u32::MAX as usize {
            return Err(PyValueError::new_err(
                "total point count exceeds PRISM's uint32 identifier space",
            ));
        }
        self.pending_vectors
            .try_reserve(v_array.len())
            .map_err(|error| {
                PyValueError::new_err(format!("cannot grow pending vector storage: {error}"))
            })?;
        for column in &mut self.pending_attrs {
            column.try_reserve(n).map_err(|error| {
                PyValueError::new_err(format!("cannot grow pending attribute storage: {error}"))
            })?;
        }

        // Logical row-major iteration supports sliced and Fortran-order arrays.
        self.pending_vectors.extend(v_array.iter().copied());
        for i in 0..n {
            for j in 0..self.num_attrs {
                self.pending_attrs[j].push(a_array[[i, j]]);
            }
        }
        self.num_points = new_point_count;
        Ok(())
    }

    /// Build the PRISM index from accumulated data.
    ///
    /// Deterministic input, schema, and configuration validation runs before
    /// staged buffers move into native construction. To avoid duplicating a
    /// potentially large vector allocation, a later allocation or topology-
    /// assembly failure consumes that staged batch and resets this object to an
    /// empty, unbuilt builder; add the data again before retrying.
    fn build(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.inner.is_some() {
            return Err(PyValueError::new_err("index already built"));
        }
        if self.num_points == 0 {
            return Err(PyValueError::new_err("no data added; call add() first"));
        }
        PrismIndex::validate_build_input(
            &self.pending_vectors,
            self.dim,
            &self.pending_attrs,
            &self.config,
        )
        .map_err(|error| {
            PyValueError::new_err(format!(
                "{error}; staged vectors and attributes were retained"
            ))
        })?;

        let vectors = std::mem::take(&mut self.pending_vectors);
        // Keep the outer schema allocation so a failed consuming build leaves a
        // valid empty builder, not one whose next `add()` indexes nothing.
        let attrs: Vec<Vec<u32>> = self.pending_attrs.iter_mut().map(std::mem::take).collect();
        let dim = self.dim;
        let config = self.config.clone();

        let index = match py.detach(move || {
            let store = PointStore::from_parts(vectors, dim, attrs)?;
            PrismIndex::build(store, config)
        }) {
            Ok(index) => index,
            Err(error) => {
                self.num_points = 0;
                return Err(PyValueError::new_err(format!(
                    "{error}; native construction failed after ownership transfer, so staged vectors and attributes were consumed and the builder was reset"
                )));
            }
        };
        self.inner = Some(index);
        Ok(())
    }

    /// Search for k nearest neighbors.
    ///
    /// Returns (ids, distances) as 1D numpy arrays. Graph and optional binary-
    /// prefilter ranking is approximate; an exact-scan route is exact. Underfill
    /// rescue guarantees the target count when enough records are eligible,
    /// not Recall@k. Use `search_exact` when exact ranking must be unconditional.
    #[pyo3(signature = (query, k = 10, ef = 200, filter = None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
        k: usize,
        ef: usize,
        filter: Option<PyFilter>,
    ) -> QueryOutput<'py> {
        let index = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("index not built; call build() first"))?;
        // Own the query before releasing the GIL; NumPy memory may not remain
        // borrowed while arbitrary Python code can run concurrently.
        let query_data: Vec<f32> = {
            let q = query.as_array();
            if q.len() != self.dim {
                return Err(PyValueError::new_err(format!(
                    "query dim {} != index dim {}",
                    q.len(),
                    self.dim
                )));
            }
            if q.iter().any(|value| !value.is_finite()) {
                return Err(PyValueError::new_err(
                    "query must contain only finite values",
                ));
            }
            q.iter().copied().collect()
        };
        drop(query);
        let f = parse_filter(filter, self.num_attrs)?;
        let outcome = py
            .detach(|| index.search(&query_data, &f, k, ef))
            .map_err(prism_error)?;
        let results = outcome.results;

        let ids: Vec<u32> = results
            .iter()
            .map(|result| original_id(index, result.id))
            .collect::<PyResult<_>>()?;
        let dists: Vec<f32> = results.iter().map(|r| r.dist).collect();

        Ok((PyArray1::from_vec(py, ids), PyArray1::from_vec(py, dists)))
    }

    /// Exact filtered top-k baseline over every eligible vector.
    #[pyo3(signature = (query, k = 10, filter = None))]
    fn search_exact<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
        k: usize,
        filter: Option<PyFilter>,
    ) -> QueryOutput<'py> {
        let index = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("index not built; call build() first"))?;
        let query_data: Vec<f32> = {
            let q = query.as_array();
            if q.len() != self.dim {
                return Err(PyValueError::new_err(format!(
                    "query dim {} != index dim {}",
                    q.len(),
                    self.dim
                )));
            }
            if q.iter().any(|value| !value.is_finite()) {
                return Err(PyValueError::new_err(
                    "query must contain only finite values",
                ));
            }
            q.iter().copied().collect()
        };
        drop(query);
        let filter = parse_filter(filter, self.num_attrs)?;
        let results = py
            .detach(|| index.search_exact(&query_data, &filter, k))
            .map_err(prism_error)?;
        let ids: Vec<u32> = results
            .iter()
            .map(|result| original_id(index, result.id))
            .collect::<PyResult<_>>()?;
        let distances: Vec<f32> = results.iter().map(|result| result.dist).collect();
        Ok((
            PyArray1::from_vec(py, ids),
            PyArray1::from_vec(py, distances),
        ))
    }

    /// Batch search: multiple queries in parallel.
    ///
    /// Returns (ids, distances) as 2D numpy arrays of shape (nq, k).
    /// Graph/binary-prefilter rows retain the same approximate-ranking/count
    /// distinction as `search`; exact-scan rows are exact.
    /// `filter`: single filter applied to all queries.
    /// `filters`: list of per-query filters (one dict per query, or None for no filter).
    #[pyo3(signature = (queries, k = 10, ef = 200, filter = None, filters = None))]
    fn batch_search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        ef: usize,
        filter: Option<PyFilter>,
        filters: Option<PerQueryFilters>,
    ) -> BatchOutput<'py> {
        let index = self
            .inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("index not built; call build() first"))?;
        let shape = queries.shape();
        let nq = shape[0];
        if shape[1] != self.dim {
            return Err(PyValueError::new_err(format!(
                "query dim {} != index dim {}",
                shape[1], self.dim
            )));
        }
        let output_len = checked_output_len(nq, k)?;

        // A per-query list takes priority over a single broadcast filter.
        let parsed_filters: Vec<Filter> = if let Some(filter_list) = filters {
            if filter_list.len() != nq {
                return Err(PyValueError::new_err(format!(
                    "filters length {} != query count {}",
                    filter_list.len(),
                    nq
                )));
            }
            filter_list
                .into_iter()
                .map(|f| parse_filter(f, self.num_attrs))
                .collect::<PyResult<Vec<_>>>()?
        } else {
            let f = parse_filter(filter, self.num_attrs)?;
            vec![f; nq]
        };

        // Copy query data and end the NumPy borrow before releasing the GIL.
        let q_data: Vec<f32> = {
            let q_array = queries.as_array();
            if q_array.iter().any(|value| !value.is_finite()) {
                return Err(PyValueError::new_err(
                    "queries must contain only finite values",
                ));
            }
            q_array.iter().copied().collect()
        };
        drop(queries);

        let outcomes = py
            .detach(|| index.batch_search(&q_data, &parsed_filters, nq, k, ef))
            .map_err(prism_error)?;
        let all_results = outcomes
            .into_iter()
            .map(|outcome| {
                outcome
                    .results
                    .iter()
                    .map(|result| Ok((original_id(index, result.id)?, result.dist)))
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;

        let mut ids = padded_vec(output_len, u32::MAX, "neighbor ID")?;
        let mut dists = padded_vec(output_len, f32::INFINITY, "distance")?;
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
    fn dim(&self) -> usize {
        self.dim
    }

    #[getter]
    fn num_points(&self) -> usize {
        self.num_points
    }

    #[getter]
    fn num_attributes(&self) -> usize {
        self.num_attrs
    }

    #[getter]
    fn is_built(&self) -> bool {
        self.inner.is_some()
    }
}

#[pyclass(name = "IvfIndex")]
struct PyIvfIndex {
    ivf: Option<IvfIndex>,
    centroids: ivf::VecStore,
    is_f32: bool,
    dim: usize,
    n: usize,
    n_clusters: usize,
    kmeans_iters: usize,
}

#[pymethods]
impl PyIvfIndex {
    #[new]
    #[pyo3(signature = (n_clusters = 4000, kmeans_iters = 5))]
    fn new(n_clusters: usize, kmeans_iters: usize) -> PyResult<Self> {
        if n_clusters == 0 || n_clusters > u16::MAX as usize + 1 {
            return Err(PyValueError::new_err(format!(
                "n_clusters must be between 1 and {}",
                u16::MAX as usize + 1
            )));
        }
        if kmeans_iters == 0 {
            return Err(PyValueError::new_err(
                "kmeans_iters must be greater than zero",
            ));
        }
        Ok(Self {
            ivf: None,
            centroids: ivf::VecStore::U8(Vec::new()),
            is_f32: false,
            dim: 0,
            n: 0,
            n_clusters,
            kmeans_iters,
        })
    }

    /// Build PRISM's filtered-IVF index from vectors and CSR tag metadata.
    ///
    /// vectors: numpy (n, dim) uint8 or float32
    /// meta_indptr: numpy (n+1,) int64, CSR row pointers
    /// meta_indices: numpy (nnz,) int32, CSR column indices (tag IDs)
    /// n_tags: vocabulary size
    fn build(
        &mut self,
        py: Python<'_>,
        vectors: &Bound<'_, PyAny>,
        meta_indptr: PyReadonlyArray1<i64>,
        meta_indices: PyReadonlyArray1<i32>,
        n_tags: usize,
    ) -> PyResult<()> {
        if self.ivf.is_some() {
            return Err(PyValueError::new_err("index already built"));
        }

        let (vec_store, n, dim) = if let Ok(arr) = vectors.extract::<PyReadonlyArray2<u8>>() {
            let shape = arr.shape();
            (
                ivf::VecStore::U8(arr.as_array().iter().copied().collect()),
                shape[0],
                shape[1],
            )
        } else if let Ok(arr) = vectors.extract::<PyReadonlyArray2<f32>>() {
            let shape = arr.shape();
            (
                ivf::VecStore::F32(arr.as_array().iter().copied().collect()),
                shape[0],
                shape[1],
            )
        } else {
            return Err(PyValueError::new_err(
                "vectors must be numpy array with dtype uint8 or float32",
            ));
        };

        if n == 0 {
            return Err(PyValueError::new_err(
                "vectors must contain at least one row",
            ));
        }
        if dim == 0 {
            return Err(PyValueError::new_err(
                "vector dimension must be greater than zero",
            ));
        }
        if self.n_clusters > n {
            return Err(PyValueError::new_err(format!(
                "n_clusters ({}) cannot exceed the point count ({n})",
                self.n_clusters
            )));
        }
        if let ivf::VecStore::F32(values) = &vec_store {
            if values.iter().any(|value| !value.is_finite()) {
                return Err(PyValueError::new_err(
                    "vectors must contain only finite values",
                ));
            }
        }
        let is_f32 = matches!(&vec_store, ivf::VecStore::F32(_));
        let indptr: Vec<i64> = meta_indptr.as_array().iter().copied().collect();
        let indices: Vec<i32> = meta_indices.as_array().iter().copied().collect();
        validate_csr(&indptr, &indices, n, n_tags, "metadata CSR")?;
        let n_clusters = self.n_clusters;
        let kmeans_iters = self.kmeans_iters;

        let (built_ivf, built_centroids) = py
            .detach(move || {
                let meta = SpMat::new(n, n_tags, indptr, indices)?;
                let (assignments, centroids) =
                    ivf::kmeans(&vec_store, n, dim, n_clusters, kmeans_iters)?;
                let index = IvfIndex::build(&vec_store, &meta, &assignments, n, dim, n_clusters)?;
                Ok((index, centroids))
            })
            .map_err(prism_error)?;

        self.ivf = Some(built_ivf);
        self.centroids = built_centroids;
        self.is_f32 = is_f32;
        self.dim = dim;
        self.n = n;
        Ok(())
    }

    /// Batch filtered search. Returns (nq, k) uint32 array of original IDs.
    ///
    /// queries: numpy (nq, dim) uint8 or float32
    /// filter_indptr: numpy (nq+1,) int64, CSR row pointers
    /// filter_indices: numpy (nnz,) int32, CSR column indices (required tag IDs)
    #[pyo3(signature = (queries, filter_indptr, filter_indices, k = 10, ef = 50, nprobe = 60, binary_rerank = 0))]
    #[allow(clippy::too_many_arguments)]
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'_, PyAny>,
        filter_indptr: PyReadonlyArray1<i64>,
        filter_indices: PyReadonlyArray1<i32>,
        k: usize,
        ef: usize,
        nprobe: usize,
        binary_rerank: usize,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let ivf_ref = self
            .ivf
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("index not built; call build() first"))?;

        let (query_store, nq, dim) = if let Ok(arr) = queries.extract::<PyReadonlyArray2<u8>>() {
            let shape = arr.shape();
            (
                ivf::VecStore::U8(arr.as_array().iter().copied().collect()),
                shape[0],
                shape[1],
            )
        } else if let Ok(arr) = queries.extract::<PyReadonlyArray2<f32>>() {
            let shape = arr.shape();
            (
                ivf::VecStore::F32(arr.as_array().iter().copied().collect()),
                shape[0],
                shape[1],
            )
        } else {
            return Err(PyValueError::new_err(
                "queries must be numpy array with dtype uint8 or float32",
            ));
        };

        let query_is_f32 = matches!(&query_store, ivf::VecStore::F32(_));
        if query_is_f32 != self.is_f32 {
            let expected = if self.is_f32 { "float32" } else { "uint8" };
            return Err(PyValueError::new_err(format!(
                "query dtype must match index dtype ({expected})"
            )));
        }
        if let ivf::VecStore::F32(values) = &query_store {
            if values.iter().any(|value| !value.is_finite()) {
                return Err(PyValueError::new_err(
                    "queries must contain only finite values",
                ));
            }
        }
        if dim != self.dim {
            return Err(PyValueError::new_err(format!(
                "query dim {} != index dim {}",
                dim, self.dim
            )));
        }
        let output_len = checked_output_len(nq, k)?;

        let indptr: Vec<i64> = filter_indptr.as_array().iter().copied().collect();
        let indices: Vec<i32> = filter_indices.as_array().iter().copied().collect();
        validate_csr(&indptr, &indices, nq, ivf_ref.n_tags(), "filter CSR")?;
        if k > 0 && nprobe == 0 {
            return Err(PyValueError::new_err(
                "nprobe must be greater than zero when k is nonzero",
            ));
        }
        if k == 0 {
            let empty = Array2::from_shape_vec((nq, 0), Vec::new()).unwrap();
            return Ok(PyArray2::from_owned_array(py, empty));
        }

        let results = py
            .detach(|| {
                let query_tags: Vec<Vec<usize>> = (0..nq)
                    .map(|qi| {
                        let start = indptr[qi] as usize;
                        let end = indptr[qi + 1] as usize;
                        let mut tags: Vec<usize> =
                            indices[start..end].iter().map(|&t| t as usize).collect();
                        tags.sort_unstable();
                        tags.dedup();
                        tags
                    })
                    .collect();

                let query_top_clusters: Vec<Vec<usize>> = (0..nq)
                    .into_par_iter()
                    .map(|qi| {
                        let tags = &query_tags[qi];

                        let candidates: std::borrow::Cow<[u16]> = if tags.is_empty() {
                            std::borrow::Cow::Owned(
                                (0..ivf_ref.n_clusters())
                                    .map(|cluster| cluster as u16)
                                    .collect(),
                            )
                        } else {
                            let first = tags[0];
                            let mut matching = ivf_ref
                                .clusters_for_tag(first)
                                .map(<[u16]>::to_vec)
                                .unwrap_or_default();
                            for &tag in &tags[1..] {
                                let Some(tag_clusters) = ivf_ref.clusters_for_tag(tag) else {
                                    matching.clear();
                                    break;
                                };
                                matching = ivf::sorted_intersect_u16(&matching, tag_clusters);
                                if matching.is_empty() {
                                    break;
                                }
                            }
                            std::borrow::Cow::Owned(matching)
                        };

                        if candidates.is_empty() {
                            return vec![];
                        }

                        let mut cluster_dists: Vec<(usize, u64)> = candidates
                            .iter()
                            .map(|&ci| {
                                let ci = ci as usize;
                                let dist = match (&query_store, &self.centroids) {
                                    (ivf::VecStore::U8(qd), ivf::VecStore::U8(cd)) => {
                                        distance::l2_sq8(
                                            &qd[qi * dim..(qi + 1) * dim],
                                            &cd[ci * dim..(ci + 1) * dim],
                                        )
                                    }
                                    (ivf::VecStore::F32(qd), ivf::VecStore::F32(cd)) => {
                                        distance::l2_squared(
                                            &qd[qi * dim..(qi + 1) * dim],
                                            &cd[ci * dim..(ci + 1) * dim],
                                        )
                                        .to_bits() as u64
                                    }
                                    _ => unreachable!("mismatched query/centroid types"),
                                };
                                (ci, dist)
                            })
                            .collect();

                        let np = nprobe.min(cluster_dists.len());
                        cluster_dists.select_nth_unstable_by_key(np - 1, |&(_, d)| d);
                        cluster_dists.truncate(np);
                        cluster_dists.sort_unstable_by_key(|&(_, d)| d);
                        cluster_dists.iter().map(|&(ci, _)| ci).collect()
                    })
                    .collect();

                let qs = match &query_store {
                    ivf::VecStore::U8(data) => ivf::QueryStore::U8(data),
                    ivf::VecStore::F32(data) => ivf::QueryStore::F32(data),
                };

                ivf_ref.batch_search_mqcb(
                    &qs,
                    nq,
                    &query_tags,
                    &query_top_clusters,
                    k,
                    ef,
                    nprobe,
                    binary_rerank,
                )
            })
            .map_err(prism_error)?;

        let mut ids = padded_vec(output_len, u32::MAX, "IVF neighbor ID")?;
        for (i, row) in results.iter().enumerate() {
            for (j, &id) in row.iter().enumerate().take(k) {
                ids[i * k + j] = id;
            }
        }

        let arr = Array2::from_shape_vec((nq, k), ids).unwrap();
        Ok(PyArray2::from_owned_array(py, arr))
    }

    #[getter]
    fn dim(&self) -> usize {
        self.dim
    }

    #[getter]
    fn num_points(&self) -> usize {
        self.n
    }

    #[getter]
    fn is_built(&self) -> bool {
        self.ivf.is_some()
    }
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
        assert!(matches!(parse_metric("cosine").unwrap(), Metric::Cosine));
        assert!(parse_metric("manhattan").is_err());
    }
}
