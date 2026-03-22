//! Verification tests for PRISM paper lemmas and theorems.

use crate::construct::{
    PrismConfig, PrismIndex,
    build_random_overlay, select_cross_neighbors, t_subsets, add_tuples,
};
use crate::filter::Filter;
use crate::graph::AdjBuilder;
use crate::point::PointStore;

use rand::prelude::*;
use std::collections::{HashSet, VecDeque};

/// Build a random d-regular graph via d/2 random permutations (Friedman model).
fn random_regular_graph(n: usize, d: usize, rng: &mut impl Rng) -> Vec<Vec<u32>> {
    assert!(d >= 2 && d.is_multiple_of(2));
    let half = d / 2;
    let mut adj: Vec<HashSet<u32>> = vec![HashSet::new(); n];

    for _ in 0..half {
        let mut perm: Vec<u32> = (0..n as u32).collect();
        perm.shuffle(rng);
        for (i, &j) in perm.iter().enumerate() {
            let i = i as u32;
            if i != j {
                adj[i as usize].insert(j);
                adj[j as usize].insert(i);
            }
        }
    }

    adj.into_iter()
        .map(|s| {
            let mut v: Vec<u32> = s.into_iter().collect();
            v.sort_unstable();
            v
        })
        .collect()
}

/// Sparse matrix-vector multiply: y = A x.
fn spmv(adj: &[Vec<u32>], x: &[f64], y: &mut [f64]) {
    for (i, neighbors) in adj.iter().enumerate() {
        y[i] = neighbors.iter().map(|&j| x[j as usize]).sum();
    }
}

/// Estimate |lambda_2| via power iteration with deflation against the all-ones eigenvector.
fn spectral_gap(adj: &[Vec<u32>], iters: usize) -> f64 {
    let n = adj.len();
    if n < 2 {
        return 0.0;
    }

    let mut rng = rand::thread_rng();
    let mut x: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() - 0.5).collect();

    // Orthogonalize against all-ones/sqrt(n) and normalize
    let mean = x.iter().sum::<f64>() / n as f64;
    for xi in &mut x {
        *xi -= mean;
    }
    let norm = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
    if norm < 1e-15 {
        return 0.0;
    }
    for xi in &mut x {
        *xi /= norm;
    }

    let mut y = vec![0.0f64; n];
    let mut lambda = 0.0;

    for _ in 0..iters {
        spmv(adj, &x, &mut y);
        // Deflate: remove component along all-ones
        let mean = y.iter().sum::<f64>() / n as f64;
        for yi in &mut y {
            *yi -= mean;
        }
        // Rayleigh quotient
        lambda = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum::<f64>();
        // Normalize
        let norm = y.iter().map(|yi| yi * yi).sum::<f64>().sqrt();
        if norm < 1e-15 {
            break;
        }
        for (xi, yi) in x.iter_mut().zip(y.iter()) {
            *xi = yi / norm;
        }
    }

    lambda.abs()
}

/// Count undirected edges within a subset of vertices.
fn count_induced_edges(adj: &[Vec<u32>], subset: &HashSet<u32>) -> usize {
    let mut count = 0;
    for &u in subset {
        for &v in &adj[u as usize] {
            if v > u && subset.contains(&v) {
                count += 1;
            }
        }
    }
    count
}

/// BFS connected components within a subset. Returns sizes in descending order.
fn connected_components(adj: &[Vec<u32>], subset: &HashSet<u32>) -> Vec<usize> {
    let mut visited = HashSet::new();
    let mut sizes = Vec::new();

    for &start in subset {
        if visited.contains(&start) {
            continue;
        }
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);
        let mut size = 0;
        while let Some(u) = queue.pop_front() {
            size += 1;
            for &v in &adj[u as usize] {
                if subset.contains(&v) && visited.insert(v) {
                    queue.push_back(v);
                }
            }
        }
        sizes.push(size);
    }

    sizes.sort_unstable_by(|a, b| b.cmp(a));
    sizes
}

/// Count distinct t-tuples covered by a set of selected points.
fn coverage_ft(store: &PointStore, selected: &[u32], subsets: &[Vec<usize>]) -> usize {
    let mut covered = HashSet::new();
    for &q in selected {
        add_tuples(store, q, &mut covered, subsets);
    }
    covered.len()
}

/// Brute-force optimal t-tuple coverage over all C(n, budget) subsets.
fn optimal_coverage(
    store: &PointStore,
    cands: &[u32],
    budget: usize,
    subsets: &[Vec<usize>],
) -> usize {
    let mut best = 0;
    let mut combo = vec![0usize; budget];
    enumerate_subsets(store, cands, subsets, &mut combo, 0, 0, &mut best);
    best
}

fn enumerate_subsets(
    store: &PointStore,
    cands: &[u32],
    subsets: &[Vec<usize>],
    combo: &mut [usize],
    depth: usize,
    start: usize,
    best: &mut usize,
) {
    if depth == combo.len() {
        let selected: Vec<u32> = combo.iter().map(|&i| cands[i]).collect();
        let cov = coverage_ft(store, &selected, subsets);
        if cov > *best {
            *best = cov;
        }
        return;
    }
    let remaining = combo.len() - depth;
    if start + remaining > cands.len() {
        return;
    }
    for i in start..=(cands.len() - remaining) {
        combo[depth] = i;
        enumerate_subsets(store, cands, subsets, combo, depth + 1, i + 1, best);
    }
}

/// Convert CSR graph to adjacency lists.
fn graph_to_adj(graph: &crate::graph::Graph) -> Vec<Vec<u32>> {
    (0..graph.n)
        .map(|i| graph.neighbors(i as u32).to_vec())
        .collect()
}

/// Build a PointStore with grid-strided synthetic attributes and random vectors.
/// Attribute j cycles with stride = product of cardinalities[0..j].
fn make_test_store(n: usize, dim: usize, cardinalities: &[usize]) -> PointStore {
    let mut rng = rand::thread_rng();
    let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();
    let attrs: Vec<Vec<u32>> = cardinalities
        .iter()
        .enumerate()
        .map(|(j, &c)| {
            let stride: usize = cardinalities[..j].iter().product::<usize>().max(1);
            (0..n).map(|i| ((i / stride) % c) as u32).collect()
        })
        .collect();
    PointStore::from_parts(vectors, dim, attrs)
}

/// Bridge score formula (Definition 5.1).
fn bridge_formula(matching: usize, total: usize, dist: f32, radius: f32) -> f32 {
    let neighbor_ratio = matching as f32 / total as f32;
    let proximity = 1.0 / (1.0 + dist / radius);
    neighbor_ratio * proximity
}

/// Lemma 2.7: Friedman spectral gap for random d-regular graphs.
/// Claim: |lambda_2| <= 2*sqrt(d-1) + epsilon w.h.p.
#[test]
fn lemma_2_7_friedman_spectral_gap() {
    let n = 500;
    let mut rng = rand::thread_rng();

    // d=4: bound = 2*sqrt(3) + 0.5 ~ 3.96
    let d4_bound = 2.0 * 3.0f64.sqrt() + 0.5;
    let mut d4_pass = 0;
    for _ in 0..10 {
        let adj = random_regular_graph(n, 4, &mut rng);
        let gap = spectral_gap(&adj, 300);
        if gap <= d4_bound {
            d4_pass += 1;
        }
    }
    assert!(d4_pass >= 8, "d=4: only {d4_pass}/10 trials <= {d4_bound:.2}");

    // d=8: bound = 2*sqrt(7) + 0.5 ~ 5.79
    let d8_bound = 2.0 * 7.0f64.sqrt() + 0.5;
    let mut d8_pass = 0;
    for _ in 0..10 {
        let adj = random_regular_graph(n, 8, &mut rng);
        let gap = spectral_gap(&adj, 300);
        if gap <= d8_bound {
            d8_pass += 1;
        }
    }
    assert!(d8_pass >= 8, "d=8: only {d8_pass}/10 trials <= {d8_bound:.2}");
}

/// Lemma 2.8(i): Expander Mixing Lemma edge count lower bound.
/// For subset S with |S|=sigma*n: e(G[S]) >= sigma*n/2 * (sigma*d - lambda).
#[test]
fn lemma_2_8i_edge_count() {
    let n = 500;
    let d = 8;
    let mut rng = rand::thread_rng();
    let adj = random_regular_graph(n, d, &mut rng);
    let lambda = spectral_gap(&adj, 300);

    for &sigma in &[0.5, 0.3, 0.2, 0.1] {
        if sigma * d as f64 <= lambda {
            continue; // bound requires sigma*d > lambda
        }
        let s_size = (sigma * n as f64) as usize;
        let lower_bound = s_size as f64 / 2.0 * (sigma * d as f64 - lambda);

        for _ in 0..5 {
            let mut perm: Vec<u32> = (0..n as u32).collect();
            perm.shuffle(&mut rng);
            let subset: HashSet<u32> = perm[..s_size].iter().copied().collect();
            let edges = count_induced_edges(&adj, &subset);
            assert!(
                edges as f64 >= lower_bound - 1.0,
                "sigma={sigma}: edges={edges}, bound={lower_bound:.1}, lambda={lambda:.2}"
            );
        }
    }
}

/// Lemma 2.8(ii): Giant component in induced subgraph.
/// When sigma*d > (3/sqrt(2))*lambda, G[S] has component >= sigma*n - 2*lambda^2*n/(sigma*d^2).
#[test]
fn lemma_2_8ii_giant_component() {
    let n = 1000;
    let d = 8;
    let mut rng = rand::thread_rng();
    let adj = random_regular_graph(n, d, &mut rng);
    // Use conservative lambda: inflate by 10% to account for power iteration underestimate
    let lambda_raw = spectral_gap(&adj, 300);
    let lambda = lambda_raw * 1.1;

    let threshold = 3.0 * lambda / (std::f64::consts::SQRT_2 * d as f64);

    for &sigma in &[0.5, 0.3, 0.2] {
        if sigma <= threshold {
            continue;
        }
        let s_size = (sigma * n as f64) as usize;
        let lower_bound =
            sigma * n as f64 - 2.0 * lambda * lambda * n as f64 / (sigma * (d as f64).powi(2));

        for _ in 0..5 {
            let mut perm: Vec<u32> = (0..n as u32).collect();
            perm.shuffle(&mut rng);
            let subset: HashSet<u32> = perm[..s_size].iter().copied().collect();
            let components = connected_components(&adj, &subset);
            let largest = *components.first().unwrap_or(&0);
            assert!(
                largest as f64 >= lower_bound,
                "sigma={sigma}: largest={largest}, bound={lower_bound:.1}, lambda={lambda:.2}"
            );
        }
    }
}

/// Theorem 6.3: PRISM filtered subgraph has a giant component (>50% of filtered points).
#[test]
fn theorem_6_3_filter_resilience() {
    let n = 500;
    let dim = 8;
    let store = make_test_store(n, dim, &[5, 5, 5]);
    let config = PrismConfig {
        m_local: 16,
        m_greedy: 12,
        m_random: 4,
        t: 2,
        alpha: 1.0,
        beam_width: 120,
        ..Default::default()
    };
    let index = PrismIndex::build(store, config);
    let adj = graph_to_adj(&index.graph);

    for val in 0..5u32 {
        let filter = Filter::eq(0, val);
        let filtered_pts: HashSet<u32> = (0..n as u32)
            .filter(|&p| filter.matches(&index.store, p))
            .collect();
        let n_f = filtered_pts.len();
        let components = connected_components(&adj, &filtered_pts);
        let largest = *components.first().unwrap_or(&0);
        assert!(
            largest as f64 >= 0.5 * n_f as f64,
            "val={val}: largest={largest}, filtered={n_f} (need >50%)"
        );
    }
}

/// Theorem 6.6 Case 1 (corrected): with t=1, every attribute value has >= 1 survivor.
/// Original proof claimed >= floor(M_g/s_j) via pigeonhole, but pigeonhole only guarantees
/// max bin >= ceil(n/k), not min bin >= floor(n/k). Corrected bound: >= 1.
/// With t=2, the greedy can produce 0 survivors for some values (2-tuple coverage
/// doesn't imply 1-tuple coverage).
#[test]
fn theorem_6_6_case1_single_attr_survival() {
    let n = 200;
    let dim = 8;
    let store = make_test_store(n, dim, &[5, 5, 5]);
    let k = 3;
    let m_g = 12;

    // Test with t=1 (covers all 1-tuples, guaranteeing every value is represented)
    let config_t1 = PrismConfig {
        m_greedy: m_g,
        t: 1,
        alpha: 0.0,
        ..Default::default()
    };
    let subsets_t1 = t_subsets(k, 1);
    let mut rng = rand::thread_rng();

    for _ in 0..50 {
        let p = rng.gen_range(0..n as u32);
        let p_vec = store.vector(p);
        let mut candidates: Vec<(u32, f32)> = (0..n as u32)
            .filter(|&q| q != p)
            .map(|q| (q, crate::distance::l2_squared(p_vec, store.vector(q))))
            .collect();
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let selected = select_cross_neighbors(&store, &candidates, &config_t1, &subsets_t1);
        assert_eq!(selected.len(), m_g);

        for j in 0..k {
            for v in 0..5u32 {
                let survivors = selected.iter().filter(|&&q| store.attr(q, j) == v).count();
                assert!(
                    survivors >= 1,
                    "p={p}, attr={j}, val={v}: 0 survivors with t=1"
                );
            }
        }
    }

    // Also test with t=2: verify the paper's original floor(M_g/s_j) >= 2 does NOT hold.
    // This documents the paper bug — the greedy can produce only 1 per value.
    let config_t2 = PrismConfig {
        m_greedy: m_g,
        t: 2,
        alpha: 0.0,
        ..Default::default()
    };
    let subsets_t2 = t_subsets(k, 2);
    let mut found_violation = false;
    for trial in 0..200 {
        let p = (trial % n) as u32;
        let p_vec = store.vector(p);
        let mut candidates: Vec<(u32, f32)> = (0..n as u32)
            .filter(|&q| q != p)
            .map(|q| (q, crate::distance::l2_squared(p_vec, store.vector(q))))
            .collect();
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let selected = select_cross_neighbors(&store, &candidates, &config_t2, &subsets_t2);
        for j in 0..k {
            for v in 0..5u32 {
                let count = selected.iter().filter(|&&q| store.attr(q, j) == v).count();
                if count < 2 {
                    found_violation = true;
                }
            }
        }
        if found_violation {
            break;
        }
    }
    assert!(
        found_violation,
        "Expected to find a case where t=2 gives < floor(M_g/s_j)=2 survivors"
    );
}

/// Theorem 6.6 Case 2: With M_g >= CAN bound, every strength-t filter has >= 1 survivor.
#[test]
fn theorem_6_6_case2_covering_array() {
    let n = 300;
    let dim = 8;
    let store = make_test_store(n, dim, &[3, 3, 3]);
    let k = 3;
    let t = 2;
    let m_g = 12;
    let config = PrismConfig {
        m_greedy: m_g,
        t,
        alpha: 0.0,
        ..Default::default()
    };
    let subsets = t_subsets(k, t);

    // All strength-2 filters: C(3,2) pairs x 3x3 value combos = 27
    let mut all_filters = Vec::new();
    for j1 in 0..k {
        for j2 in (j1 + 1)..k {
            for v1 in 0..3u32 {
                for v2 in 0..3u32 {
                    all_filters.push(Filter::new(vec![(j1, vec![v1]), (j2, vec![v2])]));
                }
            }
        }
    }

    let mut rng = rand::thread_rng();
    for _ in 0..50 {
        let p = rng.gen_range(0..n as u32);
        let p_vec = store.vector(p);
        let mut candidates: Vec<(u32, f32)> = (0..n as u32)
            .filter(|&q| q != p)
            .map(|q| (q, crate::distance::l2_squared(p_vec, store.vector(q))))
            .collect();
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let selected = select_cross_neighbors(&store, &candidates, &config, &subsets);

        for f in &all_filters {
            let survivors = selected.iter().filter(|&&q| f.matches(&store, q)).count();
            assert!(
                survivors >= 1,
                "p={p}: filter {:?} has 0 survivors among {selected:?}",
                f.constraints()
            );
        }
    }
}

/// Theorem 4.1: Greedy coverage >= (1-1/e) * optimal.
#[test]
fn theorem_4_1_coverage_guarantee() {
    let n = 50;
    let dim = 4;
    let store = make_test_store(n, dim, &[3, 3, 3]);
    let k = 3;
    let t = 2;
    let m_g = 6;
    let config = PrismConfig {
        m_greedy: m_g,
        t,
        alpha: 0.0,
        ..Default::default()
    };
    let subsets = t_subsets(k, t);
    let ratio_bound = 1.0 - 1.0 / std::f64::consts::E; // ~ 0.632
    let mut rng = rand::thread_rng();

    for _ in 0..20 {
        let p = rng.gen_range(0..n as u32);
        let p_vec = store.vector(p);

        let mut cand_ids: Vec<u32> = (0..n as u32).filter(|&q| q != p).collect();
        cand_ids.shuffle(&mut rng);
        cand_ids.truncate(20);

        let mut candidates: Vec<(u32, f32)> = cand_ids
            .iter()
            .map(|&q| (q, crate::distance::l2_squared(p_vec, store.vector(q))))
            .collect();
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let selected = select_cross_neighbors(&store, &candidates, &config, &subsets);
        let greedy_cov = coverage_ft(&store, &selected, &subsets);
        let opt_cov = optimal_coverage(&store, &cand_ids, m_g, &subsets);

        if opt_cov > 0 {
            let ratio = greedy_cov as f64 / opt_cov as f64;
            assert!(
                ratio >= ratio_bound - 0.01,
                "greedy={greedy_cov}, optimal={opt_cov}, ratio={ratio:.3}"
            );
        }
    }
}

/// Lemma 3.6: f_t (tuple coverage) is monotone and submodular.
#[test]
fn lemma_3_6_submodularity() {
    let n = 30;
    let dim = 4;
    let store = make_test_store(n, dim, &[3, 4, 3]);
    let k = 3;
    let t = 2;
    let subsets = t_subsets(k, t);

    let mut rng = rand::thread_rng();
    let all_ids: Vec<u32> = (0..n as u32).collect();

    for trial in 0..100 {
        let b_size = rng.gen_range(2..n.min(10));
        let a_size = rng.gen_range(1..b_size);
        let mut perm = all_ids.clone();
        perm.shuffle(&mut rng);
        let b: Vec<u32> = perm[..b_size].to_vec();
        let a: Vec<u32> = b[..a_size].to_vec();
        let q = perm[b_size];

        let f_a = coverage_ft(&store, &a, &subsets);
        let f_b = coverage_ft(&store, &b, &subsets);

        let mut a_q = a.clone();
        a_q.push(q);
        let mut b_q = b.clone();
        b_q.push(q);

        let f_a_q = coverage_ft(&store, &a_q, &subsets);
        let f_b_q = coverage_ft(&store, &b_q, &subsets);

        // Monotonicity
        assert!(f_a_q >= f_a, "trial {trial}: monotonicity failed");

        // Submodularity: marginal gain decreases with larger set
        let gain_a = f_a_q - f_a;
        let gain_b = f_b_q - f_b;
        assert!(
            gain_a >= gain_b,
            "trial {trial}: submodularity failed (gain_a={gain_a}, gain_b={gain_b})"
        );
    }
}

/// Definition 5.1: Bridge score formula verification.
#[test]
fn definition_5_1_bridge_score() {
    // More matching neighbors -> higher score
    assert!(bridge_formula(8, 10, 1.0, 5.0) > bridge_formula(2, 10, 1.0, 5.0));

    // Closer distance -> higher score
    assert!(bridge_formula(5, 10, 1.0, 5.0) > bridge_formula(5, 10, 10.0, 5.0));

    // Exact: (3/6) * (1/(1 + 2/4)) = 0.5 * 2/3 = 1/3
    let score = bridge_formula(3, 6, 2.0, 4.0);
    assert!((score - 1.0 / 3.0).abs() < 1e-6, "expected 1/3, got {score}");

    // Zero matching -> zero score
    assert_eq!(bridge_formula(0, 10, 1.0, 5.0), 0.0);

    // Zero distance -> proximity = 1.0
    let score = bridge_formula(5, 10, 0.0, 5.0);
    assert!((score - 0.5).abs() < 1e-6, "expected 0.5, got {score}");
}

/// Graph structure invariants: no self-loops, valid IDs, sorted neighbors, min degree.
#[test]
fn graph_structure_invariants() {
    let n = 200;
    let dim = 8;
    let store = make_test_store(n, dim, &[4, 4, 4]);
    let config = PrismConfig {
        m_local: 16,
        m_greedy: 12,
        m_random: 4,
        t: 2,
        alpha: 1.0,
        beam_width: 120,
        ..Default::default()
    };
    let index = PrismIndex::build(store, config);

    for i in 0..n as u32 {
        let neighbors = index.graph.neighbors(i);
        assert!(!neighbors.contains(&i), "self-loop at node {i}");
        for &j in neighbors {
            assert!(j < n as u32, "invalid neighbor {j} for node {i}");
        }
        for w in neighbors.windows(2) {
            assert!(w[0] < w[1], "neighbors not sorted/deduped at {i}");
        }
        assert!(
            index.graph.degree(i) >= 2,
            "degree too low at {i}: {}",
            index.graph.degree(i)
        );
    }

    let avg_deg = index.graph.num_edges() as f64 / n as f64;
    assert!(avg_deg >= 10.0, "avg degree {avg_deg:.1} too low");
    assert!(avg_deg <= 100.0, "avg degree {avg_deg:.1} too high");
}

/// Random overlay produces approximately d-regular graph.
#[test]
fn random_overlay_regularity() {
    let n = 500;
    let d = 8;
    let mut adj = AdjBuilder::new(n);
    build_random_overlay(n, d, &mut adj);
    let graph = adj.build();

    for i in 0..n as u32 {
        let deg = graph.degree(i);
        assert!(
            deg >= d - 3 && deg <= d + 3,
            "node {i}: degree {deg}, expected ~{d}"
        );
        assert!(!graph.neighbors(i).contains(&i), "self-loop at {i}");
    }
}

/// Proposition 3.2: filter_cells returns exactly the compatible leaves.
#[test]
fn proposition_3_2_filter_pruning() {
    let n = 200;
    let dim = 4;
    let store = make_test_store(n, dim, &[4, 3, 5]);
    let config = PrismConfig {
        m_local: 8,
        m_greedy: 6,
        m_random: 4,
        t: 1,
        alpha: 0.0,
        beam_width: 30,
        ..Default::default()
    };
    let index = PrismIndex::build(store, config);

    let filters = vec![
        Filter::eq(0, 2),
        Filter::eq(1, 0),
        Filter::eq(2, 4),
        Filter::new(vec![(0, vec![0]), (1, vec![1])]),
    ];

    for filter in &filters {
        let cell_indices = index.tree.filter_cells(filter.constraints());
        let cell_points: HashSet<u32> = cell_indices
            .iter()
            .flat_map(|&ci| index.tree.cells[ci].point_ids.iter().copied())
            .collect();

        // No false positives
        for &p in &cell_points {
            assert!(
                filter.matches(&index.store, p),
                "false positive: point {p} in cells but fails filter"
            );
        }

        // No false negatives
        for p in 0..n as u32 {
            if filter.matches(&index.store, p) {
                assert!(
                    cell_points.contains(&p),
                    "false negative: point {p} passes filter but not in cells"
                );
            }
        }
    }
}
