use prism_ann::construct::{PrismConfig, PrismIndex};
use prism_ann::filter::Filter;
use prism_ann::io;

use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: prism-bench <dataset-dir> [--k N] [--ef N] [--n N] [--m-local N]");
        eprintln!("  dataset-dir: path to directory containing .fvecs files");
        eprintln!("  --k: number of nearest neighbors (default: 10)");
        eprintln!("  --ef: search beam width (default: 64)");
        eprintln!("  --n: use first N base vectors (default: all)");
        eprintln!("  --m-local: local graph degree (default: 16)");
        eprintln!("  --vamana-alpha: Vamana pruning alpha (default: 1.0, e.g. 1.2)");
        eprintln!();
        eprintln!("If no dataset available, runs a synthetic benchmark.");
        run_synthetic_benchmark();
        return;
    }

    let dataset_dir = PathBuf::from(&args[1]);
    let k = parse_arg(&args, "--k", 10);
    let ef = parse_arg(&args, "--ef", 200);
    let max_n = parse_arg(&args, "--n", 0); // 0 = use all
    let m_local = parse_arg(&args, "--m-local", 16);
    let vamana_alpha = parse_float_arg(&args, "--vamana-alpha", 1.0);

    if dataset_dir.join("sift_base.fvecs").exists() {
        run_sift1m_benchmark(&dataset_dir, k, ef, max_n, m_local, vamana_alpha);
    } else {
        eprintln!("Dataset files not found in {:?}", dataset_dir);
        eprintln!("Expected: sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs");
        eprintln!("Download from: http://corpus-texmex.irisa.fr/");
        eprintln!();
        eprintln!("Running synthetic benchmark instead...");
        run_synthetic_benchmark();
    }
}

fn run_sift1m_benchmark(dir: &Path, k: usize, ef: usize, max_n: usize, m_local: usize, vamana_alpha: f32) {
    println!("=== PRISM Benchmark: SIFT1M ({} threads) ===", rayon::current_num_threads());
    println!();

    // Load base vectors
    print!("Loading base vectors...");
    let t = Instant::now();
    let (mut base_vecs, dim) = io::load_fvecs(&dir.join("sift_base.fvecs")).unwrap();
    let mut n = base_vecs.len() / dim;
    if max_n > 0 && max_n < n {
        base_vecs.truncate(max_n * dim);
        n = max_n;
    }
    println!(" {n} vectors, {dim}d ({:.1}s)", t.elapsed().as_secs_f64());

    // Load query vectors
    print!("Loading query vectors...");
    let t = Instant::now();
    let (query_vecs, _) = io::load_fvecs(&dir.join("sift_query.fvecs")).unwrap();
    let nq = query_vecs.len() / dim;
    println!(" {nq} queries ({:.1}s)", t.elapsed().as_secs_f64());

    // Load ground truth (only valid for full dataset)
    print!("Loading ground truth...");
    let t = Instant::now();
    let gt = io::load_ivecs(&dir.join("sift_groundtruth.ivecs")).unwrap();
    println!(" {} entries ({:.1}s)", gt.len(), t.elapsed().as_secs_f64());

    // Build index with synthetic attributes
    let cardinalities = &[10, 10, 10]; // k=3 attributes, 10 values each
    println!("Building PRISM index ({n} pts, 3 attrs × 10 values, M_local={m_local}, α_v={vamana_alpha})...");
    let t = Instant::now();
    let store = io::build_store_with_synthetic_attrs(base_vecs, dim, cardinalities);
    let config = PrismConfig { m_local, vamana_alpha, ..PrismConfig::default() };
    let index = PrismIndex::build(store, config);
    let build_time = t.elapsed().as_secs_f64();
    println!("  Build time: {build_time:.1}s");
    println!("  Cells: {}", index.tree.cells.len());
    println!("  Edges: {}", index.graph.num_edges());

    // Graph degree statistics
    let degrees: Vec<usize> = (0..n).map(|i| index.graph.degree(i as u32)).collect();
    let avg_deg = degrees.iter().sum::<usize>() as f64 / n as f64;
    let min_deg = *degrees.iter().min().unwrap_or(&0);
    let max_deg = *degrees.iter().max().unwrap_or(&0);
    println!("  Avg degree: {avg_deg:.1}, min: {min_deg}, max: {max_deg}");

    // Local (intra-cell) edge density
    let mut total_local = 0usize;
    for cell in &index.tree.cells {
        let pts = &cell.point_ids;
        let cell_set: std::collections::HashSet<u32> = pts.iter().copied().collect();
        for &p in pts {
            let local = index.graph.neighbors(p).iter()
                .filter(|&&nb| cell_set.contains(&nb)).count();
            total_local += local;
        }
    }
    println!("  Avg local edges/node: {:.1}", total_local as f64 / n as f64);
    println!();

    // Benchmark: no filter
    println!("--- No Filter (ef={ef}) ---");
    let use_gt = max_n == 0; // ground truth only valid for full dataset
    benchmark_queries(&index, &query_vecs, dim, if use_gt { &gt } else { &[] }, k, ef, &Filter::none());

    // Benchmark: single-attribute filter (σ ≈ 10%)
    println!("--- Single Attr Filter (σ≈10%, ef={ef}) ---");
    benchmark_queries(&index, &query_vecs, dim, &[], k, ef, &Filter::eq(0, 0));

    // Benchmark: two-attribute filter (σ ≈ 1%)
    println!("--- Two Attr Filter (σ≈1%, ef={ef}) ---");
    let filter_1 = Filter::new(vec![(0, vec![0]), (1, vec![0])]);
    benchmark_queries(&index, &query_vecs, dim, &[], k, ef, &filter_1);
}

fn benchmark_queries(
    index: &PrismIndex,
    query_vecs: &[f32],
    dim: usize,
    gt: &[Vec<u32>],
    k: usize,
    ef: usize,
    filter: &Filter,
) {
    use prism_ann::distance;
    let nq = query_vecs.len() / dim;
    let n = index.store.len;
    let metric = index.config.metric;

    // Phase 1: Timed search (QPS measurement, parallel across queries)
    let t = Instant::now();
    let all_results: Vec<_> = (0..nq)
        .into_par_iter()
        .map(|i| {
            let query = &query_vecs[i * dim..(i + 1) * dim];
            index.search(query, filter, k, ef)
        })
        .collect();
    let elapsed = t.elapsed().as_secs_f64();
    let qps = nq as f64 / elapsed;

    let total_results: usize = all_results.iter().map(|r| r.len()).sum();

    // Phase 2: Recall computation (not timed)
    let recall_queries = nq.min(1000); // limit brute-force recall to 1000 queries
    let mut total_recall = 0.0f64;
    let compute_recall = !gt.is_empty() || filter.strength() > 0;

    if compute_recall {
        // Build original→internal ID map for precomputed GT comparison
        let original_to_internal: std::collections::HashMap<u32, u32> =
            if !gt.is_empty() && filter.strength() == 0 {
                index.original_ids.iter()
                    .enumerate()
                    .map(|(internal, &original)| (original, internal as u32))
                    .collect()
            } else {
                std::collections::HashMap::new()
            };

        for i in 0..recall_queries {
            let query = &query_vecs[i * dim..(i + 1) * dim];
            let results = &all_results[i];

            if filter.strength() == 0 && !gt.is_empty() && i < gt.len() {
                // Tie-aware: use distance of k-th GT neighbor as threshold
                let gt_ids: Vec<u32> = gt[i].iter().take(k).copied().collect();
                if let Some(&k_th_original) = gt_ids.last() {
                    if let Some(&k_th_internal) = original_to_internal.get(&k_th_original) {
                        let gt_k_dist = distance::distance(
                            query, index.store.vector(k_th_internal), metric,
                        );
                        let found = results.iter()
                            .filter(|r| r.dist <= gt_k_dist)
                            .count()
                            .min(k);
                        total_recall += found as f64 / k as f64;
                    }
                }
            } else if filter.strength() > 0 {
                // Brute-force filtered ground truth (tie-aware)
                let mut dists: Vec<(u32, f32)> = (0..n as u32)
                    .filter(|&p| filter.matches(&index.store, p))
                    .map(|p| (p, distance::distance(query, index.store.vector(p), metric)))
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                if dists.len() >= k {
                    let gt_k_dist = dists[k - 1].1;
                    let found = results.iter()
                        .filter(|r| r.dist <= gt_k_dist)
                        .count()
                        .min(k);
                    total_recall += found as f64 / k as f64;
                } else {
                    total_recall += 1.0;
                }
            }
        }
    }

    println!("  Queries: {nq}");
    println!("  QPS: {qps:.0}");
    println!("  Avg results: {:.1}", total_results as f64 / nq as f64);
    if compute_recall {
        println!("  Recall@{k}: {:.4}", total_recall / recall_queries as f64);
    }
    println!("  Total time: {elapsed:.3}s");
    println!();
}

fn run_synthetic_benchmark() {
    println!("=== PRISM Synthetic Benchmark ===");
    println!();

    let n = 10_000;
    let dim = 128;
    let k_attrs = 3;
    let cardinalities = &[10, 10, 10];

    use rand::prelude::*;
    let mut rng = rand::thread_rng();
    let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();

    println!("Building PRISM index ({n} points, {dim}d, {k_attrs} attrs)...");
    let t = Instant::now();
    let store = io::build_store_with_synthetic_attrs(vectors, dim, cardinalities);
    let config = PrismConfig {
        m_local: 16,
        m_greedy: 12,
        m_random: 4,
        beam_width: 32,
        ..Default::default()
    };
    let index = PrismIndex::build(store, config);
    println!("  Build time: {:.2}s", t.elapsed().as_secs_f64());
    println!("  Cells: {}", index.tree.cells.len());
    println!("  Edges: {}", index.graph.num_edges());
    println!();

    let nq = 100;
    let k = 10;
    let ef = 64;

    let queries: Vec<f32> = (0..nq * dim).map(|_| rng.gen::<f32>()).collect();

    println!("--- No Filter ---");
    run_queries(&index, &queries, dim, nq, k, ef, &Filter::none());

    println!("--- Single Attr (σ≈10%) ---");
    run_queries(&index, &queries, dim, nq, k, ef, &Filter::eq(0, 0));

    println!("--- Two Attr (σ≈1%) ---");
    let f2 = Filter::new(vec![(0, vec![0]), (1, vec![0])]);
    run_queries(&index, &queries, dim, nq, k, ef, &f2);

    println!("--- Three Attr (σ≈0.1%) ---");
    let f3 = Filter::new(vec![(0, vec![0]), (1, vec![0]), (2, vec![0])]);
    run_queries(&index, &queries, dim, nq, k, ef, &f3);
}

fn run_queries(
    index: &PrismIndex,
    queries: &[f32],
    dim: usize,
    nq: usize,
    k: usize,
    ef: usize,
    filter: &Filter,
) {
    let t = Instant::now();
    let results: Vec<_> = (0..nq)
        .into_par_iter()
        .map(|i| {
            let q = &queries[i * dim..(i + 1) * dim];
            index.search(q, filter, k, ef)
        })
        .collect();
    let elapsed = t.elapsed().as_secs_f64();
    let qps = nq as f64 / elapsed;
    let total_results: usize = results.iter().map(|r| r.len()).sum();

    println!("  QPS: {qps:.0}");
    println!("  Avg results: {:.1}", total_results as f64 / nq as f64);
    println!("  Total: {elapsed:.3}s");
    println!();
}

fn parse_arg(args: &[String], flag: &str, default: usize) -> usize {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(default)
}

fn parse_float_arg(args: &[String], flag: &str, default: f32) -> f32 {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(default)
}
