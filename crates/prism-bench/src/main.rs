use prism_ann::construct::{PrismConfig, PrismIndex};
use prism_ann::filter::Filter;
use prism_ann::io;
use prism_ann::point::PointStore;
use prism_ann::quantize::sq8_candidate_backend_name;
use prism_ann::search::{SearchExecution, SearchOutcome, SearchResult};

use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const MIN_TIMING_ROUNDS: usize = 3;
const MAX_TIMING_ROUNDS: usize = 100_000;
// A one-second floor materially reduces scheduler/turbo noise on the small
// synthetic batches while keeping the full local harness practical.
const MIN_TIMING_DURATION: Duration = Duration::from_secs(1);

struct TimingSummary {
    rounds: usize,
    elapsed_secs: f64,
    aggregate_qps: f64,
    min_round_qps: f64,
    max_round_qps: f64,
    round_qps_stddev: f64,
}

#[derive(Clone, Copy)]
struct SiftBenchmarkOptions {
    k: usize,
    ef: usize,
    max_n: usize,
    max_nq: usize,
    time_exact: bool,
    m_local: usize,
    m_greedy: usize,
    m_random: usize,
    beam_width: usize,
    single_filter_only: bool,
    broad_filter_sweep: bool,
    multi_cell_scan_threshold: usize,
    vamana_alpha: f32,
}

#[derive(Clone, Copy)]
struct QueryBenchmarkOptions<'a> {
    ground_truth: &'a [Vec<u32>],
    k: usize,
    ef: usize,
    time_exact: bool,
}

/// Warm once, then repeat a complete query batch until both the minimum round
/// count and minimum elapsed time are met. The returned results are from the
/// final measured round; all measured rounds contribute to aggregate QPS.
fn measure_query_batch<T>(nq: usize, mut run: impl FnMut() -> T) -> (T, TimingSummary) {
    assert!(nq > 0, "benchmark requires at least one query");

    // Exclude one-time thread-pool initialization and cold instruction/data
    // effects from the measured rounds.
    drop(run());

    let mut round_secs = Vec::new();
    let mut last = None;
    let mut total = Duration::ZERO;
    while (round_secs.len() < MIN_TIMING_ROUNDS || total < MIN_TIMING_DURATION)
        && round_secs.len() < MAX_TIMING_ROUNDS
    {
        let start = Instant::now();
        last = Some(run());
        let elapsed = start.elapsed();
        total += elapsed;
        round_secs.push(elapsed.as_secs_f64().max(f64::MIN_POSITIVE));
    }

    let round_qps: Vec<f64> = round_secs
        .iter()
        .map(|&seconds| nq as f64 / seconds)
        .collect();
    let mean = round_qps.iter().sum::<f64>() / round_qps.len() as f64;
    let variance = round_qps
        .iter()
        .map(|&value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / round_qps.len() as f64;
    let elapsed_secs = total.as_secs_f64();
    let summary = TimingSummary {
        rounds: round_secs.len(),
        elapsed_secs,
        aggregate_qps: nq as f64 * round_secs.len() as f64 / elapsed_secs,
        min_round_qps: round_qps.iter().copied().fold(f64::INFINITY, f64::min),
        max_round_qps: round_qps.iter().copied().fold(0.0, f64::max),
        round_qps_stddev: variance.sqrt(),
    };
    (last.expect("at least one timing round must run"), summary)
}

fn print_search_timing(timing: &TimingSummary) {
    println!("  End-to-end Rust search QPS: {:.0}", timing.aggregate_qps);
    println!(
        "  Timing: {} measured rounds after 1 warmup, {:.3}s total; round QPS min={:.0}, max={:.0}, sd={:.0}",
        timing.rounds,
        timing.elapsed_secs,
        timing.min_round_qps,
        timing.max_round_qps,
        timing.round_qps_stddev
    );
}

fn print_exact_timing(timing: &TimingSummary, rows: &[(Vec<SearchResult>, Duration)]) {
    println!("  Matched search_exact QPS: {:.0}", timing.aggregate_qps);
    println!(
        "  Exact timing: {} measured rounds after 1 warmup, {:.3}s total; round QPS min={:.0}, max={:.0}, sd={:.0}",
        timing.rounds,
        timing.elapsed_secs,
        timing.min_round_qps,
        timing.max_round_qps,
        timing.round_qps_stddev
    );
    let exact_p50 = raw_duration_percentile_ms(rows.iter().map(|(_, elapsed)| *elapsed), 0.50);
    let exact_p95 = raw_duration_percentile_ms(rows.iter().map(|(_, elapsed)| *elapsed), 0.95);
    let exact_p99 = raw_duration_percentile_ms(rows.iter().map(|(_, elapsed)| *elapsed), 0.99);
    println!(
        "  Last-round per-query search_exact latency p50/p95/p99={exact_p50:.3}/{exact_p95:.3}/{exact_p99:.3}ms"
    );
}

fn raw_duration_percentile_ms(durations: impl Iterator<Item = Duration>, percentile: f64) -> f64 {
    let mut values: Vec<f64> = durations
        .map(|duration| duration.as_secs_f64() * 1_000.0)
        .collect();
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable_by(f64::total_cmp);
    let rank = (percentile * values.len() as f64).ceil() as usize;
    values[rank.saturating_sub(1).min(values.len() - 1)]
}

fn duration_percentile_ms(
    outcomes: &[SearchOutcome],
    select: impl Fn(&SearchOutcome) -> Duration,
    percentile: f64,
) -> f64 {
    raw_duration_percentile_ms(outcomes.iter().map(select), percentile)
}

fn print_fallback_diagnostics(outcomes: &[SearchOutcome]) {
    let query_count = outcomes.len();
    if query_count == 0 {
        return;
    }
    let fallback_count = outcomes
        .iter()
        .filter(|outcome| outcome.diagnostics.used_exact_fallback)
        .count();
    let complete_count = outcomes
        .iter()
        .filter(|outcome| outcome.diagnostics.complete)
        .count();
    let first_execution = outcomes[0].diagnostics.primary_execution;
    let execution_consistent = outcomes
        .iter()
        .all(|outcome| outcome.diagnostics.primary_execution == first_execution);
    let primary_mean_ms = outcomes
        .iter()
        .map(|outcome| outcome.diagnostics.primary_elapsed.as_secs_f64())
        .sum::<f64>()
        * 1_000.0
        / query_count as f64;
    let fallback_mean_ms = if fallback_count == 0 {
        0.0
    } else {
        outcomes
            .iter()
            .filter(|outcome| outcome.diagnostics.used_exact_fallback)
            .map(|outcome| outcome.diagnostics.fallback_elapsed.as_secs_f64())
            .sum::<f64>()
            * 1_000.0
            / fallback_count as f64
    };
    let total_p50 = duration_percentile_ms(outcomes, |o| o.diagnostics.total_elapsed, 0.50);
    let total_p95 = duration_percentile_ms(outcomes, |o| o.diagnostics.total_elapsed, 0.95);
    let total_p99 = duration_percentile_ms(outcomes, |o| o.diagnostics.total_elapsed, 0.99);
    let primary_p50 = duration_percentile_ms(outcomes, |o| o.diagnostics.primary_elapsed, 0.50);
    let primary_p95 = duration_percentile_ms(outcomes, |o| o.diagnostics.primary_elapsed, 0.95);
    let primary_p99 = duration_percentile_ms(outcomes, |o| o.diagnostics.primary_elapsed, 0.99);
    println!(
        "  Last-round diagnostics: exact fallback {fallback_count}/{query_count} ({:.2}%), core-complete {complete_count}/{query_count}",
        fallback_count as f64 * 100.0 / query_count as f64
    );
    if execution_consistent {
        println!("  Primary execution: {first_execution:?}");
    } else {
        println!("  Primary execution: mixed across the measured queries");
    }
    println!(
        "  Core phase means: primary={primary_mean_ms:.3}ms/query, fallback={fallback_mean_ms:.3}ms/fallback (per-query durations, not wall time)"
    );
    println!(
        "  Last-round per-query latency: total p50/p95/p99={total_p50:.3}/{total_p95:.3}/{total_p99:.3}ms; primary p50/p95/p99={primary_p50:.3}/{primary_p95:.3}/{primary_p99:.3}ms"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!(
            "Usage: prism-bench <dataset-dir> [--k N] [--ef N] [--n N] [--nq N] [--m-local N]"
        );
        eprintln!("  dataset-dir: path to directory containing .fvecs files");
        eprintln!("  --k: number of nearest neighbors (default: 10)");
        eprintln!("  --ef: search beam width (default: 200)");
        eprintln!("  --n: use first N base vectors (default: all)");
        eprintln!("  --nq: use first N query vectors (default: all)");
        eprintln!("  --time-exact: time search_exact with the same query batch");
        eprintln!("  --m-local: local graph degree (default: 48)");
        eprintln!("  --m-greedy: greedy cross-cell degree (default: 12)");
        eprintln!("  --m-random: random cross-cell degree (default: 4)");
        eprintln!("  --beam-width: construction search-list width (default: 128)");
        eprintln!("  --single-filter-only: run only the 10% SIFT filter case");
        eprintln!("  --broad-filter-sweep: also run 50%, 60%, 70%, 80%, and 90% SIFT filters");
        eprintln!("  --multi-cell-scan-threshold: fragmented-filter scan limit (default: 500000)");
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
    let max_nq = parse_arg(&args, "--nq", 0); // 0 = use all
    let time_exact = args.iter().any(|arg| arg == "--time-exact");
    let m_local = parse_arg(&args, "--m-local", 48);
    let m_greedy = parse_arg(&args, "--m-greedy", 12);
    let m_random = parse_arg(&args, "--m-random", 4);
    let beam_width = parse_arg(&args, "--beam-width", 128);
    let single_filter_only = args.iter().any(|arg| arg == "--single-filter-only");
    let broad_filter_sweep = args.iter().any(|arg| arg == "--broad-filter-sweep");
    let multi_cell_scan_threshold = parse_arg(&args, "--multi-cell-scan-threshold", 500_000);
    let vamana_alpha = parse_float_arg(&args, "--vamana-alpha", 1.0);
    if k == 0 {
        eprintln!("--k must be greater than zero for Recall@k benchmarking");
        std::process::exit(2);
    }

    if dataset_dir.join("sift_base.fvecs").exists() {
        run_sift1m_benchmark(
            &dataset_dir,
            SiftBenchmarkOptions {
                k,
                ef,
                max_n,
                max_nq,
                time_exact,
                m_local,
                m_greedy,
                m_random,
                beam_width,
                single_filter_only,
                broad_filter_sweep,
                multi_cell_scan_threshold,
                vamana_alpha,
            },
        );
    } else {
        eprintln!("Dataset files not found in {:?}", dataset_dir);
        eprintln!("Expected: sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs");
        eprintln!("Download from: http://corpus-texmex.irisa.fr/");
        eprintln!();
        eprintln!("Running synthetic benchmark instead...");
        run_synthetic_benchmark();
    }
}

fn run_sift1m_benchmark(dir: &Path, options: SiftBenchmarkOptions) {
    let SiftBenchmarkOptions {
        k,
        ef,
        max_n,
        max_nq,
        time_exact,
        m_local,
        m_greedy,
        m_random,
        beam_width,
        single_filter_only,
        broad_filter_sweep,
        multi_cell_scan_threshold,
        vamana_alpha,
    } = options;
    println!(
        "=== PRISM Benchmark: SIFT1M ({} threads) ===",
        rayon::current_num_threads()
    );
    println!("SQ8 candidate backend: {}", sq8_candidate_backend_name());
    println!(
        "Search timing includes all Rust `search` work, including exact underfill fallback; per-query fallback diagnostics come from the final measured round."
    );
    println!();

    print!("Loading base vectors...");
    let t = Instant::now();
    let (mut base_vecs, dim) = io::load_fvecs(&dir.join("sift_base.fvecs")).unwrap();
    let mut n = base_vecs.len() / dim;
    if max_n > 0 && max_n < n {
        base_vecs.truncate(max_n * dim);
        n = max_n;
    }
    println!(" {n} vectors, {dim}d ({:.1}s)", t.elapsed().as_secs_f64());

    print!("Loading query vectors...");
    let t = Instant::now();
    let (mut query_vecs, _) = io::load_fvecs(&dir.join("sift_query.fvecs")).unwrap();
    let requested_query_values = max_nq.saturating_mul(dim);
    if max_nq > 0 && requested_query_values < query_vecs.len() {
        query_vecs.truncate(requested_query_values);
    }
    let nq = query_vecs.len() / dim;
    println!(
        " {nq} queries{} ({:.1}s)",
        if max_nq > 0 {
            " (explicit --nq cap)"
        } else {
            ""
        },
        t.elapsed().as_secs_f64()
    );

    print!("Loading ground truth...");
    let t = Instant::now();
    let gt = io::load_ivecs(&dir.join("sift_groundtruth.ivecs")).unwrap();
    println!(" {} entries ({:.1}s)", gt.len(), t.elapsed().as_secs_f64());

    let cardinalities = &[10, 10, 10]; // k=3 attributes, 10 values each
    println!(
        "Building PRISM index ({n} pts, 3 attrs x 10 values, M_local={m_local}, M_greedy={m_greedy}, M_random={m_random}, beam={beam_width}, alpha_v={vamana_alpha})..."
    );
    let store = io::build_store_with_synthetic_attrs(base_vecs, dim, cardinalities)
        .expect("valid SIFT synthetic attributes");
    let t = Instant::now();
    let config = PrismConfig {
        m_local,
        m_greedy,
        m_random,
        beam_width,
        vamana_alpha,
        multi_cell_scan_threshold,
        ..PrismConfig::default()
    };
    println!("  Graph build seed: {}", config.build_seed);
    println!(
        "  Cross-cell exact ranking limit: {} populated cells",
        config.cross_cell_exact_ranking_limit
    );
    println!(
        "  Multi-cell scan threshold: {} eligible points",
        config.multi_cell_scan_threshold
    );
    let index = PrismIndex::build(store, config).expect("SIFT index build failed");
    let build_time = t.elapsed().as_secs_f64();
    println!("  Index build time (PointStore already prepared): {build_time:.1}s");
    println!("  Cells: {}", index.tree().cells().len());
    println!("  Edges: {}", index.graph().num_edges());

    let degrees: Vec<usize> = (0..n)
        .map(|i| {
            index
                .graph()
                .degree(i as u32)
                .expect("benchmark point ID must be in range")
        })
        .collect();
    let avg_deg = degrees.iter().sum::<usize>() as f64 / n as f64;
    let min_deg = *degrees.iter().min().unwrap_or(&0);
    let max_deg = *degrees.iter().max().unwrap_or(&0);
    println!("  Avg degree: {avg_deg:.1}, min: {min_deg}, max: {max_deg}");

    let mut total_local = 0usize;
    for cell in index.tree().cells() {
        let pts = cell.point_ids();
        let cell_set: std::collections::HashSet<u32> = pts.iter().copied().collect();
        for &p in pts {
            let local = index
                .graph()
                .neighbors(p)
                .expect("cell point ID must be in graph")
                .iter()
                .filter(|&&nb| cell_set.contains(&nb))
                .count();
            total_local += local;
        }
    }
    println!(
        "  Avg local edges/node: {:.1}",
        total_local as f64 / n as f64
    );
    println!();

    let filtered_options = QueryBenchmarkOptions {
        ground_truth: &[],
        k,
        ef,
        time_exact,
    };

    if !single_filter_only {
        println!("--- No Filter (ef={ef}) ---");
        let use_gt = max_n == 0; // ground truth only valid for full dataset
        benchmark_queries(
            &index,
            &query_vecs,
            dim,
            &Filter::none(),
            QueryBenchmarkOptions {
                ground_truth: if use_gt { &gt } else { &[] },
                ..filtered_options
            },
        );
    }

    println!("--- Single Attr Filter (sigma~10%, ef={ef}) ---");
    benchmark_queries(
        &index,
        &query_vecs,
        dim,
        &Filter::eq(0, 0),
        filtered_options,
    );

    if broad_filter_sweep {
        for (label, values) in [
            ("50%", (0..5).collect::<Vec<u32>>()),
            ("60%", (0..6).collect::<Vec<u32>>()),
            ("70%", (0..7).collect::<Vec<u32>>()),
            ("80%", (0..8).collect::<Vec<u32>>()),
            ("90%", (0..9).collect::<Vec<u32>>()),
        ] {
            println!("--- IN Attr Filter (sigma~{label}, ef={ef}) ---");
            benchmark_queries(
                &index,
                &query_vecs,
                dim,
                &Filter::new(vec![(0, values)]),
                filtered_options,
            );
        }
    }

    if !single_filter_only {
        println!("--- Two Attr Filter (sigma~1%, ef={ef}) ---");
        let filter_1 = Filter::new(vec![(0, vec![0]), (1, vec![0])]);
        benchmark_queries(&index, &query_vecs, dim, &filter_1, filtered_options);
    }
}

fn benchmark_queries(
    index: &PrismIndex,
    query_vecs: &[f32],
    dim: usize,
    filter: &Filter,
    options: QueryBenchmarkOptions<'_>,
) {
    let QueryBenchmarkOptions {
        ground_truth: gt,
        k,
        ef,
        time_exact,
    } = options;
    use prism_ann::distance;
    let nq = query_vecs.len() / dim;
    let store = index.store();
    let n = store.len();
    let metric = index.config().metric;
    let plan = index
        .plan_filter(filter)
        .expect("benchmark filter planning failed");

    // The reported number covers query encoding, filter planning, traversal,
    // reranking, and any exact underfill fallback inside `search`.
    let (all_outcomes, timing) = measure_query_batch(nq, || {
        (0..nq)
            .into_par_iter()
            .map(|i| {
                let query = &query_vecs[i * dim..(i + 1) * dim];
                index
                    .search(query, filter, k, ef)
                    .expect("benchmark search failed")
            })
            .collect::<Vec<_>>()
    });
    // Opt-in because a full-dataset exact scan per query can dominate the run.
    // Uses the identical query batch, Rayon pool, warmup, and timing floor.
    let exact_measurement = if time_exact {
        Some(measure_query_batch(nq, || {
            (0..nq)
                .into_par_iter()
                .map(|i| {
                    let query = &query_vecs[i * dim..(i + 1) * dim];
                    let start = Instant::now();
                    let results = index
                        .search_exact(query, filter, k)
                        .expect("exact benchmark search failed");
                    (results, start.elapsed())
                })
                .collect::<Vec<_>>()
        }))
    } else {
        None
    };

    let total_results: usize = all_outcomes
        .iter()
        .map(|outcome| outcome.results.len())
        .sum();
    let target_per_query = k.min(plan.matching_points);
    let valid_unique_counts: Vec<usize> = all_outcomes
        .iter()
        .map(|outcome| {
            outcome
                .results
                .iter()
                .filter(|result| filter.matches(store, result.id))
                .map(|result| result.id)
                .collect::<std::collections::HashSet<_>>()
                .len()
        })
        .collect();
    let completeness = if target_per_query == 0 {
        if total_results == 0 {
            1.0
        } else {
            0.0
        }
    } else {
        valid_unique_counts
            .iter()
            .map(|&count| count.min(target_per_query))
            .sum::<usize>() as f64
            / (nq * target_per_query) as f64
    };
    let complete_query_rate = valid_unique_counts
        .iter()
        .filter(|&&count| count >= target_per_query)
        .count() as f64
        / nq as f64;
    let predicate_valid = all_outcomes
        .iter()
        .flat_map(|outcome| &outcome.results)
        .all(|result| filter.matches(store, result.id));

    let recall_queries = nq.min(1000); // limit brute-force recall to 1000 queries
    let mut total_recall = 0.0f64;
    let compute_recall = !gt.is_empty() || filter.strength() > 0;

    if compute_recall {
        let original_to_internal: std::collections::HashMap<u32, u32> =
            if !gt.is_empty() && filter.strength() == 0 {
                (0..n as u32)
                    .map(|internal| {
                        (
                            index
                                .original_id(internal)
                                .expect("benchmark internal ID must be in range"),
                            internal,
                        )
                    })
                    .collect()
            } else {
                std::collections::HashMap::new()
            };

        for i in 0..recall_queries {
            let query = &query_vecs[i * dim..(i + 1) * dim];
            let results = &all_outcomes[i].results;

            if filter.strength() == 0 && !gt.is_empty() && i < gt.len() {
                // Tie-aware recall: any distinct result no farther than the
                // k-th GT distance is accepted, so an arbitrary choice among
                // exact-distance ties is not penalized.
                let gt_ids: Vec<u32> = gt[i].iter().take(k).copied().collect();
                if let Some(&k_th_original) = gt_ids.last() {
                    if let Some(&k_th_internal) = original_to_internal.get(&k_th_original) {
                        let gt_k_dist = distance::distance(
                            query,
                            store
                                .vector(k_th_internal)
                                .expect("ground-truth internal ID must be in range"),
                            metric,
                        );
                        let found = results
                            .iter()
                            .filter(|r| r.dist <= gt_k_dist)
                            .map(|result| result.id)
                            .collect::<std::collections::HashSet<_>>()
                            .len()
                            .min(gt_ids.len());
                        total_recall += found as f64 / gt_ids.len() as f64;
                    }
                }
            } else if let Some((exact_rows, _)) = &exact_measurement {
                let truth = &exact_rows[i].0;
                if let Some(k_th) = truth.last() {
                    let found = results
                        .iter()
                        .filter(|result| {
                            result.dist <= k_th.dist && filter.matches(store, result.id)
                        })
                        .map(|result| result.id)
                        .collect::<std::collections::HashSet<_>>()
                        .len()
                        .min(truth.len());
                    total_recall += found as f64 / truth.len() as f64;
                } else {
                    total_recall += if results.is_empty() { 1.0 } else { 0.0 };
                }
            } else if filter.strength() > 0 {
                // Brute-force filtered ground truth with the same tie-aware
                // distance-threshold definition used above.
                let mut dists: Vec<(u32, f32)> = (0..n as u32)
                    .filter(|&p| filter.matches(store, p))
                    .map(|p| {
                        (
                            p,
                            distance::distance(
                                query,
                                store.vector(p).expect("scan point ID must be in range"),
                                metric,
                            ),
                        )
                    })
                    .collect();
                dists.sort_by(|a, b| a.1.total_cmp(&b.1));
                let target = k.min(dists.len());
                if target > 0 {
                    let gt_k_dist = dists[target - 1].1;
                    let found = results
                        .iter()
                        .filter(|result| {
                            result.dist <= gt_k_dist && filter.matches(store, result.id)
                        })
                        .map(|result| result.id)
                        .collect::<std::collections::HashSet<_>>()
                        .len()
                        .min(target);
                    total_recall += found as f64 / target as f64;
                } else {
                    total_recall += if results.is_empty() { 1.0 } else { 0.0 };
                }
            }
        }
    }

    println!("  Queries: {nq}");
    println!(
        "  Plan: {:?}, eligible={} ({:.4}%), cells={}",
        plan.regime,
        plan.matching_points,
        plan.selectivity * 100.0,
        plan.matching_cells
    );
    print_search_timing(&timing);
    if let Some((exact_rows, exact_timing)) = &exact_measurement {
        print_exact_timing(exact_timing, exact_rows);
    }
    print_fallback_diagnostics(&all_outcomes);
    println!("  Avg results: {:.1}", total_results as f64 / nq as f64);
    println!("  Predicate valid: {predicate_valid}");
    println!("  Result completeness: {completeness:.4}");
    println!("  Fully complete queries: {complete_query_rate:.4}");
    if compute_recall {
        println!(
            "  Tie-aware Recall@{k}: {:.4}",
            total_recall / recall_queries as f64
        );
    }
    println!();
}

fn run_synthetic_benchmark() {
    let graph_build_seed = PrismConfig::default().build_seed;
    println!("=== PRISM Synthetic Benchmark ===");
    println!("  Threads: {}", rayon::current_num_threads());
    println!("  SQ8 candidate backend: {}", sq8_candidate_backend_name());
    println!("  Vector/query RNG seed: 42; graph build seed: {graph_build_seed}");
    println!(
        "  Search timing includes exact underfill fallback; per-query fallback diagnostics come from the final measured round"
    );
    println!();

    let n = 10_000;
    let dim = 128;
    let k_attrs = 3;
    let cardinalities = [10usize, 10, 10];

    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(42);
    let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();

    println!("Building PRISM index ({n} points, {dim}d, {k_attrs} attrs)...");
    let mut attributes = vec![vec![0u32; n]; k_attrs];
    let mut stride = 1usize;
    for (column, &cardinality) in attributes.iter_mut().zip(&cardinalities) {
        for (point, value) in column.iter_mut().enumerate() {
            *value = ((point / stride) % cardinality) as u32;
        }
        stride *= cardinality;
    }
    // Moving nine of [9,9,9]'s members into [9,9,8] leaves a one-point cell, which
    // buys a true 0.01% LOW fixture without a fourth attribute or more cells.
    let rare_point = 999usize;
    let (first_two, third) = attributes.split_at_mut(2);
    let (first, second) = first_two.split_at(1);
    for (point, ((&value_0, &value_1), value_2)) in first[0]
        .iter()
        .zip(&second[0])
        .zip(&mut third[0])
        .enumerate()
    {
        if point != rare_point && value_0 == 9 && value_1 == 9 && *value_2 == 9 {
            *value_2 = 8;
        }
    }
    let store = PointStore::from_parts(vectors, dim, attributes)
        .expect("valid synthetic fixture with a 0.01% tail cell");
    let t = Instant::now();
    let config = PrismConfig::default();
    println!(
        "  Cross-cell exact ranking limit: {} populated cells",
        config.cross_cell_exact_ranking_limit
    );
    let index = PrismIndex::build(store, config).expect("synthetic index build failed");
    println!(
        "  Index build time (PointStore already prepared): {:.2}s",
        t.elapsed().as_secs_f64()
    );
    println!("  Cells: {}", index.tree().cells().len());
    println!("  Edges: {}", index.graph().num_edges());
    println!();

    let nq = 100;
    let k = 10;
    let ef = 64;

    let queries: Vec<f32> = (0..nq * dim).map(|_| rng.gen::<f32>()).collect();

    println!("--- No Filter ---");
    run_queries(&index, &queries, dim, nq, k, ef, &Filter::none());

    println!("--- IN Filter (sigma about 50%) ---");
    let f50 = Filter::new(vec![(0, vec![0, 1, 2, 3, 4])]);
    run_queries(&index, &queries, dim, nq, k, ef, &f50);

    println!("--- Single Attr (sigma~10%) ---");
    run_queries(&index, &queries, dim, nq, k, ef, &Filter::eq(0, 0));

    println!("--- Two Attr (sigma~1%) ---");
    let f2 = Filter::new(vec![(0, vec![0]), (1, vec![0])]);
    run_queries(&index, &queries, dim, nq, k, ef, &f2);

    println!("--- Three Attr (sigma~0.1%) ---");
    let f3 = Filter::new(vec![(0, vec![0]), (1, vec![0]), (2, vec![0])]);
    run_queries(&index, &queries, dim, nq, k, ef, &f3);

    println!("--- Rare Three Attr (sigma~0.01%) ---");
    let f_rare = Filter::new(vec![(0, vec![9]), (1, vec![9]), (2, vec![9])]);
    run_queries(&index, &queries, dim, nq, k, ef, &f_rare);

    // Where the tuple fixture exercises the selectivity router, this one pins the
    // physical decision: 2,500 eligible points take ExactScan under the 20,000
    // default, while scan_threshold=0 forces LocalGraph for SQ8 A/B work.
    println!("=== PRISM Synthetic Large-Cell Fixture ===");
    let large_vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();
    let public_store = io::build_store_with_synthetic_attrs(large_vectors.clone(), dim, &[4])
        .expect("valid public-route large-cell fixture");
    let baseline_store = io::build_store_with_synthetic_attrs(large_vectors.clone(), dim, &[4])
        .expect("valid large-cell baseline fixture");
    let forced_store = io::build_store_with_synthetic_attrs(large_vectors, dim, &[4])
        .expect("valid forced-graph large-cell fixture");
    let default_config = PrismConfig::default();
    let default_expansion = default_config.graph_expansion;
    println!(
        "  Cross-cell exact ranking limit: {} populated cells",
        default_config.cross_cell_exact_ranking_limit
    );

    let t = Instant::now();
    let public_index = PrismIndex::build(public_store, default_config.clone())
        .expect("large-cell public-route index build failed");
    println!(
        "  public-route index build (PointStore prepared): {:.2}s",
        t.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    let baseline_index = PrismIndex::build(
        baseline_store,
        PrismConfig {
            scan_threshold: 0,
            graph_expansion: 1,
            ..default_config.clone()
        },
    )
    .expect("large-cell baseline index build failed");
    println!(
        "  graph_expansion=1 index build (PointStore prepared): {:.2}s",
        t.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    let forced_index = PrismIndex::build(
        forced_store,
        PrismConfig {
            scan_threshold: 0,
            ..default_config
        },
    )
    .expect("large-cell forced-graph index build failed");
    println!(
        "  forced graph_expansion={default_expansion} index build (PointStore prepared): {:.2}s",
        t.elapsed().as_secs_f64()
    );
    println!(
        "  Cells: {} (2,500 points/cell)",
        forced_index.tree().cells().len()
    );
    let large_filter = Filter::eq(0, 0);
    let route_probe = &queries[..dim];
    assert_eq!(
        public_index
            .search(route_probe, &large_filter, k, ef)
            .expect("public-route probe failed")
            .diagnostics
            .primary_execution,
        SearchExecution::ExactScan
    );
    for index in [&baseline_index, &forced_index] {
        assert_eq!(
            index
                .search(route_probe, &large_filter, k, ef)
                .expect("forced-graph route probe failed")
                .diagnostics
                .primary_execution,
            SearchExecution::LocalGraph
        );
    }
    println!("--- Large Cell public default (expected ExactScan, sigma~25%) ---");
    run_queries(&public_index, &queries, dim, nq, k, ef, &large_filter);
    println!("--- Large Cell forced LocalGraph ablation (graph_expansion=1, sigma~25%) ---");
    run_queries(&baseline_index, &queries, dim, nq, k, ef, &large_filter);
    println!(
        "--- Large Cell forced LocalGraph default expansion (graph_expansion={default_expansion}, sigma~25%) ---"
    );
    run_queries(&forced_index, &queries, dim, nq, k, ef, &large_filter);
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
    let plan = index
        .plan_filter(filter)
        .expect("benchmark filter planning failed");
    let (outcomes, timing) = measure_query_batch(nq, || {
        (0..nq)
            .into_par_iter()
            .map(|i| {
                let q = &queries[i * dim..(i + 1) * dim];
                index
                    .search(q, filter, k, ef)
                    .expect("benchmark search failed")
            })
            .collect::<Vec<_>>()
    });
    let total_results: usize = outcomes.iter().map(|outcome| outcome.results.len()).sum();

    // Time the exact oracle separately with the identical query batch, Rayon
    // pool, warmup, and minimum-duration boundary used by public search.
    let (exact, exact_timing) = measure_query_batch(nq, || {
        (0..nq)
            .into_par_iter()
            .map(|i| {
                let q = &queries[i * dim..(i + 1) * dim];
                let start = Instant::now();
                let results = index
                    .search_exact(q, filter, k)
                    .expect("exact benchmark search failed");
                (results, start.elapsed())
            })
            .collect::<Vec<_>>()
    });
    let mut recall = 0.0f64;
    let mut completeness = 0.0f64;
    let mut fully_complete = 0usize;
    let mut predicate_valid = true;
    for i in 0..nq {
        let target = exact[i].0.len();
        let valid_unique: std::collections::HashSet<u32> = outcomes[i]
            .results
            .iter()
            .filter(|result| filter.matches(index.store(), result.id))
            .map(|result| result.id)
            .collect();
        predicate_valid &= outcomes[i]
            .results
            .iter()
            .all(|result| filter.matches(index.store(), result.id));
        if target == 0 {
            let correct_empty = outcomes[i].results.is_empty();
            recall += if correct_empty { 1.0 } else { 0.0 };
            completeness += if correct_empty { 1.0 } else { 0.0 };
            fully_complete += usize::from(correct_empty);
            continue;
        }
        let threshold = exact[i].0[target - 1].dist;
        let found = outcomes[i]
            .results
            .iter()
            .filter(|result| result.dist <= threshold && filter.matches(index.store(), result.id))
            .map(|result| result.id)
            .collect::<std::collections::HashSet<_>>()
            .len()
            .min(target);
        recall += found as f64 / target as f64;
        completeness += valid_unique.len().min(target) as f64 / target as f64;
        fully_complete += usize::from(valid_unique.len() >= target);
    }

    println!(
        "  Plan: {:?}, eligible={} ({:.4}%), cells={}",
        plan.regime,
        plan.matching_points,
        plan.selectivity * 100.0,
        plan.matching_cells
    );
    print_search_timing(&timing);
    print_exact_timing(&exact_timing, &exact);
    print_fallback_diagnostics(&outcomes);
    println!("  Avg results: {:.1}", total_results as f64 / nq as f64);
    println!("  Predicate valid: {predicate_valid}");
    println!("  Result completeness: {:.4}", completeness / nq as f64);
    println!(
        "  Fully complete queries: {:.4}",
        fully_complete as f64 / nq as f64
    );
    println!("  Tie-aware Recall@{k}: {:.4}", recall / nq as f64);
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
