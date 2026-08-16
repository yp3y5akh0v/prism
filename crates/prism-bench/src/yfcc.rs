//! YFCC-10M filtered search benchmark.
//!
//! PRISM filtered-IVF with multi-query cell batching (MQCB).
//! No vector duplication: 10M vectors stored once.

use prism_ann::distance;
use prism_ann::ivf::{kmeans, sorted_intersect_u16, IvfIndex, QueryStore, SpMat, VecStore};

use rayon::prelude::*;
use std::collections::HashSet;
use std::io::Read;
use std::path::Path;
use std::time::{Duration, Instant};

const MIN_KERNEL_ROUNDS: usize = 2;
const MAX_KERNEL_ROUNDS: usize = 5;
const MIN_KERNEL_DURATION: Duration = Duration::from_secs(1);

struct KernelTiming {
    rounds: usize,
    elapsed_secs: f64,
    aggregate_qps: f64,
    min_round_qps: f64,
    max_round_qps: f64,
    round_qps_stddev: f64,
}

fn measure_mqcb_core<T>(nq: usize, mut run: impl FnMut() -> T) -> (T, KernelTiming) {
    assert!(nq > 0, "benchmark requires at least one query");
    let mut round_secs = Vec::new();
    let mut total = Duration::ZERO;
    let mut last = None;
    while (round_secs.len() < MIN_KERNEL_ROUNDS || total < MIN_KERNEL_DURATION)
        && round_secs.len() < MAX_KERNEL_ROUNDS
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
    let timing = KernelTiming {
        rounds: round_secs.len(),
        elapsed_secs,
        aggregate_qps: nq as f64 * round_secs.len() as f64 / elapsed_secs,
        min_round_qps: round_qps.iter().copied().fold(f64::INFINITY, f64::min),
        max_round_qps: round_qps.iter().copied().fold(0.0, f64::max),
        round_qps_stddev: variance.sqrt(),
    };
    (last.expect("at least one timing round must run"), timing)
}

fn candidate_clusters_for_tags<'a>(
    tags: &[usize],
    n_tags: usize,
    n_clusters: usize,
    clusters_for_tag: impl Fn(usize) -> Option<&'a [u16]>,
) -> Vec<u16> {
    if tags.is_empty() {
        return (0..n_clusters).map(|cluster| cluster as u16).collect();
    }

    let mut unique_tags = tags.to_vec();
    unique_tags.sort_unstable();
    unique_tags.dedup();
    if unique_tags[0] >= n_tags {
        return Vec::new();
    }

    let mut matching = clusters_for_tag(unique_tags[0])
        .map(<[u16]>::to_vec)
        .unwrap_or_default();
    for &tag in &unique_tags[1..] {
        if tag >= n_tags {
            return Vec::new();
        }
        let Some(tag_clusters) = clusters_for_tag(tag) else {
            return Vec::new();
        };
        matching = sorted_intersect_u16(&matching, tag_clusters);
        if matching.is_empty() {
            break;
        }
    }
    matching
}

fn build_ground_truth_sets(
    ground_truth: &[i32],
    nq: usize,
    gt_k: usize,
    requested_k: usize,
) -> Vec<HashSet<u32>> {
    assert_eq!(
        ground_truth.len(),
        nq.checked_mul(gt_k)
            .expect("ground-truth shape exceeds addressable memory"),
        "ground-truth data length must equal nq * gt_k"
    );
    let accepted = requested_k.min(gt_k);
    (0..nq)
        .map(|qi| {
            ground_truth[qi * gt_k..(qi + 1) * gt_k]
                .iter()
                .take(accepted)
                .filter(|&&id| id >= 0)
                .map(|&id| id as u32)
                .collect()
        })
        .collect()
}

fn recall_at_k(results: &[Vec<u32>], truth: &[HashSet<u32>], k: usize) -> f64 {
    assert_eq!(results.len(), truth.len());
    if results.is_empty() {
        return 1.0;
    }

    let mut total = 0.0f64;
    for (row, expected) in results.iter().zip(truth) {
        let found = row
            .iter()
            .take(k)
            .copied()
            .filter(|id| expected.contains(id))
            .collect::<HashSet<_>>()
            .len();
        if expected.is_empty() {
            total += if row.is_empty() { 1.0 } else { 0.0 };
        } else {
            total += found as f64 / expected.len() as f64;
        }
    }
    total / results.len() as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("datasets/yfcc10m");
    let dir = Path::new(dir);
    let num_clusters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let kmeans_iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);

    if let Some(threads) = args.get(4).and_then(|s| s.parse().ok()) {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .unwrap();
    }

    println!(
        "=== PRISM Benchmark: YFCC-10M ({} threads) ===\n",
        rayon::current_num_threads()
    );
    println!(
        "Reported search QPS is the warmed multi-query cell batching (MQCB) core rate: internal query binary encoding (when binary_rerank > 0) is included; CSR tag parsing, tag-cluster selection, and centroid ranking are precomputed and excluded.\n"
    );
    println!("K-means initialization seed: 42 (fixed for repeatable builds).\n");

    print!("Loading base vectors...");
    let t = Instant::now();
    let (base_u8, n, dim) = read_u8bin(&dir.join("base.10M.u8bin"));
    println!(" {n} vectors, {dim}d ({:.1}s)", t.elapsed().as_secs_f64());

    print!("Loading base metadata...");
    let t = Instant::now();
    let base_meta = read_spmat(&dir.join("base.metadata.10M.spmat"));
    println!(
        " {} vectors, {} tags, {} entries ({:.1}s)",
        base_meta.rows(),
        base_meta.cols(),
        base_meta.indices().len(),
        t.elapsed().as_secs_f64()
    );

    print!("Loading queries...");
    let t = Instant::now();
    let (queries_u8, nq, qd) = read_u8bin(&dir.join("query.public.100K.u8bin"));
    assert_eq!(qd, dim);
    println!(" {nq} queries ({:.1}s)", t.elapsed().as_secs_f64());

    print!("Loading query metadata...");
    let t = Instant::now();
    let query_meta = read_spmat(&dir.join("query.metadata.public.100K.spmat"));
    query_meta
        .validate(nq)
        .unwrap_or_else(|error| panic!("invalid query metadata: {error}"));
    println!(
        " {} queries ({:.1}s)",
        query_meta.rows(),
        t.elapsed().as_secs_f64()
    );

    print!("Loading ground truth...");
    let t = Instant::now();
    let (gt, gt_nq, gt_k) = read_ibin(&dir.join("GT.public.ibin"));
    assert_eq!(gt_nq, nq);
    println!(
        " {gt_nq} queries, k={gt_k} ({:.1}s)",
        t.elapsed().as_secs_f64()
    );

    let query_tags: Vec<Vec<usize>> = (0..nq)
        .map(|qi| {
            let start = query_meta.indptr()[qi] as usize;
            let end = query_meta.indptr()[qi + 1] as usize;
            let mut tags: Vec<usize> = query_meta.indices()[start..end]
                .iter()
                .map(|&t| t as usize)
                .collect();
            tags.sort_unstable();
            tags.dedup();
            tags
        })
        .collect();

    let zero_tag_count = query_tags.iter().filter(|t| t.is_empty()).count();
    let one_tag_count = query_tags.iter().filter(|t| t.len() == 1).count();
    let two_tag_count = query_tags.iter().filter(|t| t.len() == 2).count();
    let many_tag_count = query_tags.iter().filter(|t| t.len() > 2).count();
    println!(
        "\nQuery breakdown: {zero_tag_count} 0-tag, {one_tag_count} 1-tag, {two_tag_count} 2-tag, {many_tag_count} >2-tag\n"
    );

    println!("K-means clustering ({num_clusters} clusters, {kmeans_iters} iters)...");
    let t = Instant::now();
    let base_store = VecStore::U8(base_u8);
    let (assignments, centroids) =
        kmeans(&base_store, n, dim, num_clusters, kmeans_iters).expect("YFCC k-means failed");
    println!("  K-means done ({:.1}s)", t.elapsed().as_secs_f64());

    print!("Building IVF index...");
    let t = Instant::now();
    let ivf = IvfIndex::build(&base_store, &base_meta, &assignments, n, dim, num_clusters)
        .expect("valid YFCC IVF index");
    println!(" ({:.1}s)", t.elapsed().as_secs_f64());

    drop(base_store); // free original vectors (reordered copy in ivf)
    drop(assignments);

    let k = 10;
    let configs: Vec<(usize, usize)> = vec![
        (50, 30),
        (50, 40),
        (50, 45),
        (50, 50),
        (50, 55),
        (50, 60),
        (50, 80),
        (50, 100),
        (100, 40),
        (100, 60),
        (100, 80),
        (100, 100),
        (200, 60),
        (200, 100),
        (500, 150),
    ];

    let max_nprobe = configs.iter().map(|&(_, np)| np).max().unwrap();
    print!("Precomputing filtered centroid assignments (max nprobe={max_nprobe})...");
    let t = Instant::now();
    let query_top_clusters: Vec<Vec<usize>> = (0..nq)
        .into_par_iter()
        .map(|qi| {
            let q_u8 = &queries_u8[qi * dim..(qi + 1) * dim];
            let tags = &query_tags[qi];

            // Empty filters search all clusters. Nonempty filters intersect
            // every required tag, not merely the first two.
            let candidates =
                candidate_clusters_for_tags(tags, ivf.n_tags(), ivf.n_clusters(), |tag| {
                    ivf.clusters_for_tag(tag)
                });

            if candidates.is_empty() {
                return vec![];
            }

            let centroids_u8 = match &centroids {
                VecStore::U8(v) => v,
                _ => unreachable!(),
            };
            let mut cluster_dists: Vec<(usize, u64)> = candidates
                .iter()
                .map(|&ci| {
                    let ci = ci as usize;
                    let cent = &centroids_u8[ci * dim..(ci + 1) * dim];
                    (ci, distance::l2_sq8(q_u8, cent))
                })
                .collect();

            let np = max_nprobe.min(cluster_dists.len());
            cluster_dists.select_nth_unstable_by_key(np - 1, |&(_, d)| d);
            cluster_dists.truncate(np);
            cluster_dists.sort_unstable_by_key(|&(_, d)| d);
            cluster_dists.iter().map(|&(ci, _)| ci).collect()
        })
        .collect();
    println!(" ({:.3}s)", t.elapsed().as_secs_f64());

    // Standard ID-overlap Recall@k for the official ground truth: only the
    // first k truth IDs are eligible hits, and duplicate result IDs count once.
    let gt_sets = build_ground_truth_sets(&gt, nq, gt_k, k);

    // One global untimed warmup is enough to initialize Rayon and fault the
    // common query/index pages before per-configuration timing.
    let (warm_ef, warm_nprobe) = configs[0];
    drop(
        ivf.batch_search_mqcb(
            &QueryStore::U8(&queries_u8),
            nq,
            &query_tags,
            &query_top_clusters,
            k,
            warm_ef,
            warm_nprobe,
            0,
        )
        .expect("YFCC warmup search failed"),
    );

    for &br in &[0] {
        println!("\n--- MQCB Search (k={k}, binary_rerank={br}) ---");
        for &(ef, n_probe) in &configs {
            let (results, timing) = measure_mqcb_core(nq, || {
                ivf.batch_search_mqcb(
                    &QueryStore::U8(&queries_u8),
                    nq,
                    &query_tags,
                    &query_top_clusters,
                    k,
                    ef,
                    n_probe,
                    br,
                )
                .expect("YFCC measured search failed")
            });
            let recall = recall_at_k(&results, &gt_sets, k);
            println!(
                "  ef={ef:3}, nprobe={n_probe:3}: MQCB-core QPS={:8.0}, Recall@{k}={recall:.4}, rounds={}, measured={:.3}s, round-QPS[min={:.0}, max={:.0}, sd={:.0}]",
                timing.aggregate_qps,
                timing.rounds,
                timing.elapsed_secs,
                timing.min_round_qps,
                timing.max_round_qps,
                timing.round_qps_stddev
            );
        }
    }

    println!(
        "\nIndex: {n} base vectors, {} clusters, {dim}d, {} tags",
        num_clusters,
        base_meta.cols(),
    );
}

fn read_u8bin(path: &Path) -> (Vec<u8>, usize, usize) {
    let mut f = std::fs::File::open(path).expect("open u8bin");
    let mut header = [0u8; 8];
    f.read_exact(&mut header).expect("read header");
    let n = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
    let d = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
    let mut data = vec![0u8; n * d];
    f.read_exact(&mut data).expect("read data");
    (data, n, d)
}

fn read_ibin(path: &Path) -> (Vec<i32>, usize, usize) {
    let mut f = std::fs::File::open(path).expect("open ibin");
    let mut header = [0u8; 8];
    f.read_exact(&mut header).expect("read header");
    let n = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
    let k = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
    let mut raw = vec![0u8; n * k * 4];
    f.read_exact(&mut raw).expect("read data");
    let data: Vec<i32> = raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (data, n, k)
}

fn read_spmat(path: &Path) -> SpMat {
    let mut f = std::fs::File::open(path).expect("open spmat");
    let mut buf8 = [0u8; 8];

    f.read_exact(&mut buf8).unwrap();
    let rows = i64::from_le_bytes(buf8) as usize;
    f.read_exact(&mut buf8).unwrap();
    let cols = i64::from_le_bytes(buf8) as usize;
    f.read_exact(&mut buf8).unwrap();
    let nnz = i64::from_le_bytes(buf8) as usize;

    let mut indptr_raw = vec![0u8; (rows + 1) * 8];
    f.read_exact(&mut indptr_raw).unwrap();
    let indptr: Vec<i64> = indptr_raw
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let mut indices_raw = vec![0u8; nnz * 4];
    f.read_exact(&mut indices_raw).unwrap();
    let indices: Vec<i32> = indices_raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    SpMat::new(rows, cols, indptr, indices).expect("valid spmat file")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_clusters_support_empty_and_arbitrary_tag_counts() {
        let postings = [vec![0, 1, 2], vec![1, 2], vec![2, 3], vec![2]];
        assert_eq!(
            candidate_clusters_for_tags(&[], postings.len(), 4, |tag| {
                postings.get(tag).map(Vec::as_slice)
            }),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            candidate_clusters_for_tags(&[0], postings.len(), 4, |tag| {
                postings.get(tag).map(Vec::as_slice)
            }),
            vec![0, 1, 2]
        );
        assert_eq!(
            candidate_clusters_for_tags(&[0, 1, 2], postings.len(), 4, |tag| {
                postings.get(tag).map(Vec::as_slice)
            }),
            vec![2]
        );
        assert_eq!(
            candidate_clusters_for_tags(&[2, 0, 2, 1], postings.len(), 4, |tag| {
                postings.get(tag).map(Vec::as_slice)
            }),
            vec![2]
        );
        assert!(
            candidate_clusters_for_tags(&[99], postings.len(), 4, |tag| {
                postings.get(tag).map(Vec::as_slice)
            })
            .is_empty()
        );
    }

    #[test]
    fn recall_uses_only_top_k_truth_and_deduplicates_results() {
        let gt = vec![0, 1, 2, 3, 10, 11, 12, 13];
        let truth = build_ground_truth_sets(&gt, 2, 4, 2);

        // IDs 2/3 and 12/13 are in the file but outside the requested top 2.
        assert_eq!(recall_at_k(&[vec![2, 3], vec![12, 13]], &truth, 2), 0.0);

        // A duplicated correct ID contributes one hit, not two.
        assert_eq!(recall_at_k(&[vec![0, 0], vec![10, 11]], &truth, 2), 0.75);
    }

    #[test]
    fn empty_truth_requires_an_empty_result_for_full_credit() {
        let truth = vec![HashSet::new(), HashSet::new()];
        assert_eq!(recall_at_k(&[Vec::new(), vec![7]], &truth, 10), 0.5);
    }
}
