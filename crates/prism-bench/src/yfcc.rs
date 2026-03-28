//! YFCC-10M filtered search benchmark.
//!
//! IVF² (geometric clusters × tag posting lists) + MQCB.
//! No vector duplication — 10M vectors stored once.

use prism_ann::binary::BinaryStore;
use prism_ann::distance;
use prism_ann::ivf::{IvfIndex, QueryStore, SpMat, VecStore, kmeans, sorted_intersect_u16};
use prism_ann::point::PointStore;

use rayon::prelude::*;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).map(|s| s.as_str()).unwrap_or("datasets/yfcc10m");
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

    // Load base vectors (uint8)
    print!("Loading base vectors...");
    let t = Instant::now();
    let (base_u8, n, dim) = read_u8bin(&dir.join("base.10M.u8bin"));
    println!(" {n} vectors, {dim}d ({:.1}s)", t.elapsed().as_secs_f64());

    // Load base metadata (tag associations)
    print!("Loading base metadata...");
    let t = Instant::now();
    let base_meta = read_spmat(&dir.join("base.metadata.10M.spmat"));
    println!(
        " {} vectors, {} tags, {} entries ({:.1}s)",
        base_meta.rows, base_meta.cols, base_meta.indices.len(), t.elapsed().as_secs_f64()
    );

    // Load queries
    print!("Loading queries...");
    let t = Instant::now();
    let (queries_u8, nq, qd) = read_u8bin(&dir.join("query.public.100K.u8bin"));
    assert_eq!(qd, dim);
    println!(" {nq} queries ({:.1}s)", t.elapsed().as_secs_f64());

    // Load query metadata (required tags)
    print!("Loading query metadata...");
    let t = Instant::now();
    let query_meta = read_spmat(&dir.join("query.metadata.public.100K.spmat"));
    println!(
        " {} queries ({:.1}s)",
        query_meta.rows, t.elapsed().as_secs_f64()
    );

    // Load ground truth
    print!("Loading ground truth...");
    let t = Instant::now();
    let (gt, gt_nq, gt_k) = read_ibin(&dir.join("GT.public.ibin"));
    assert_eq!(gt_nq, nq);
    println!(
        " {gt_nq} queries, k={gt_k} ({:.1}s)",
        t.elapsed().as_secs_f64()
    );

    // Parse query filters
    let query_tags: Vec<Vec<usize>> = (0..nq)
        .map(|qi| {
            let start = query_meta.indptr[qi] as usize;
            let end = query_meta.indptr[qi + 1] as usize;
            query_meta.indices[start..end]
                .iter()
                .map(|&t| t as usize)
                .collect()
        })
        .collect();

    let one_tag_count = query_tags.iter().filter(|t| t.len() == 1).count();
    let two_tag_count = query_tags.iter().filter(|t| t.len() == 2).count();
    println!("\nQuery breakdown: {one_tag_count} 1-tag, {two_tag_count} 2-tag\n");

    // K-means clustering
    println!("K-means clustering ({num_clusters} clusters, {kmeans_iters} iters)...");
    let t = Instant::now();
    let base_store = VecStore::U8(base_u8);
    let (assignments, centroids) = kmeans(&base_store, n, dim, num_clusters, kmeans_iters);
    println!("  K-means done ({:.1}s)", t.elapsed().as_secs_f64());

    // Build IVF index (reorder + per-cluster tag index)
    print!("Building IVF index...");
    let t = Instant::now();
    let ivf = IvfIndex::build(&base_store, &base_meta, &assignments, n, dim, num_clusters);
    println!(" ({:.1}s)", t.elapsed().as_secs_f64());

    drop(base_store); // free original vectors (reordered copy in ivf)
    drop(assignments);

    // Build binary codes from reordered vectors
    print!("Building binary codes...");
    let t = Instant::now();
    let vectors_u8 = match &ivf.vectors {
        VecStore::U8(v) => v,
        _ => unreachable!(),
    };
    let base_f32: Vec<f32> = vectors_u8.iter().map(|&v| v as f32).collect();
    let store = PointStore::from_parts(base_f32, dim, vec![vec![0u32; n]]);
    let binary = BinaryStore::build(&store);
    drop(store);
    println!(" ({:.1}s)", t.elapsed().as_secs_f64());

    // Benchmark filtered search
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

    // Precompute query binary codes (one-time, independent of ef/nprobe)
    print!("Precomputing query binary codes...");
    let t = Instant::now();
    let query_binary: Vec<Vec<u64>> = (0..nq)
        .into_par_iter()
        .map(|qi| {
            let q_u8 = &queries_u8[qi * dim..(qi + 1) * dim];
            let q_f32: Vec<f32> = q_u8.iter().map(|&v| v as f32).collect();
            binary.encode_query(&q_f32)
        })
        .collect();
    println!(" ({:.3}s)", t.elapsed().as_secs_f64());

    // Precompute filtered centroid assignments (skip clusters without matching tags)
    let max_nprobe = configs.iter().map(|&(_, np)| np).max().unwrap();
    print!("Precomputing filtered centroid assignments (max nprobe={max_nprobe})...");
    let t = Instant::now();
    let query_top_clusters: Vec<Vec<usize>> = (0..nq)
        .into_par_iter()
        .map(|qi| {
            let q_u8 = &queries_u8[qi * dim..(qi + 1) * dim];
            let tags = &query_tags[qi];

            // Get candidate clusters (only those with matching vectors)
            let candidates: std::borrow::Cow<[u16]> = if tags.len() == 1 {
                let t = tags[0];
                if t < ivf.tag_clusters.len() {
                    std::borrow::Cow::Borrowed(&ivf.tag_clusters[t])
                } else {
                    std::borrow::Cow::Owned(vec![])
                }
            } else {
                let t0 = tags[0];
                let t1 = tags[1];
                let a = if t0 < ivf.tag_clusters.len() { &ivf.tag_clusters[t0][..] } else { &[] };
                let b = if t1 < ivf.tag_clusters.len() { &ivf.tag_clusters[t1][..] } else { &[] };
                std::borrow::Cow::Owned(sorted_intersect_u16(a, b))
            };

            if candidates.is_empty() {
                return vec![];
            }

            // Compute centroid distances only for candidate clusters
            let centroids_u8 = match &centroids {
                VecStore::U8(v) => v,
                _ => unreachable!(),
            };
            let mut cluster_dists: Vec<(usize, u32)> = candidates.iter()
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

    // Precompute ground truth sets for recall computation
    let gt_sets: Vec<std::collections::HashSet<u32>> = (0..nq)
        .map(|qi| {
            gt[qi * gt_k..(qi + 1) * gt_k]
                .iter()
                .filter(|&&id| id >= 0)
                .map(|&id| id as u32)
                .collect()
        })
        .collect();

    let compute_recall = |results: &[Vec<u32>], k: usize| -> f64 {
        let mut total = 0.0f64;
        for qi in 0..nq {
            let found = results[qi].iter().take(k).filter(|id| gt_sets[qi].contains(id)).count();
            let gt_valid = gt_sets[qi].len().min(k);
            if gt_valid > 0 {
                total += found as f64 / gt_valid as f64;
            } else {
                total += 1.0;
            }
        }
        total / nq as f64
    };

    for &br in &[0] {
        println!("\n--- MQCB Search (k={k}, binary_rerank={br}) ---");
        for &(ef, n_probe) in &configs {
            let t = Instant::now();
            let results = ivf.batch_search_mqcb(
                &QueryStore::U8(&queries_u8), nq, &query_tags, &query_binary, &query_top_clusters,
                &binary, k, ef, n_probe, br,
            );
            let elapsed = t.elapsed().as_secs_f64();
            let qps = nq as f64 / elapsed;
            let recall = compute_recall(&results, k);
            println!(
                "  ef={ef:3}, nprobe={n_probe:3}: QPS={qps:8.0}, Recall@{k}={recall:.4}, Time={elapsed:.3}s"
            );
        }
    }

    println!(
        "\nIndex: {n} base vectors, {} clusters, {dim}d, {} tags",
        num_clusters, base_meta.cols,
    );
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

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

    SpMat {
        rows,
        cols,
        indptr,
        indices,
    }
}
