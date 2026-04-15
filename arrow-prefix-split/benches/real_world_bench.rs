// Licensed under the Apache License, Version 2.0.

//! Real-world benchmark: prefix-split search on Public BI Benchmark datasets.
//!
//! Datasets used
//! ─────────────
//! • Hatred_1   – Twitter hate-speech data.  Filter column: `Keyword` (col 13).
//!   Queries: single-value equality  ("retard")  and  IN-clause
//!            ("fatass","fatso","fattie") derived from the real SQL files.
//!
//! • Generico_1 – Guatemalan advertising spend data.
//!   Filter column: `Medio` (col 24) and `Anunciante` (col 0).
//!   Query: IN ('BANTRAB/TODOTICKET', 'TODOTICKET', 'TODOTICKET.COM') on Anunciante.
//!
//! Each sample CSV has 19 data rows (pipe-delimited, no header, "null" for NULLs).
//! We replicate the rows to reach TARGET_ROWS so the benchmark is meaningful.
//!
//! # Functionality gaps noted
//! The queries also use range predicates on string columns
//!   ("State" >= 'AK' AND "State" <= 'CT')
//! which PrefixSplitArray does NOT currently support.  Only the equality /
//! IN-clause portions are benchmarked here.
//!
//! Run with:
//!   cargo bench -p arrow-prefix-split --bench real_world_bench

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use arrow_array::{Array, StringArray};
use arrow_prefix_split::{PrefixSplitBuilder, PrefixSplitConfig, search_eq};

// ── Constants ──────────────────────────────────────────────────────────────────

const TARGET_ROWS: usize = 100_000;

const HATRED_CSV: &str =
    "/Users/zixpeng/Documents/Research/CS681/public_bi_benchmark/benchmark/Hatred/samples/Hatred_1.sample.csv";

const GENERICO_CSV: &str =
    "/Users/zixpeng/Documents/Research/CS681/public_bi_benchmark/benchmark/Generico/samples/Generico_1.sample.csv";

// ── CSV loader ─────────────────────────────────────────────────────────────────

/// Load a pipe-delimited CSV file (no header row) and extract column `col_idx`.
/// "null" values are treated as None.  Rows are replicated up to `target_rows`.
fn load_column(path: &str, col_idx: usize, target_rows: usize) -> Vec<Option<String>> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Cannot read {path}: {e}"));

    let raw: Vec<Option<String>> = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let field = line.split('|').nth(col_idx).unwrap_or("").trim().to_string();
            if field.eq_ignore_ascii_case("null") || field.is_empty() {
                None
            } else {
                Some(field)
            }
        })
        .collect();

    assert!(!raw.is_empty(), "No rows loaded from {path}");

    // Replicate to reach target_rows.
    let mut out = Vec::with_capacity(target_rows);
    while out.len() < target_rows {
        let take = raw.len().min(target_rows - out.len());
        out.extend_from_slice(&raw[..take]);
    }
    out
}

// ── Array builders ─────────────────────────────────────────────────────────────

fn build_string_array(col: &[Option<String>]) -> StringArray {
    StringArray::from(col.iter().map(|v| v.as_deref()).collect::<Vec<_>>())
}

fn build_psa(
    col: &[Option<String>],
    threshold: f64,
    plateau: f64,
) -> arrow_prefix_split::PrefixSplitArray {
    let cfg = PrefixSplitConfig {
        segment_size: 1_000,
        prefix_distinguish_threshold: threshold,
        max_prefix_len: 64,
        plateau_min_gain_fraction: plateau,
        plateau_min_progress: 0.5,
    };
    let mut b = PrefixSplitBuilder::with_config(cfg);
    for v in col {
        b.append_option(v.as_deref());
    }
    b.finish()
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/// Run an IN-clause by calling search_eq for each needle and merging indices.
fn search_in(array: &arrow_prefix_split::PrefixSplitArray, needles: &[&str]) -> Vec<usize> {
    let mut result: Vec<usize> = needles.iter().flat_map(|n| search_eq(array, n)).collect();
    result.sort_unstable();
    result.dedup();
    result
}

fn sa_search_eq(sa: &StringArray, needle: &str) -> Vec<usize> {
    (0..sa.len())
        .filter(|&i| !sa.is_null(i) && sa.value(i) == needle)
        .collect()
}

fn sa_search_in(sa: &StringArray, needles: &[&str]) -> Vec<usize> {
    (0..sa.len())
        .filter(|&i| !sa.is_null(i) && needles.contains(&sa.value(i)))
        .collect()
}

// ── Benchmarks ─────────────────────────────────────────────────────────────────

/// Hatred_1 · Keyword column
/// Queries: single equality and IN-clause derived from real SQL files.
fn bench_hatred_keyword(c: &mut Criterion) {
    let col = load_column(HATRED_CSV, 13, TARGET_ROWS);
    let n = col.len();

    let sa          = build_string_array(&col);
    let psa_90      = build_psa(&col, 0.9, 0.0);   // 90 % threshold, no plateau
    let psa_plateau = build_psa(&col, 0.9, 0.01);  // plateau detection on

    // Diagnostics
    eprintln!(
        "[Hatred/Keyword] rows={n}  prefix_len(90pct)={}  prefix_len(plateau)={}",
        psa_90.segments()[0].prefix_len,
        psa_plateau.segments()[0].prefix_len,
    );

    // ── (a) Single equality: "retard"  (from query 17) ─────────────────────
    {
        let needle = "retard";
        verify_eq(&psa_90, &sa, needle);

        let mut g = c.benchmark_group("hatred_keyword_eq");
        g.throughput(Throughput::Elements(n as u64));

        g.bench_function("StringArray",   |b| b.iter(|| criterion::black_box(sa_search_eq(&sa, needle))));
        g.bench_function("PSA_90pct",     |b| b.iter(|| criterion::black_box(search_eq(&psa_90, needle))));
        g.bench_function("PSA_plateau",   |b| b.iter(|| criterion::black_box(search_eq(&psa_plateau, needle))));
        g.finish();
    }

    // ── (b) IN clause: ('fatass','fatso','fattie')  (from query 10) ────────
    {
        let needles = ["fatass", "fatso", "fattie"];
        verify_in(&psa_90, &sa, &needles);

        let mut g = c.benchmark_group("hatred_keyword_in3");
        g.throughput(Throughput::Elements(n as u64));

        g.bench_function("StringArray",   |b| b.iter(|| criterion::black_box(sa_search_in(&sa, &needles))));
        g.bench_function("PSA_90pct",     |b| b.iter(|| criterion::black_box(search_in(&psa_90, &needles))));
        g.bench_function("PSA_plateau",   |b| b.iter(|| criterion::black_box(search_in(&psa_plateau, &needles))));
        g.finish();
    }
}

/// Generico_1 · Anunciante column
/// Query: IN ('BANTRAB/TODOTICKET','TODOTICKET','TODOTICKET.COM')  (from query 1)
fn bench_generico_anunciante(c: &mut Criterion) {
    let col = load_column(GENERICO_CSV, 0, TARGET_ROWS);
    let n = col.len();

    let sa          = build_string_array(&col);
    let psa_90      = build_psa(&col, 0.9, 0.0);
    let psa_plateau = build_psa(&col, 0.9, 0.01);

    eprintln!(
        "[Generico/Anunciante] rows={n}  prefix_len(90pct)={}  prefix_len(plateau)={}",
        psa_90.segments()[0].prefix_len,
        psa_plateau.segments()[0].prefix_len,
    );

    let needles = ["BANTRAB/TODOTICKET", "TODOTICKET", "TODOTICKET.COM"];
    verify_in(&psa_90, &sa, &needles);

    let mut g = c.benchmark_group("generico_anunciante_in3");
    g.throughput(Throughput::Elements(n as u64));

    g.bench_function("StringArray",   |b| b.iter(|| criterion::black_box(sa_search_in(&sa, &needles))));
    g.bench_function("PSA_90pct",     |b| b.iter(|| criterion::black_box(search_in(&psa_90, &needles))));
    g.bench_function("PSA_plateau",   |b| b.iter(|| criterion::black_box(search_in(&psa_plateau, &needles))));
    g.finish();
}

/// Generico_1 · Medio column
/// Single equality: 'RADIO'  (representative filter derived from query patterns)
fn bench_generico_medio(c: &mut Criterion) {
    let col = load_column(GENERICO_CSV, 24, TARGET_ROWS);
    let n = col.len();

    let sa          = build_string_array(&col);
    let psa_90      = build_psa(&col, 0.9, 0.0);
    let psa_plateau = build_psa(&col, 0.9, 0.01);

    eprintln!(
        "[Generico/Medio] rows={n}  prefix_len(90pct)={}  prefix_len(plateau)={}",
        psa_90.segments()[0].prefix_len,
        psa_plateau.segments()[0].prefix_len,
    );

    let needle = "RADIO";
    verify_eq(&psa_90, &sa, needle);

    let mut g = c.benchmark_group("generico_medio_eq");
    g.throughput(Throughput::Elements(n as u64));

    g.bench_function("StringArray",   |b| b.iter(|| criterion::black_box(sa_search_eq(&sa, needle))));
    g.bench_function("PSA_90pct",     |b| b.iter(|| criterion::black_box(search_eq(&psa_90, needle))));
    g.bench_function("PSA_plateau",   |b| b.iter(|| criterion::black_box(search_eq(&psa_plateau, needle))));
    g.finish();
}

// ── Correctness helpers ────────────────────────────────────────────────────────

fn verify_eq(psa: &arrow_prefix_split::PrefixSplitArray, sa: &StringArray, needle: &str) {
    let psa_hits = search_eq(psa, needle);
    let sa_hits  = sa_search_eq(sa, needle);
    assert_eq!(psa_hits, sa_hits, "equality mismatch for needle={needle:?}");
}

fn verify_in(
    psa: &arrow_prefix_split::PrefixSplitArray,
    sa: &StringArray,
    needles: &[&str],
) {
    let psa_hits = search_in(psa, needles);
    let sa_hits  = sa_search_in(sa, needles);
    assert_eq!(
        psa_hits, sa_hits,
        "IN-clause mismatch for needles={needles:?}"
    );
}

// ── Entry point ────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_hatred_keyword,
    bench_generico_anunciante,
    bench_generico_medio,
);
criterion_main!(benches);
