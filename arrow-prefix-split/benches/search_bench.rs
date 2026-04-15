// Licensed under the Apache License, Version 2.0.

//! Benchmark: prefix-split search vs. standard Arrow StringArray linear scan.
//!
//! Run with:
//!   cargo bench -p arrow-prefix-split
//!
//! # What is measured
//!
//! For a synthetic column of N string rows (default 100 000) with realistic
//! prefix structure (many strings sharing the first 4 bytes), we compare:
//!
//! * `psa_search_eq`  – `arrow_prefix_split::search_eq`  (prefix-first)
//! * `sa_linear_scan` – iterate `StringArray` and compare each element
//!
//! Both return the same set of matching indices.  The benchmark reports
//! throughput (rows / ns ≈ rows / second) so that results remain comparable
//! across machines.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use arrow_array::{Array, StringArray};
use arrow_prefix_split::{PrefixSplitBuilder, PrefixSplitConfig, search_eq};

fn make_psa(strings: &[&str]) -> arrow_prefix_split::PrefixSplitArray {
    make_psa_with_threshold(strings, 0.9)
}

fn make_psa_with_threshold(
    strings: &[&str],
    threshold: f64,
) -> arrow_prefix_split::PrefixSplitArray {
    let cfg = PrefixSplitConfig {
        segment_size: 1_000,
        prefix_distinguish_threshold: threshold,
        max_prefix_len: 64,
        ..PrefixSplitConfig::default()
    };
    let mut b = PrefixSplitBuilder::with_config(cfg);
    for s in strings {
        b.append_value(s);
    }
    b.finish()
}

// ── Dataset construction ──────────────────────────────────────────────────────

/// Generates `n` strings that share a 4-byte prefix ("data") and vary after
/// that.  Every `freq`-th string is the `needle`.
fn make_dataset(n: usize, needle: &str, freq: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            if i % freq == 0 {
                needle.to_string()
            } else {
                format!("data{i:08}_extra_suffix")
            }
        })
        .collect()
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

fn bench_search(c: &mut Criterion) {
    let n = 100_000usize;
    let needle = "data_target_value";
    let freq = 200; // one match every 200 rows → 500 matches total

    // ── Build StringArray ─────────────────────────────────────────────────
    let dataset = make_dataset(n, needle, freq);
    let string_refs: Vec<&str> = dataset.iter().map(String::as_str).collect();
    let sa = StringArray::from(string_refs.clone());

    // ── Build PrefixSplitArray ────────────────────────────────────────────
    let psa = make_psa(&string_refs);

    // ── Verify both approaches agree ──────────────────────────────────────
    let psa_hits = search_eq(&psa, needle);
    let sa_hits: Vec<usize> = (0..sa.len())
        .filter(|&i| !sa.is_null(i) && sa.value(i) == needle)
        .collect();
    assert_eq!(
        psa_hits, sa_hits,
        "PrefixSplitArray and StringArray disagree on search results"
    );

    // ── Criterion group ───────────────────────────────────────────────────
    let mut group = c.benchmark_group("equality_search");
    group.throughput(Throughput::Elements(n as u64));

    group.bench_with_input(
        BenchmarkId::new("PrefixSplitArray", n),
        &n,
        |bench, _| {
            bench.iter(|| {
                let hits = search_eq(&psa, needle);
                criterion::black_box(hits)
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("StringArray_linear", n),
        &n,
        |bench, _| {
            bench.iter(|| {
                let hits: Vec<usize> = (0..sa.len())
                    .filter(|&i| !sa.is_null(i) && sa.value(i) == needle)
                    .collect();
                criterion::black_box(hits)
            });
        },
    );

    group.finish();
}

/// Vary the dataset size to observe scaling behaviour.
fn bench_scaling(c: &mut Criterion) {
    let needle = "data_target_value";
    let freq = 100;

    let mut group = c.benchmark_group("search_scaling");

    for &n in &[10_000usize, 50_000, 100_000, 500_000] {
        let dataset = make_dataset(n, needle, freq);
        let string_refs: Vec<&str> = dataset.iter().map(String::as_str).collect();
        let sa = StringArray::from(string_refs.clone());
        let psa = make_psa(&string_refs);

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(
            BenchmarkId::new("PrefixSplitArray", n),
            &n,
            |bench, _| bench.iter(|| criterion::black_box(search_eq(&psa, needle))),
        );

        group.bench_with_input(
            BenchmarkId::new("StringArray_linear", n),
            &n,
            |bench, _| {
                bench.iter(|| {
                    let hits: Vec<usize> = (0..sa.len())
                        .filter(|&i| !sa.is_null(i) && sa.value(i) == needle)
                        .collect();
                    criterion::black_box(hits)
                })
            },
        );
    }

    group.finish();
}

/// Same-length scenario: every haystack string is the same length as the needle.
///
/// This is the case where the old design's `lengths[i]` filter gave NO benefit
/// (all lengths equal, so no early-out).  The flag-based design avoids the
/// `lengths[]` random-access entirely, relying on the suffix-length check instead.
fn bench_same_length(c: &mut Criterion) {
    let n = 100_000usize;
    // All strings are exactly 17 bytes ("data_target_value" is also 17 bytes).
    let needle = "data_target_value";
    let freq = 200;

    // Haystack: 17-byte strings that share the "data" prefix but differ after.
    // needle matches every freq-th row.
    let dataset: Vec<String> = (0..n)
        .map(|i| {
            if i % freq == 0 {
                needle.to_string()
            } else {
                // Exactly 17 bytes, shares "data_" prefix, differs in suffix.
                format!("data_{:012}", i)
            }
        })
        .collect();

    let string_refs: Vec<&str> = dataset.iter().map(String::as_str).collect();
    let sa = StringArray::from(string_refs.clone());
    let psa = make_psa(&string_refs);

    // Verify agreement.
    let psa_hits = search_eq(&psa, needle);
    let sa_hits: Vec<usize> = (0..sa.len())
        .filter(|&i| !sa.is_null(i) && sa.value(i) == needle)
        .collect();
    assert_eq!(psa_hits, sa_hits, "same-length: PSA and SA disagree");

    let mut group = c.benchmark_group("same_length_search");
    group.throughput(Throughput::Elements(n as u64));

    group.bench_with_input(
        BenchmarkId::new("PrefixSplitArray", n),
        &n,
        |bench, _| bench.iter(|| criterion::black_box(search_eq(&psa, needle))),
    );

    group.bench_with_input(
        BenchmarkId::new("StringArray_linear", n),
        &n,
        |bench, _| {
            bench.iter(|| {
                let hits: Vec<usize> = (0..sa.len())
                    .filter(|&i| !sa.is_null(i) && sa.value(i) == needle)
                    .collect();
                criterion::black_box(hits)
            })
        },
    );

    group.finish();
}

/// Sweep threshold values 80/85/90/95 % on both the different-length and
/// same-length datasets to find the best setting.
fn bench_threshold_sweep(c: &mut Criterion) {
    let n = 100_000usize;
    let needle = "data_target_value";
    let freq = 200;

    // Different-length dataset: haystack strings are 25 bytes, needle is 17.
    let diff_dataset = make_dataset(n, needle, freq);
    let diff_refs: Vec<&str> = diff_dataset.iter().map(String::as_str).collect();

    // Same-length dataset: every string is exactly 17 bytes.
    let same_dataset: Vec<String> = (0..n)
        .map(|i| {
            if i % freq == 0 {
                needle.to_string()
            } else {
                format!("data_{:012}", i)
            }
        })
        .collect();
    let same_refs: Vec<&str> = same_dataset.iter().map(String::as_str).collect();

    // ── Different-length sweep ────────────────────────────────────────────
    {
        let mut group = c.benchmark_group("threshold_diff_len");
        group.throughput(Throughput::Elements(n as u64));

        for &(pct, t) in &[(80u32, 0.80f64), (85, 0.85), (90, 0.90), (95, 0.95)] {
            let psa = make_psa_with_threshold(&diff_refs, t);
            // Sanity check: report the prefix_len chosen for the first segment.
            let p = psa.segments()[0].prefix_len;
            eprintln!("diff-len  threshold={pct}%  prefix_len={p}");
            group.bench_with_input(
                BenchmarkId::new("PSA", format!("{pct}pct")),
                &pct,
                |bench, _| bench.iter(|| criterion::black_box(search_eq(&psa, needle))),
            );
        }
        group.finish();
    }

    // ── Same-length sweep ─────────────────────────────────────────────────
    {
        let mut group = c.benchmark_group("threshold_same_len");
        group.throughput(Throughput::Elements(n as u64));

        for &(pct, t) in &[(80u32, 0.80f64), (85, 0.85), (90, 0.90), (95, 0.95)] {
            let psa = make_psa_with_threshold(&same_refs, t);
            let p = psa.segments()[0].prefix_len;
            eprintln!("same-len  threshold={pct}%  prefix_len={p}");
            group.bench_with_input(
                BenchmarkId::new("PSA", format!("{pct}pct")),
                &pct,
                |bench, _| bench.iter(|| criterion::black_box(search_eq(&psa, needle))),
            );
        }
        group.finish();
    }
}

/// Plateau-detection benchmark.
///
/// Uses a low-diversity dataset modelled on URL paths:
///   "https://example.com/users/<uid>/profile"
///
/// All strings share a 28-byte prefix ("https://example.com/users/").  The
/// distinctness curve rises fast in the first ~10 bytes (the shared scheme +
/// host) then flatlines completely once the prefix covers the whole shared
/// portion.  Plateau detection should stop much earlier than the 90% threshold
/// alone would, yielding a shorter prefix slab and denser cache lines.
fn bench_plateau(c: &mut Criterion) {
    let n = 100_000usize;
    // needle is one specific URL that appears every freq rows.
    let needle = "https://example.com/users/00000/profile";
    let freq = 200;

    let dataset: Vec<String> = (0..n)
        .map(|i| {
            if i % freq == 0 {
                needle.to_string()
            } else {
                format!("https://example.com/users/{:05}/profile", i % 100_000)
            }
        })
        .collect();
    let string_refs: Vec<&str> = dataset.iter().map(String::as_str).collect();
    let sa = StringArray::from(string_refs.clone());

    // Build with plateau detection enabled (default config).
    let psa_plateau = make_psa_with_threshold(&string_refs, 0.9);

    // Build with plateau detection disabled (fraction = 0.0).
    let psa_no_plateau = {
        let cfg = PrefixSplitConfig {
            segment_size: 1_000,
            prefix_distinguish_threshold: 0.9,
            max_prefix_len: 64,
            plateau_min_gain_fraction: 0.0,
            plateau_min_progress: 0.0,
        };
        let mut b = PrefixSplitBuilder::with_config(cfg);
        for s in &string_refs {
            b.append_value(s);
        }
        b.finish()
    };

    let p_with    = psa_plateau.segments()[0].prefix_len;
    let p_without = psa_no_plateau.segments()[0].prefix_len;
    eprintln!("plateau: prefix_len WITH detection={p_with}, WITHOUT={p_without}");

    // Verify correctness.
    let hits_plateau    = search_eq(&psa_plateau, needle);
    let hits_no_plateau = search_eq(&psa_no_plateau, needle);
    let sa_hits: Vec<usize> = (0..sa.len())
        .filter(|&i| !sa.is_null(i) && sa.value(i) == needle)
        .collect();
    assert_eq!(hits_plateau,    sa_hits, "plateau PSA disagrees with SA");
    assert_eq!(hits_no_plateau, sa_hits, "no-plateau PSA disagrees with SA");

    let mut group = c.benchmark_group("plateau_detection");
    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("PSA_plateau_on", |bench| {
        bench.iter(|| criterion::black_box(search_eq(&psa_plateau, needle)))
    });
    group.bench_function("PSA_plateau_off", |bench| {
        bench.iter(|| criterion::black_box(search_eq(&psa_no_plateau, needle)))
    });
    group.bench_function("StringArray_linear", |bench| {
        bench.iter(|| {
            let hits: Vec<usize> = (0..sa.len())
                .filter(|&i| !sa.is_null(i) && sa.value(i) == needle)
                .collect();
            criterion::black_box(hits)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_search, bench_scaling, bench_same_length, bench_threshold_sweep, bench_plateau);
criterion_main!(benches);
