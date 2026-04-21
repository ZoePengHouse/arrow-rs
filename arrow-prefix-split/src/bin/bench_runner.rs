// Licensed under the Apache License, Version 2.0.
//
//! Run the auto-generated benchmark definitions (produced by `column_stats`)
//! against StringArray, PSA-90%, and PSA-plateau, and report timing results.
//!
//! Workflow:
//!   1. Run `column_stats` (with --use-samples or real data) to generate
//!      benchmark JSON files in ./benchmarks/.
//!   2. Run this binary to execute every query in those files and compare
//!      the three designs.
//!
//! # Usage (sample data — works locally)
//!
//!   cargo run -p arrow-prefix-split --bin bench_runner --release -- \
//!       --use-samples --all
//!
//!   cargo run -p arrow-prefix-split --bin bench_runner --release -- \
//!       --use-samples --run bench_01_IGlocations2_IGlocations2_1_caption
//!
//! # Usage (real data — run on the server)
//!
//!   cargo run -p arrow-prefix-split --bin bench_runner --release -- \
//!       --data /path/to/public_bi_benchmark --all
//!
//! # Options
//!
//!   --data           PATH  path to public_bi_benchmark (default: local path)
//!   --benchmarks-dir DIR   directory containing benchmark JSON files (default: ./benchmarks)
//!   --run            NAME  run only the benchmark whose filename stem matches NAME
//!   --all                  run all benchmark files found in --benchmarks-dir  (default)
//!   --use-samples          read column data from samples/*.sample.csv
//!   --iters          N     timing iterations per query (default: 20 real / 200 samples)
//!   --csv            FILE  write results to a CSV file

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use arrow_array::{Array, StringArray};
use arrow_prefix_split::{
    PrefixSplitBuilder, PrefixSplitConfig, search_eq, search_in, search_ne, search_range,
};
use serde::{Deserialize, Serialize};

// ── Benchmark def (mirrors column_stats's BenchmarkDef) ───────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
struct QueryDef {
    id:         usize,
    #[serde(rename = "type")]
    query_type: String,
    label:      String,
    values:     Vec<String>,
    #[serde(default)]
    low_inc:    bool,
    #[serde(default)]
    high_inc:   bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct BenchmarkDef {
    rank:       usize,
    benchmark:  String,
    table:      String,
    column:     String,
    col_idx:    usize,
    avg_len:    f64,
    prefix_len: usize,
    psa_score:  f64,
    queries:    Vec<QueryDef>,
}

// ── CLI ───────────────────────────────────────────────────────────────────────

struct Config {
    data_root:      PathBuf,
    benchmarks_dir: PathBuf,
    run_name:       Option<String>, // None = all
    use_samples:    bool,
    iters:          usize,
    csv_out:        Option<PathBuf>,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().skip(1).collect();

    fn flag_val<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
        args.windows(2).find(|w| w[0] == flag).map(|w| w[1].as_str())
    }
    fn has_flag(args: &[String], flag: &str) -> bool {
        args.iter().any(|a| a == flag)
    }

    let data = flag_val(&args, "--data")
        .unwrap_or("/Users/zixpeng/Documents/Research/CS681/public_bi_benchmark");

    let use_samples  = has_flag(&args, "--use-samples");
    let default_iters = if use_samples { 200 } else { 20 };
    let iters = flag_val(&args, "--iters")
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_iters);

    // --run NAME means only run the file whose stem matches NAME.
    // --all (or neither flag) means run everything.
    let run_name = if has_flag(&args, "--all") {
        None
    } else {
        flag_val(&args, "--run").map(str::to_string)
    };

    Config {
        data_root:      PathBuf::from(data),
        benchmarks_dir: PathBuf::from(flag_val(&args, "--benchmarks-dir").unwrap_or("benchmarks")),
        run_name,
        use_samples,
        iters,
        csv_out: flag_val(&args, "--csv").map(PathBuf::from),
    }
}

// ── Data loading ──────────────────────────────────────────────────────────────

/// Construct the CSV file path for a benchmark definition.
fn data_path(cfg: &Config, def: &BenchmarkDef) -> Option<PathBuf> {
    let bench_dir = cfg.data_root.join("benchmark").join(&def.benchmark);

    if cfg.use_samples {
        // Try <table>.sample.csv directly, then <table>_1.sample.csv, etc.
        let samples_dir = bench_dir.join("samples");
        let exact = samples_dir.join(format!("{}.sample.csv", def.table));
        if exact.exists() { return Some(exact); }
        // Glob: any file whose stem (without .sample) starts with def.table
        if let Ok(entries) = std::fs::read_dir(&samples_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "csv") {
                    let stem = path.file_stem().unwrap().to_string_lossy();
                    let tname = stem.trim_end_matches(".sample");
                    if tname == def.table || tname.starts_with(&format!("{}_", def.table)) {
                        return Some(path);
                    }
                }
            }
        }
    } else {
        let data_dir = bench_dir.join("data");
        let exact = data_dir.join(format!("{}.csv", def.table));
        if exact.exists() { return Some(exact); }
        if let Ok(entries) = std::fs::read_dir(&data_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "csv") {
                    let stem = path.file_stem().unwrap().to_string_lossy().to_string();
                    if stem == def.table || stem.starts_with(&format!("{}_", def.table)) {
                        return Some(path);
                    }
                }
            }
        }
    }
    None
}

/// Load one column from a pipe-delimited CSV. Returns None per null/empty field.
fn load_column(path: &Path, col_idx: usize) -> Vec<Option<String>> {
    use std::io::{BufRead, BufReader};
    let file = match std::fs::File::open(path) {
        Ok(f)  => f,
        Err(e) => { eprintln!("  [warn] Cannot read {}: {}", path.display(), e); return Vec::new(); }
    };
    BufReader::new(file).lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let field = line.split('|').nth(col_idx).unwrap_or("").trim().to_string();
            if field.is_empty() || field.eq_ignore_ascii_case("null") { None } else { Some(field) }
        })
        .collect()
}

// ── Array builders ────────────────────────────────────────────────────────────

fn build_sa(col: &[Option<String>]) -> StringArray {
    StringArray::from(col.iter().map(|v| v.as_deref()).collect::<Vec<_>>())
}

fn build_psa(
    col:       &[Option<String>],
    threshold: f64,
    plateau:   f64,
) -> arrow_prefix_split::PrefixSplitArray {
    let cfg = PrefixSplitConfig {
        segment_size:                 1_000,
        prefix_distinguish_threshold: threshold,
        max_prefix_len:               64,
        plateau_min_gain_fraction:    plateau,
        plateau_min_progress:         0.5,
    };
    let mut b = PrefixSplitBuilder::with_config(cfg);
    for v in col { b.append_option(v.as_deref()); }
    b.finish()
}

// ── Prefix stats ──────────────────────────────────────────────────────────────

struct PrefixStats { min: usize, median: usize, max: usize }

fn prefix_stats(psa: &arrow_prefix_split::PrefixSplitArray) -> PrefixStats {
    let mut lens: Vec<usize> = psa.segments()
        .iter()
        .filter(|s| s.sealed)
        .map(|s| s.prefix_len)
        .collect();
    if lens.is_empty() { return PrefixStats { min: 0, median: 0, max: 0 }; }
    lens.sort_unstable();
    PrefixStats {
        min:    *lens.first().unwrap(),
        median: lens[lens.len() / 2],
        max:    *lens.last().unwrap(),
    }
}

// ── Timing ────────────────────────────────────────────────────────────────────

fn time_fn<F: Fn() -> usize>(f: F, iters: usize) -> Duration {
    let warmup = (iters / 5).max(3);
    for _ in 0..warmup { std::hint::black_box(f()); }
    let mut times: Vec<Duration> = (0..iters).map(|_| {
        let t = Instant::now();
        std::hint::black_box(f());
        t.elapsed()
    }).collect();
    times.sort();
    times[times.len() / 2]
}

// ── StringArray search helpers ────────────────────────────────────────────────

fn sa_search_eq(sa: &StringArray, needle: &str) -> usize {
    (0..sa.len()).filter(|&i| !sa.is_null(i) && sa.value(i) == needle).count()
}
fn sa_search_ne(sa: &StringArray, needle: &str) -> usize {
    (0..sa.len()).filter(|&i| !sa.is_null(i) && sa.value(i) != needle).count()
}
fn sa_search_in(sa: &StringArray, needles: &[String]) -> usize {
    (0..sa.len())
        .filter(|&i| !sa.is_null(i) && needles.iter().any(|n| sa.value(i) == n.as_str()))
        .count()
}
fn sa_search_range(sa: &StringArray, low: &str, low_inc: bool, high: &str, high_inc: bool) -> usize {
    (0..sa.len()).filter(|&i| {
        if sa.is_null(i) { return false; }
        let v = sa.value(i);
        let ok_lo = low.is_empty()  || if low_inc  { v >= low  } else { v > low  };
        let ok_hi = high.is_empty() || if high_inc { v <= high } else { v < high };
        ok_lo && ok_hi
    }).count()
}

// ── PSA search helpers ────────────────────────────────────────────────────────

fn psa_search_in(psa: &arrow_prefix_split::PrefixSplitArray, needles: &[String]) -> usize {
    let refs: Vec<&str> = needles.iter().map(String::as_str).collect();
    search_in(psa, &refs).len()
}
fn psa_search_range(
    psa: &arrow_prefix_split::PrefixSplitArray,
    low: &str, low_inc: bool, high: &str, high_inc: bool,
) -> usize {
    search_range(psa, low, low_inc, high, high_inc).len()
}

// ── Result record ─────────────────────────────────────────────────────────────

struct QueryResult {
    query_id:   usize,
    label:      String,
    query_type: String,
    matches:    usize,
    /// All timings stored in nanoseconds so sub-µs sample runs show real numbers.
    sa_ns:      u128,
    psa90_ns:   u128,
    psapl_ns:   u128,
}

impl QueryResult {
    fn speedup_90(&self) -> f64 {
        if self.psa90_ns == 0 { 1.0 } else { self.sa_ns as f64 / self.psa90_ns as f64 }
    }
    fn speedup_pl(&self) -> f64 {
        if self.psapl_ns == 0 { 1.0 } else { self.sa_ns as f64 / self.psapl_ns as f64 }
    }
    /// Format a nanosecond value as "42 ns", "1.3 µs", "2.5 ms" automatically.
    fn fmt_time(ns: u128) -> String {
        if ns < 1_000 {
            format!("{} ns", ns)
        } else if ns < 1_000_000 {
            format!("{:.1} µs", ns as f64 / 1_000.0)
        } else {
            format!("{:.2} ms", ns as f64 / 1_000_000.0)
        }
    }
}

// ── Core: run one benchmark def ───────────────────────────────────────────────

fn run_one(def: &BenchmarkDef, cfg: &Config) -> (Vec<QueryResult>, PrefixStats, PrefixStats, usize) {
    let path = match data_path(cfg, def) {
        Some(p) => p,
        None => {
            eprintln!(
                "  [bench {}] Cannot find data file for {}/{} ({}mode)",
                def.rank, def.benchmark, def.table,
                if cfg.use_samples { "sample " } else { "" }
            );
            let empty_ps = PrefixStats { min: 0, median: 0, max: 0 };
            return (vec![], empty_ps, PrefixStats { min: 0, median: 0, max: 0 }, 0);
        }
    };

    eprint!("  Loading {}/{} col {} … ", def.table, path.file_name().unwrap().to_string_lossy(), def.col_idx);
    let col = load_column(&path, def.col_idx);
    if col.is_empty() {
        eprintln!("(empty)");
        let empty_ps = PrefixStats { min: 0, median: 0, max: 0 };
        return (vec![], empty_ps, PrefixStats { min: 0, median: 0, max: 0 }, 0);
    }
    eprintln!("{} rows", col.len());

    let sa    = build_sa(&col);
    let psa90 = build_psa(&col, 0.9, 0.0);
    let psapl = build_psa(&col, 0.9, 0.01);
    let p90   = prefix_stats(&psa90);
    let ppl   = prefix_stats(&psapl);
    let iters = cfg.iters;
    let nrows = col.len();

    let mut results = Vec::new();

    for q in &def.queries {
        let (matches, t_sa, t_90, t_pl) = match q.query_type.as_str() {
            "Eq" => {
                let v = &q.values[0];
                let m  = sa_search_eq(&sa, v);
                let ts = time_fn(|| sa_search_eq(&sa, v), iters);
                let t9 = time_fn(|| search_eq(&psa90, v).len(), iters);
                let tp = time_fn(|| search_eq(&psapl, v).len(), iters);
                (m, ts, t9, tp)
            }
            "Ne" => {
                let v = &q.values[0];
                let m  = sa_search_ne(&sa, v);
                let ts = time_fn(|| sa_search_ne(&sa, v), iters);
                let t9 = time_fn(|| search_ne(&psa90, v).len(), iters);
                let tp = time_fn(|| search_ne(&psapl, v).len(), iters);
                (m, ts, t9, tp)
            }
            "In" => {
                let vals = q.values.clone();
                let m  = sa_search_in(&sa, &vals);
                let ts = time_fn(|| sa_search_in(&sa, &vals), iters);
                let t9 = time_fn(|| psa_search_in(&psa90, &vals), iters);
                let tp = time_fn(|| psa_search_in(&psapl, &vals), iters);
                (m, ts, t9, tp)
            }
            "Range" => {
                let lo = &q.values[0];
                let hi = &q.values[1];
                let li = q.low_inc;
                let hi_inc = q.high_inc;
                let m  = sa_search_range(&sa, lo, li, hi, hi_inc);
                let ts = time_fn(|| sa_search_range(&sa, lo, li, hi, hi_inc), iters);
                let t9 = time_fn(|| psa_search_range(&psa90, lo, li, hi, hi_inc), iters);
                let tp = time_fn(|| psa_search_range(&psapl, lo, li, hi, hi_inc), iters);
                (m, ts, t9, tp)
            }
            other => {
                eprintln!("  [warn] Unknown query type '{}' in query {}", other, q.id);
                continue;
            }
        };

        results.push(QueryResult {
            query_id:   q.id,
            label:      q.label.clone(),
            query_type: q.query_type.clone(),
            matches,
            sa_ns:    t_sa.as_nanos(),
            psa90_ns: t_90.as_nanos(),
            psapl_ns: t_pl.as_nanos(),
        });
    }

    (results, p90, ppl, nrows)
}

// ── Output ────────────────────────────────────────────────────────────────────

/// Print one benchmark as a speedup table.
///
/// Layout (each row = one query):
///
///   Rank 1 · IGlocations2 / IGlocations2_1 / caption
///   20 rows · avg_len=154.7 · P_len=3 · score=142.2
///   Prefix stats (sealed segs): PSA-90%  0/0/0 · PSA-PL  0/0/0
///
///    #  Type      Query                              Matches  SA time   ×PSA-90%  ×PSA-PL
///   ─────────────────────────────────────────────────────────────────────────────────────
///    1  EQ        '#earlymorning #Beach…'                  2   420 ns     1.40x    1.41x
///    2  EQ        'Growing up on these…'                   1   385 ns     1.35x    1.36x
///   ...
///   ─────────────────────────────────────────────────────────────────────────────────────
///      PSA-90%: wins 7/10 queries · avg speedup 1.22x
///      PSA-PL:  wins 7/10 queries · avg speedup 1.23x
///
/// The ×PSA-90% / ×PSA-PL columns show SA_time / PSA_time:
///   > 1.0x  → PSA is faster  (good)
///   = 1.0x  → same speed
///   < 1.0x  → PSA is slower  (overhead exceeds saving)
fn print_benchmark_results(
    def:     &BenchmarkDef,
    results: &[QueryResult],
    p90:     &PrefixStats,
    ppl:     &PrefixStats,
    rows:    usize,
) {
    if results.is_empty() { return; }

    // ── Header ────────────────────────────────────────────────────────────────
    println!();
    println!("  Rank {:>2} · {} / {} / {}",
        def.rank, def.benchmark, def.table, def.column);
    println!("  {} rows · avg_len={:.1} · P_len={} · score={:.1}",
        rows, def.avg_len, def.prefix_len, def.psa_score);
    println!("  Prefix stats (sealed segs): PSA-90%  {}/{}/{}  ·  PSA-PL  {}/{}/{}",
        p90.min, p90.median, p90.max,
        ppl.min, ppl.median, ppl.max);
    println!();

    // ── Column headers ────────────────────────────────────────────────────────
    //   #  Type      Query                              Matches  SA time   ×PSA-90%  ×PSA-PL
    println!("  {:>2}  {:<8}  {:<35}  {:>7}  {:>8}  {:>9}  {:>8}",
        "#", "Type", "Query", "Matches", "SA time", "×PSA-90%", "×PSA-PL");
    println!("  {}", "─".repeat(85));

    // ── One row per query ─────────────────────────────────────────────────────
    for r in results {
        // Speedup = SA_time / PSA_time.  >1 means PSA wins.
        let fmt_sp = |sp: f64| -> String {
            if r.sa_ns == 0 {
                "  —   ".into()
            } else {
                // Colour-code direction: show a + marker when PSA wins.
                let marker = if sp > 1.05 { "▲" } else if sp < 0.95 { "▼" } else { " " };
                format!("{}{:.2}x", marker, sp)
            }
        };

        println!("  {:>2}  {:<8}  {:<35}  {:>7}  {:>8}  {:>9}  {:>8}",
            r.query_id,
            r.query_type,
            trunc(&r.label, 35),
            r.matches,
            QueryResult::fmt_time(r.sa_ns),
            fmt_sp(r.speedup_90()),
            fmt_sp(r.speedup_pl()),
        );
    }

    // ── Per-benchmark summary ─────────────────────────────────────────────────
    println!("  {}", "─".repeat(85));

    let wins90 = results.iter().filter(|r| r.psa90_ns < r.sa_ns).count();
    let winspl = results.iter().filter(|r| r.psapl_ns < r.sa_ns).count();
    let avg90  = results.iter().map(|r| r.speedup_90()).sum::<f64>() / results.len() as f64;
    let avgpl  = results.iter().map(|r| r.speedup_pl()).sum::<f64>() / results.len() as f64;

    println!("  PSA-90%: wins {}/{} queries · avg speedup {:.2}x",
        wins90, results.len(), avg90);
    println!("  PSA-PL:  wins {}/{} queries · avg speedup {:.2}x",
        winspl, results.len(), avgpl);
    println!("  (▲ = PSA faster  ▼ = PSA slower  SA time = StringArray baseline)");
}

fn print_global_summary(all: &[(BenchmarkDef, Vec<QueryResult>)]) {
    if all.is_empty() { return; }

    let total_queries: usize = all.iter().map(|(_, rs)| rs.len()).sum();
    let wins90: usize = all.iter().flat_map(|(_, rs)| rs)
        .filter(|r| r.psa90_ns < r.sa_ns).count();
    let winspl: usize = all.iter().flat_map(|(_, rs)| rs)
        .filter(|r| r.psapl_ns < r.sa_ns).count();
    let avg90: f64 = all.iter().flat_map(|(_, rs)| rs)
        .map(|r| r.speedup_90()).sum::<f64>() / total_queries as f64;
    let avgpl: f64 = all.iter().flat_map(|(_, rs)| rs)
        .map(|r| r.speedup_pl()).sum::<f64>() / total_queries as f64;

    // Per-type breakdown.
    let by_type = |qt: &str| -> (usize, usize, f64, usize, f64) {
        let qs: Vec<&QueryResult> = all.iter()
            .flat_map(|(_, rs)| rs)
            .filter(|r| r.query_type == qt)
            .collect();
        let n = qs.len();
        if n == 0 { return (0, 0, 0.0, 0, 0.0); }
        let w90 = qs.iter().filter(|r| r.psa90_ns < r.sa_ns).count();
        let wpl = qs.iter().filter(|r| r.psapl_ns < r.sa_ns).count();
        let a90 = qs.iter().map(|r| r.speedup_90()).sum::<f64>() / n as f64;
        let apl = qs.iter().map(|r| r.speedup_pl()).sum::<f64>() / n as f64;
        (n, w90, a90, wpl, apl)
    };

    println!();
    println!("{}", "═".repeat(72));
    println!("  GLOBAL SUMMARY  —  {} benchmarks · {} queries total",
        all.len(), total_queries);
    println!("{}", "═".repeat(72));
    println!();
    println!("  {:<8}  {:>7}  {:>9}  {:>9}  {:>9}  {:>9}",
        "Type", "Queries", "90% wins", "90% avg", "PL wins", "PL avg");
    println!("  {}", "─".repeat(60));

    for qt in &["Eq", "Ne", "In", "Range"] {
        let (n, w90, a90, wpl, apl) = by_type(qt);
        if n == 0 { continue; }
        println!("  {:<8}  {:>7}  {:>9}  {:>8.2}x  {:>9}  {:>8.2}x",
            qt, n, w90, a90, wpl, apl);
    }

    println!("  {}", "─".repeat(60));
    println!("  {:<8}  {:>7}  {:>9}  {:>8.2}x  {:>9}  {:>8.2}x",
        "TOTAL", total_queries, wins90, avg90, winspl, avgpl);
    println!();
    println!("{}", "═".repeat(72));
    println!();
    println!("  SA = StringArray baseline.  Speedup = SA_time / PSA_time.");
    println!("  >1.0x means PSA is faster;  <1.0x means PSA has overhead.");
    println!();
}

// ── CSV output ────────────────────────────────────────────────────────────────

fn write_csv(all: &[(BenchmarkDef, Vec<QueryResult>)], path: &Path) -> std::io::Result<()> {
    use std::fmt::Write as FmtWrite;
    let mut out = String::new();
    writeln!(out,
        "rank,benchmark,table,column,col_idx,avg_len,prefix_len,psa_score,\
         query_id,query_type,label,matches,sa_ns,psa90_ns,psapl_ns,speedup_90,speedup_pl"
    ).unwrap();
    for (def, results) in all {
        for r in results {
            writeln!(out,
                "{},{},{},{},{},{:.1},{},{:.1},{},{},{},{},{},{},{},{:.4},{:.4}",
                def.rank, def.benchmark, def.table,
                csv_escape(&def.column), def.col_idx,
                def.avg_len, def.prefix_len, def.psa_score,
                r.query_id, r.query_type,
                csv_escape(&r.label),
                r.matches, r.sa_ns, r.psa90_ns, r.psapl_ns,
                r.speedup_90(), r.speedup_pl(),
            ).unwrap();
        }
    }
    std::fs::write(path, out)
}

fn trunc(s: &str, n: usize) -> String {
    if s.chars().count() <= n { s.to_string() }
    else {
        let end = s.char_indices().nth(n - 1).map_or(s.len(), |(i, _)| i);
        format!("{}…", &s[..end])
    }
}
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else { s.to_string() }
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let cfg = parse_args();

    // ── Load benchmark definition files ──────────────────────────────────────
    if !cfg.benchmarks_dir.exists() {
        eprintln!("Error: benchmarks dir not found: {}", cfg.benchmarks_dir.display());
        eprintln!("Run `column_stats` first to generate benchmark definition files.");
        std::process::exit(1);
    }

    let mut def_paths: Vec<PathBuf> = std::fs::read_dir(&cfg.benchmarks_dir)
        .expect("Cannot read benchmarks dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |e| e == "json"))
        .collect();
    def_paths.sort();

    // Filter by --run NAME if specified.
    if let Some(ref name) = cfg.run_name {
        def_paths.retain(|p| {
            p.file_stem()
                .map(|s| s.to_string_lossy().contains(name.as_str()))
                .unwrap_or(false)
        });
    }

    if def_paths.is_empty() {
        eprintln!("No benchmark files found in {}{}",
            cfg.benchmarks_dir.display(),
            cfg.run_name.as_deref().map(|n| format!(" matching '{n}'")).unwrap_or_default()
        );
        std::process::exit(1);
    }

    let mode = if cfg.use_samples { "sample files (--use-samples)" } else { "real data files" };
    println!("Mode:    {mode}");
    println!("Iters:   {} timing iterations per query", cfg.iters);
    println!("Designs: StringArray | PSA-90% | PSA-plateau(1%)");
    println!("Found {} benchmark file(s)\n", def_paths.len());

    // ── Run each benchmark ────────────────────────────────────────────────────
    let mut all: Vec<(BenchmarkDef, Vec<QueryResult>)> = Vec::new();

    for path in &def_paths {
        let json = match std::fs::read_to_string(path) {
            Ok(s)  => s,
            Err(e) => { eprintln!("Cannot read {}: {}", path.display(), e); continue; }
        };
        let def: BenchmarkDef = match serde_json::from_str(&json) {
            Ok(d)  => d,
            Err(e) => { eprintln!("Cannot parse {}: {}", path.display(), e); continue; }
        };

        eprintln!("Running rank {:>2}: {}/{}/{} …",
            def.rank, def.benchmark, def.table, def.column);

        let (results, p90, ppl, nrows) = run_one(&def, &cfg);
        print_benchmark_results(&def, &results, &p90, &ppl, nrows);
        all.push((def, results));
    }

    // ── Global summary ────────────────────────────────────────────────────────
    if all.len() > 1 {
        print_global_summary(&all);
    }

    // ── CSV output ────────────────────────────────────────────────────────────
    if let Some(csv_path) = &cfg.csv_out {
        write_csv(&all, csv_path).expect("Failed to write CSV");
        println!("Results written to {}", csv_path.display());
    }
}
