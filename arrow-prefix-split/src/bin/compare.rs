// Licensed under the Apache License, Version 2.0.
//
//! Compare StringArray vs PrefixSplitArray (90 % threshold) vs PrefixSplitArray
//! (plateau detection) on real Public BI Benchmark datasets.
//!
//! # Usage (server — real data in data/ directories)
//!
//!   cargo run -p arrow-prefix-split --bin compare --release -- --all
//!   cargo run -p arrow-prefix-split --bin compare --release -- --benchmarks Hatred,Generico
//!
//! # Usage (local — sample data, to verify the code works)
//!
//!   cargo run -p arrow-prefix-split --bin compare --release -- --all --use-samples
//!   cargo run -p arrow-prefix-split --bin compare --release -- --benchmarks Hatred,Generico --use-samples
//!
//! # Options
//!
//!   --data    PATH   path to the public_bi_benchmark directory
//!                    (default: /Users/zixpeng/Documents/Research/CS681/public_bi_benchmark)
//!   --benchmarks A,B run only the listed benchmarks (comma-separated)
//!   --all            run all benchmarks found under benchmark/
//!   --use-samples    read from samples/*.sample.csv instead of data/*.csv
//!                    (for local testing when real data is not present)
//!   --iters   N      timing iterations per filter (default: 20 for real data,
//!                    200 when --use-samples because sample files are tiny)
//!   --csv     FILE   write results to a CSV file in addition to stdout

use std::collections::{HashMap, HashSet};
use std::fmt::Write as FmtWrite;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use arrow_array::{Array, StringArray};
use arrow_prefix_split::{
    PrefixSplitBuilder, PrefixSplitConfig, search_eq, search_in, search_ne, search_range,
};

// ── CLI ───────────────────────────────────────────────────────────────────────

struct Config {
    benchmark_root: PathBuf,
    benchmarks:     Option<Vec<String>>, // None = all
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
    let benchmark_root = PathBuf::from(data).join("benchmark");

    let benchmarks = if has_flag(&args, "--all") {
        None
    } else if let Some(v) = flag_val(&args, "--benchmarks") {
        Some(v.split(',').map(str::trim).map(String::from).collect())
    } else {
        Some(vec!["Hatred".into(), "Generico".into(), "CityMaxCapita".into()])
    };

    let use_samples = has_flag(&args, "--use-samples");

    // Default iters: higher for sample mode (files are tiny, each call is ~µs),
    // lower for real-data mode (each call is ~ms, 20 is enough).
    let default_iters = if use_samples { 200 } else { 20 };
    let iters = flag_val(&args, "--iters")
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_iters);

    Config { benchmark_root, benchmarks, use_samples, iters, csv_out: flag_val(&args, "--csv").map(PathBuf::from) }
}

// ── Schema parsing ────────────────────────────────────────────────────────────

/// Map column_name → (0-based index, is_string).
fn parse_table_schema(sql: &str) -> HashMap<String, (usize, bool)> {
    let string_types = ["varchar", "char", "text", "character varying", "bpchar"];
    let mut cols: HashMap<String, (usize, bool)> = HashMap::new();
    let mut idx = 0usize;
    for line in sql.lines() {
        let t = line.trim();
        if !t.starts_with('"') { continue; }
        if let Some(end) = t[1..].find('"') {
            let col_name = t[1..1 + end].to_string();
            let rest = t[2 + end..].trim().to_lowercase();
            let is_string = string_types.iter().any(|&st| rest.starts_with(st));
            cols.insert(col_name, (idx, is_string));
            idx += 1;
        }
    }
    cols
}

// ── Query filter extraction ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum FilterType {
    Eq,
    In,
    Ne,
    /// values = [low, high]; empty string = unbounded side.
    Range { low_inclusive: bool, high_inclusive: bool },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Filter {
    table:       String,
    col_name:    String,
    col_idx:     usize,
    filter_type: FilterType,
    /// EQ/Ne: [val]   In: [v1,v2,...]   Range: [low, high]
    values:      Vec<String>,
}

#[derive(Debug, Clone)]
struct HalfBound {
    col_name:  String,
    col_idx:   usize,
    value:     String,
    is_lower:  bool,
    inclusive: bool,
}

fn extract_filters(
    sql:         &str,
    table_name:  &str,
    string_cols: &HashMap<String, usize>,
) -> Vec<Filter> {
    let mut filters = Vec::new();
    let mut seen:   HashSet<Filter> = HashSet::new();
    let mut bounds: Vec<HalfBound>  = Vec::new();

    let chars: Vec<char> = sql.chars().collect();
    let n = chars.len();
    let mut i = 0;

    while i < n {
        if chars[i] == '.' && i + 1 < n && chars[i + 1] == '"' {
            let start = i + 2;
            if let Some(rel_end) = sql[start..].find('"') {
                let col_name    = sql[start..start + rel_end].to_string();
                let after_quote = start + rel_end + 1;

                if let Some(&col_idx) = string_cols.get(&col_name) {
                    let rest = sql[after_quote..].trim_start();

                    if rest.starts_with("= '") {
                        if let Some(val) = parse_quoted_string(&rest[3..]) {
                            push_filter(&mut filters, &mut seen, Filter {
                                table: table_name.into(), col_name, col_idx,
                                filter_type: FilterType::Eq, values: vec![val],
                            });
                        }
                    } else if rest.starts_with("!= '") || rest.starts_with("<> '") {
                        if let Some(val) = parse_quoted_string(&rest[4..]) {
                            push_filter(&mut filters, &mut seen, Filter {
                                table: table_name.into(), col_name, col_idx,
                                filter_type: FilterType::Ne, values: vec![val],
                            });
                        }
                    } else if rest.starts_with(">= '") {
                        if let Some(val) = parse_quoted_string(&rest[4..]) {
                            bounds.push(HalfBound { col_name, col_idx, value: val, is_lower: true,  inclusive: true  });
                        }
                    } else if rest.starts_with("> '") {
                        if let Some(val) = parse_quoted_string(&rest[3..]) {
                            bounds.push(HalfBound { col_name, col_idx, value: val, is_lower: true,  inclusive: false });
                        }
                    } else if rest.starts_with("<= '") {
                        if let Some(val) = parse_quoted_string(&rest[4..]) {
                            bounds.push(HalfBound { col_name, col_idx, value: val, is_lower: false, inclusive: true  });
                        }
                    } else if rest.starts_with("< '") {
                        if let Some(val) = parse_quoted_string(&rest[3..]) {
                            bounds.push(HalfBound { col_name, col_idx, value: val, is_lower: false, inclusive: false });
                        }
                    } else {
                        let rest_upper = rest.to_uppercase();
                        if rest_upper.starts_with("IN (") || rest_upper.starts_with("IN('") {
                            if let Some(paren) = rest.find('(') {
                                if let Some(close) = find_closing_paren(&rest[paren..]) {
                                    let inner = &rest[paren + 1..paren + close];
                                    let mut vals = parse_in_values(inner);
                                    vals.sort();
                                    if !vals.is_empty() {
                                        push_filter(&mut filters, &mut seen, Filter {
                                            table: table_name.into(), col_name, col_idx,
                                            filter_type: FilterType::In, values: vals,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                i = after_quote;
                continue;
            }
        }
        i += 1;
    }

    // Pair lower + upper bounds on the same column into Range filters.
    let mut used = vec![false; bounds.len()];
    for (li, lb) in bounds.iter().enumerate() {
        if used[li] || !lb.is_lower { continue; }
        let paired = bounds.iter().enumerate().find(|(ui, ub)| {
            !used[*ui] && !ub.is_lower && ub.col_name == lb.col_name
        });
        if let Some((ui, ub)) = paired {
            used[li] = true; used[ui] = true;
            push_filter(&mut filters, &mut seen, Filter {
                table: table_name.into(), col_name: lb.col_name.clone(), col_idx: lb.col_idx,
                filter_type: FilterType::Range { low_inclusive: lb.inclusive, high_inclusive: ub.inclusive },
                values: vec![lb.value.clone(), ub.value.clone()],
            });
        }
    }
    // Unpaired bounds become one-sided ranges.
    for (bi, b) in bounds.iter().enumerate() {
        if used[bi] { continue; }
        let (values, low_inc, high_inc) = if b.is_lower {
            (vec![b.value.clone(), String::new()], b.inclusive, false)
        } else {
            (vec![String::new(), b.value.clone()], false, b.inclusive)
        };
        push_filter(&mut filters, &mut seen, Filter {
            table: table_name.into(), col_name: b.col_name.clone(), col_idx: b.col_idx,
            filter_type: FilterType::Range { low_inclusive: low_inc, high_inclusive: high_inc },
            values,
        });
    }

    filters
}

fn push_filter(out: &mut Vec<Filter>, seen: &mut HashSet<Filter>, f: Filter) {
    if seen.insert(f.clone()) { out.push(f); }
}

fn parse_quoted_string(s: &str) -> Option<String> {
    s.find('\'').map(|end| s[..end].to_string())
}

fn find_closing_paren(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => { depth -= 1; if depth == 0 { return Some(i); } }
            _   => {}
        }
    }
    None
}

fn parse_in_values(inner: &str) -> Vec<String> {
    let mut vals = Vec::new();
    let mut rest = inner;
    loop {
        rest = rest.trim();
        if rest.is_empty() { break; }
        if rest.starts_with('\'') {
            rest = &rest[1..];
            if let Some(end) = rest.find('\'') {
                vals.push(rest[..end].to_string());
                rest = rest[end + 1..].trim_start_matches(',');
            } else { break; }
        } else if let Some(comma) = rest.find(',') {
            rest = &rest[comma + 1..];
        } else { break; }
    }
    vals
}

// ── CSV loading ───────────────────────────────────────────────────────────────

/// Load a single column from a pipe-delimited CSV file, as-is (no replication).
/// "null" and empty fields are returned as None.
fn load_column(path: &Path, col_idx: usize) -> Vec<Option<String>> {
    let content = match std::fs::read_to_string(path) {
        Ok(c)  => c,
        Err(e) => { eprintln!("  [warn] Cannot read {}: {}", path.display(), e); return Vec::new(); }
    };
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let field = line.split('|').nth(col_idx).unwrap_or("").trim().to_string();
            if field.eq_ignore_ascii_case("null") || field.is_empty() { None } else { Some(field) }
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

// ── Prefix-length statistics across all segments ──────────────────────────────

struct PrefixStats {
    segments: usize,
    min:      usize,
    median:   usize,
    max:      usize,
}

fn prefix_stats(psa: &arrow_prefix_split::PrefixSplitArray) -> PrefixStats {
    let mut lens: Vec<usize> = psa.segments()
        .iter()
        .filter(|s| s.sealed)
        .map(|s| s.prefix_len)
        .collect();
    if lens.is_empty() {
        return PrefixStats { segments: 0, min: 0, median: 0, max: 0 };
    }
    lens.sort_unstable();
    PrefixStats {
        segments: lens.len(),
        min:      *lens.first().unwrap(),
        median:   lens[lens.len() / 2],
        max:      *lens.last().unwrap(),
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
fn sa_search_in(sa: &StringArray, needles: &[String]) -> usize {
    (0..sa.len()).filter(|&i| !sa.is_null(i) && needles.iter().any(|n| sa.value(i) == n.as_str())).count()
}
fn sa_search_ne(sa: &StringArray, needle: &str) -> usize {
    (0..sa.len()).filter(|&i| !sa.is_null(i) && sa.value(i) != needle).count()
}
fn sa_search_range(sa: &StringArray, low: &str, low_inc: bool, high: &str, high_inc: bool) -> usize {
    (0..sa.len()).filter(|&i| {
        if sa.is_null(i) { return false; }
        let v = sa.value(i);
        let ok_low  = low.is_empty()  || if low_inc  { v >= low  } else { v > low  };
        let ok_high = high.is_empty() || if high_inc { v <= high } else { v < high };
        ok_low && ok_high
    }).count()
}

// ── PSA search helpers ────────────────────────────────────────────────────────

fn psa_search_in_helper(psa: &arrow_prefix_split::PrefixSplitArray, needles: &[String]) -> usize {
    let refs: Vec<&str> = needles.iter().map(String::as_str).collect();
    search_in(psa, &refs).len()
}
fn psa_search_range_helper(
    psa: &arrow_prefix_split::PrefixSplitArray,
    low: &str, low_inc: bool, high: &str, high_inc: bool,
) -> usize {
    search_range(psa, low, low_inc, high, high_inc).len()
}

// ── Result record ─────────────────────────────────────────────────────────────

struct Row {
    benchmark:  String,
    table:      String,
    column:     String,
    filter:     String,
    total_rows: usize,
    sa_us:      u64,
    psa90_us:   u64,
    psapl_us:   u64,
    /// prefix_len min/median/max across all sealed segments, for each design
    p90:        PrefixStats,
    ppl:        PrefixStats,
    sa_count:   usize,
}

// ── Core loop ─────────────────────────────────────────────────────────────────

fn run_benchmark(benchmark_dir: &Path, cfg: &Config) -> Vec<Row> {
    let name = benchmark_dir.file_name().unwrap().to_string_lossy().to_string();
    let mut rows = Vec::new();

    let tables_dir  = benchmark_dir.join("tables");
    let queries_dir = benchmark_dir.join("queries");
    // Real data lives in data/, samples live in samples/.
    let data_dir = if cfg.use_samples {
        benchmark_dir.join("samples")
    } else {
        benchmark_dir.join("data")
    };

    if !tables_dir.exists() || !queries_dir.exists() || !data_dir.exists() {
        if !cfg.use_samples && !benchmark_dir.join("data").exists() {
            eprintln!("  [{}] no data/ directory — run with --use-samples for local testing", name);
        }
        return rows;
    }

    // ── Load table schemas ────────────────────────────────────────────────
    let mut table_schemas: HashMap<String, HashMap<String, (usize, bool)>> = HashMap::new();
    if let Ok(entries) = std::fs::read_dir(&tables_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "sql") {
                let sql   = std::fs::read_to_string(&path).unwrap_or_default();
                let stem  = path.file_stem().unwrap().to_string_lossy();
                let tname = stem.trim_end_matches(".table").to_string();
                table_schemas.insert(tname, parse_table_schema(&sql));
            }
        }
    }

    // ── Locate data files ─────────────────────────────────────────────────
    // Maps table_name → path to its CSV file.
    let mut table_data: HashMap<String, PathBuf> = HashMap::new();
    if let Ok(entries) = std::fs::read_dir(&data_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "csv") {
                let stem  = path.file_stem().unwrap().to_string_lossy();
                // samples: "Hatred_1.sample" → strip ".sample"
                // real:    "Hatred_1"        → no stripping needed
                let tname = stem.trim_end_matches(".sample").to_string();
                table_data.insert(tname, path);
            }
        }
    }

    // ── Collect unique filters from all query files ───────────────────────
    let mut all_filters: Vec<Filter> = Vec::new();
    let mut filter_set:  HashSet<Filter> = HashSet::new();

    if let Ok(entries) = std::fs::read_dir(&queries_dir) {
        let mut qpaths: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().map_or(false, |e| e == "sql"))
            .collect();
        qpaths.sort();

        for qpath in &qpaths {
            let sql = std::fs::read_to_string(qpath).unwrap_or_default();
            for (tname, schema) in &table_schemas {
                let string_cols: HashMap<String, usize> = schema
                    .iter()
                    .filter_map(|(col, &(idx, is_str))| {
                        if is_str { Some((col.clone(), idx)) } else { None }
                    })
                    .collect();
                if !sql.contains(&format!("\"{}\"", tname)) { continue; }
                for f in extract_filters(&sql, tname, &string_cols) {
                    if filter_set.insert(f.clone()) { all_filters.push(f); }
                }
            }
        }
    }

    if all_filters.is_empty() {
        eprintln!("  [{}] no string filters found in queries", name);
        return rows;
    }

    // ── Sort by (table, col_idx) so we build each column array once ───────
    all_filters.sort_by(|a, b| a.table.cmp(&b.table).then(a.col_idx.cmp(&b.col_idx)));

    let mut current_key: Option<(String, usize)> = None;
    let mut sa:    Option<StringArray> = None;
    let mut psa90: Option<arrow_prefix_split::PrefixSplitArray> = None;
    let mut psapl: Option<arrow_prefix_split::PrefixSplitArray> = None;
    let mut col_data: Vec<Option<String>> = Vec::new();

    for filter in &all_filters {
        let key = (filter.table.clone(), filter.col_idx);

        if current_key.as_ref() != Some(&key) {
            current_key = Some(key.clone());

            let data_path = match table_data.get(&filter.table) {
                Some(p) => p.clone(),
                None    => {
                    eprintln!("  [{}] no data file for table {}", name, filter.table);
                    sa = None; psa90 = None; psapl = None;
                    col_data = Vec::new();
                    continue;
                }
            };

            eprint!("    loading {}/{} col {} ({} rows) … ",
                filter.table, data_path.file_name().unwrap().to_string_lossy(),
                filter.col_idx,
                // count lines without loading all text twice — just show "?" here
                "?"
            );
            col_data = load_column(&data_path, filter.col_idx);
            eprintln!("{} rows loaded", col_data.len());

            if col_data.is_empty() {
                sa = None; psa90 = None; psapl = None;
                continue;
            }

            sa    = Some(build_sa(&col_data));
            psa90 = Some(build_psa(&col_data, 0.9, 0.0));
            psapl = Some(build_psa(&col_data, 0.9, 0.01));
        }

        let (sa_arr, psa90_arr, psapl_arr) = match (&sa, &psa90, &psapl) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => continue,
        };

        let p90 = prefix_stats(psa90_arr);
        let ppl = prefix_stats(psapl_arr);
        let iters = cfg.iters;

        let (filter_label, t_sa, count_sa, t_psa90, t_psapl) = match &filter.filter_type {

            FilterType::Eq => {
                let n = filter.values[0].clone();
                let label = format!("EQ '{}'", n);
                let count = sa_search_eq(sa_arr, &n);
                let t_sa  = time_fn(|| sa_search_eq(sa_arr, &n), iters);
                let t_90  = time_fn(|| search_eq(psa90_arr, &n).len(), iters);
                let t_pl  = time_fn(|| search_eq(psapl_arr, &n).len(), iters);
                (label, t_sa, count, t_90, t_pl)
            }

            FilterType::In => {
                let vals  = filter.values.clone();
                let label = format!("IN({}) '{}'{}",
                    vals.len(), vals[0],
                    if vals.len() > 1 { "',..." } else { "'" }
                );
                let count = sa_search_in(sa_arr, &vals);
                let t_sa  = time_fn(|| sa_search_in(sa_arr, &vals), iters);
                let t_90  = time_fn(|| psa_search_in_helper(psa90_arr, &vals), iters);
                let t_pl  = time_fn(|| psa_search_in_helper(psapl_arr, &vals), iters);
                (label, t_sa, count, t_90, t_pl)
            }

            FilterType::Ne => {
                let n     = filter.values[0].clone();
                let label = format!("NE '{}'", n);
                let count = sa_search_ne(sa_arr, &n);
                let t_sa  = time_fn(|| sa_search_ne(sa_arr, &n), iters);
                let t_90  = time_fn(|| search_ne(psa90_arr, &n).len(), iters);
                let t_pl  = time_fn(|| search_ne(psapl_arr, &n).len(), iters);
                (label, t_sa, count, t_90, t_pl)
            }

            FilterType::Range { low_inclusive, high_inclusive } => {
                let low      = filter.values[0].clone();
                let high     = filter.values[1].clone();
                let low_inc  = *low_inclusive;
                let high_inc = *high_inclusive;
                let lo_sym   = if low_inc  { ">=" } else { ">" };
                let hi_sym   = if high_inc { "<=" } else { "<" };
                let label = match (low.is_empty(), high.is_empty()) {
                    (false, false) => format!("{lo_sym}'{low}' {hi_sym}'{high}'"),
                    (false, true)  => format!("{lo_sym}'{low}'"),
                    (true,  false) => format!("{hi_sym}'{high}'"),
                    (true,  true)  => "RANGE(unbounded)".into(),
                };
                let count = sa_search_range(sa_arr, &low, low_inc, &high, high_inc);
                let t_sa  = time_fn(|| sa_search_range(sa_arr, &low, low_inc, &high, high_inc), iters);
                let t_90  = time_fn(|| psa_search_range_helper(psa90_arr, &low, low_inc, &high, high_inc), iters);
                let t_pl  = time_fn(|| psa_search_range_helper(psapl_arr, &low, low_inc, &high, high_inc), iters);
                (label, t_sa, count, t_90, t_pl)
            }
        };

        rows.push(Row {
            benchmark:  name.clone(),
            table:      filter.table.clone(),
            column:     filter.col_name.clone(),
            filter:     filter_label,
            total_rows: col_data.len(),
            sa_us:      t_sa.as_micros()    as u64,
            psa90_us:   t_psa90.as_micros() as u64,
            psapl_us:   t_psapl.as_micros() as u64,
            p90,
            ppl,
            sa_count:   count_sa,
        });
    }

    rows
}

// ── Output ────────────────────────────────────────────────────────────────────

fn print_table(rows: &[Row]) {
    // Header — two lines to fit in 132 columns
    println!(
        "\n{:<15} {:<13} {:<18} {:<26} {:>8} {:>7} {:>7} {:>7} {:>9} {:>9} {:>7} {:>7}",
        "Benchmark", "Table", "Column", "Filter",
        "Rows", "SA µs", "90% µs", "PL µs",
        "P90(med)", "PPl(med)",
        "90%/SA", "PL/SA"
    );
    println!("{}", "-".repeat(132));

    let mut prev = String::new();
    for r in rows {
        if r.benchmark != prev && !prev.is_empty() { println!(); }
        prev = r.benchmark.clone();

        let fmt_sp = |psa_us: u64| -> String {
            if psa_us > 0 { format!("{:.2}x", r.sa_us as f64 / psa_us as f64) } else { "—".into() }
        };
        let p90_str = format!("{}/{}/{}", r.p90.min, r.p90.median, r.p90.max);
        let ppl_str = format!("{}/{}/{}", r.ppl.min, r.ppl.median, r.ppl.max);

        println!(
            "{:<15} {:<13} {:<18} {:<26} {:>8} {:>7} {:>7} {:>7} {:>9} {:>9} {:>7} {:>7}",
            truncate(&r.benchmark, 15),
            truncate(&r.table, 13),
            truncate(&r.column, 18),
            truncate(&r.filter, 26),
            r.total_rows,
            r.sa_us, r.psa90_us, r.psapl_us,
            p90_str, ppl_str,
            fmt_sp(r.psa90_us),
            fmt_sp(r.psapl_us),
        );
    }
    println!();

    if rows.is_empty() { return; }

    let total       = rows.len();
    let psa90_wins  = rows.iter().filter(|r| r.psa90_us < r.sa_us).count();
    let psa90_loses = rows.iter().filter(|r| r.psa90_us > r.sa_us).count();
    let pl_wins     = rows.iter().filter(|r| r.psapl_us < r.sa_us).count();

    let avg_90: f64 = rows.iter().filter(|r| r.psa90_us > 0)
        .map(|r| r.sa_us as f64 / r.psa90_us as f64).sum::<f64>() / total as f64;
    let avg_pl: f64 = rows.iter().filter(|r| r.psapl_us > 0)
        .map(|r| r.sa_us as f64 / r.psapl_us as f64).sum::<f64>() / total as f64;

    let by_type = |prefix: &str| rows.iter().filter(|r| r.filter.starts_with(prefix)).count();
    println!("Summary: {total} filters across {} benchmarks",
        rows.iter().map(|r| &r.benchmark).collect::<HashSet<_>>().len());
    println!("  by type: EQ={}, IN={}, NE={}, RANGE={}",
        by_type("EQ"), by_type("IN("), by_type("NE"),
        by_type(">=") + by_type("<=") + by_type(">='") + by_type("<='"));
    println!();
    println!("  PSA-90%  faster than StringArray: {psa90_wins}/{total}");
    println!("  PSA-90%  slower than StringArray: {psa90_loses}/{total}");
    println!("  PSA-PL   faster than StringArray: {pl_wins}/{total}");
    println!("  Avg PSA-90%  speedup: {avg_90:.2}x");
    println!("  Avg PSA-PL   speedup: {avg_pl:.2}x");
    println!();
    println!("  P90(med) column = min/median/max prefix_len across sealed segments (PSA-90%)");
    println!("  PPl(med) column = min/median/max prefix_len across sealed segments (PSA-plateau)");
}

fn write_csv(rows: &[Row], path: &Path) -> std::io::Result<()> {
    let mut out = String::new();
    writeln!(out,
        "benchmark,table,column,filter,total_rows,sa_us,psa90_us,psapl_us,matches,\
         p90_min,p90_med,p90_max,ppl_min,ppl_med,ppl_max,speedup_90,speedup_pl"
    ).unwrap();
    for r in rows {
        let sp90 = if r.psa90_us > 0 { format!("{:.4}", r.sa_us as f64 / r.psa90_us as f64) } else { "0".into() };
        let sppl = if r.psapl_us > 0 { format!("{:.4}", r.sa_us as f64 / r.psapl_us as f64) } else { "0".into() };
        writeln!(out,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            r.benchmark, r.table,
            csv_escape(&r.column), csv_escape(&r.filter),
            r.total_rows, r.sa_us, r.psa90_us, r.psapl_us, r.sa_count,
            r.p90.min, r.p90.median, r.p90.max,
            r.ppl.min, r.ppl.median, r.ppl.max,
            sp90, sppl,
        ).unwrap();
    }
    std::fs::write(path, out)
}

fn truncate(s: &str, n: usize) -> String {
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

// ── Feature coverage report ───────────────────────────────────────────────────

fn report_feature_coverage(benchmark_dirs: &[PathBuf]) {
    println!("=== Query feature coverage ===\n");
    let (mut eq, mut in_, mut ne, mut range, mut like, mut total) = (0,0,0,0,0,0);
    for dir in benchmark_dirs {
        let qdir = dir.join("queries");
        if let Ok(entries) = std::fs::read_dir(qdir) {
            for entry in entries.flatten() {
                if entry.path().extension().map_or(false, |e| e == "sql") {
                    let sql = std::fs::read_to_string(entry.path()).unwrap_or_default();
                    let up  = sql.to_uppercase();
                    total += 1;
                    if up.contains("= '")   { eq    += 1; }
                    if up.contains(" IN (") { in_   += 1; }
                    if up.contains("!= '") || up.contains("<> '") { ne += 1; }
                    if up.contains(">= '") || up.contains("<= '") { range += 1; }
                    if up.contains("LIKE '") { like += 1; }
                }
            }
        }
    }
    println!("  Across {total} query files:");
    println!("  = 'val'      (EQ):    {eq}   ✓ search_eq");
    println!("  IN (…)       (IN):    {in_}   ✓ search_in  (single pass)");
    println!("  != / <>      (NE):    {ne}    ✓ search_ne");
    println!("  >= / <=      (RANGE): {range}   ✓ search_range (prefix-accelerated)");
    println!("  LIKE         (LIKE):  {like}   ✗ not implemented");
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let cfg = parse_args();

    if !cfg.benchmark_root.exists() {
        eprintln!("Error: benchmark root not found: {}", cfg.benchmark_root.display());
        eprintln!("Pass --data PATH to specify the public_bi_benchmark directory.");
        std::process::exit(1);
    }

    let mode = if cfg.use_samples { "sample files (--use-samples)" } else { "real data files" };
    println!("Mode:    {mode}");
    println!("Iters:   {} timing iterations per filter", cfg.iters);
    println!("Designs: StringArray | PSA-90% | PSA-plateau(1%)");

    let mut bench_dirs: Vec<PathBuf> = std::fs::read_dir(&cfg.benchmark_root)
        .expect("Cannot read benchmark root")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    bench_dirs.sort();

    let bench_dirs: Vec<PathBuf> = match &cfg.benchmarks {
        None        => bench_dirs,
        Some(names) => bench_dirs.into_iter()
            .filter(|p| {
                let n = p.file_name().unwrap().to_string_lossy().to_string();
                names.iter().any(|name| name.eq_ignore_ascii_case(&n))
            })
            .collect(),
    };

    println!("Running {} benchmark(s)\n", bench_dirs.len());

    let mut all_rows: Vec<Row> = Vec::new();
    for dir in &bench_dirs {
        let bname = dir.file_name().unwrap().to_string_lossy();
        eprintln!("Processing {} …", bname);
        let rr = run_benchmark(dir, &cfg);
        eprintln!("  → {} filters benchmarked", rr.len());
        all_rows.extend(rr);
    }

    print_table(&all_rows);
    report_feature_coverage(&bench_dirs);

    if let Some(csv_path) = &cfg.csv_out {
        write_csv(&all_rows, csv_path).expect("Failed to write CSV");
        println!("\nResults written to {}", csv_path.display());
    }
}
