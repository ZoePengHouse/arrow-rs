// Licensed under the Apache License, Version 2.0.
//
//! Scan every string column in every table of the Public BI Benchmark.
//! For each column, compute:
//!   - average value length
//!   - estimated prefix_len (same algorithm PSA uses at 90% threshold)
//!   - PSA score = (avg_len - prefix_len) × skip_fraction
//!     where skip_fraction = 1 - 1/distinct
//!
//! After ranking, writes one JSON benchmark-definition file per top-N column
//! into --benchmarks-dir (default: ./benchmarks/).  Each file contains
//! 10 auto-generated filter queries (EQ × 3, NE × 1, IN × 3, RANGE × 3)
//! whose values are drawn from the actual column data.
//!
//! # Usage (sample data — works locally)
//!
//!   cargo run -p arrow-prefix-split --bin column_stats --release -- \
//!       --use-samples
//!
//! # Usage (real data — run on the server)
//!
//!   cargo run -p arrow-prefix-split --bin column_stats --release -- \
//!       --data /path/to/public_bi_benchmark
//!
//! # Options
//!
//!   --data           PATH  path to the public_bi_benchmark directory
//!   --use-samples          read from samples/*.sample.csv instead of data/*.csv
//!   --top            N     show the top N columns and generate N benchmark files (default: 10)
//!   --sort           FIELD sort by: score (default), avg_len, suffix_bytes, skip
//!   --min-len        N     only include columns with avg_len >= N (default: 20)
//!   --min-card       F     only include columns with skip_fraction >= F (default: 0.0)
//!   --benchmarks-dir DIR   directory to write benchmark JSON files (default: ./benchmarks)
//!   --csv            FILE  write full results (all columns) to a CSV file

use std::collections::{HashMap, HashSet};
use std::fmt::Write as FmtWrite;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum SortBy { Score, AvgLen, SuffixBytes, SkipFraction }

struct Config {
    benchmark_root: PathBuf,
    use_samples:    bool,
    top_n:          usize,
    sort_by:        SortBy,
    min_avg_len:    f64,
    min_card:       f64,
    csv_out:        Option<PathBuf>,
    benchmarks_dir: PathBuf,
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

    let sort_by = match flag_val(&args, "--sort").unwrap_or("score") {
        "avg_len"      => SortBy::AvgLen,
        "suffix_bytes" => SortBy::SuffixBytes,
        "skip"         => SortBy::SkipFraction,
        _              => SortBy::Score,
    };

    Config {
        benchmark_root: PathBuf::from(data).join("benchmark"),
        use_samples:    has_flag(&args, "--use-samples"),
        top_n:          flag_val(&args, "--top").and_then(|s| s.parse().ok()).unwrap_or(10),
        sort_by,
        min_avg_len:    flag_val(&args, "--min-len").and_then(|s| s.parse().ok()).unwrap_or(20.0),
        min_card:       flag_val(&args, "--min-card").and_then(|s| s.parse().ok()).unwrap_or(0.0),
        csv_out:        flag_val(&args, "--csv").map(PathBuf::from),
        benchmarks_dir: PathBuf::from(
            flag_val(&args, "--benchmarks-dir").unwrap_or("benchmarks")
        ),
    }
}

// ── Schema parsing ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct ColDef {
    name:    String,
    col_idx: usize,
}

fn parse_schema(sql: &str) -> Vec<ColDef> {
    let string_types = ["varchar", "char", "text", "character varying", "bpchar"];
    let mut cols = Vec::new();
    let mut idx  = 0usize;
    for line in sql.lines() {
        let t = line.trim();
        if !t.starts_with('"') { continue; }
        if let Some(end) = t[1..].find('"') {
            let name  = t[1..1 + end].to_string();
            let rest  = t[2 + end..].trim().to_lowercase();
            let is_str = string_types.iter().any(|&st| rest.starts_with(st));
            if is_str {
                cols.push(ColDef { name, col_idx: idx });
            }
            idx += 1;
        }
    }
    cols
}

// ── Prefix-length estimator (same logic as PrefixSplitBuilder) ───────────────

fn estimate_prefix_len(values: &[&str]) -> usize {
    let non_null: Vec<&[u8]> = values.iter()
        .filter(|v| !v.is_empty() && !v.eq_ignore_ascii_case("null"))
        .map(|v| v.as_bytes())
        .collect();

    if non_null.is_empty() { return 0; }

    let total_distinct = non_null.iter().collect::<HashSet<_>>().len();
    if total_distinct <= 1 { return 1; }

    let target = (total_distinct as f64 * 0.9).ceil() as usize;
    let max_p  = 64usize;

    for p in 1..=max_p {
        let distinct_prefixes: HashSet<&[u8]> = non_null.iter()
            .map(|v| &v[..v.len().min(p)])
            .collect();
        if distinct_prefixes.len() >= target {
            return p;
        }
    }
    max_p
}

// ── CSV scanning ──────────────────────────────────────────────────────────────

struct ColData {
    values:     Vec<String>,  // all non-null values (trimmed)
    null_count: u64,
}

fn scan_file(path: &Path, cols: &[ColDef]) -> HashMap<usize, ColData> {
    let file = match std::fs::File::open(path) {
        Ok(f)  => f,
        Err(e) => { eprintln!("  [warn] Cannot open {}: {}", path.display(), e); return HashMap::new(); }
    };

    if cols.is_empty() { return HashMap::new(); }

    let max_idx = cols.iter().map(|c| c.col_idx).max().unwrap_or(0);
    let mut data: HashMap<usize, ColData> = cols.iter()
        .map(|c| (c.col_idx, ColData { values: Vec::new(), null_count: 0 }))
        .collect();

    let mut sorted_cols: Vec<&ColDef> = cols.iter().collect();
    sorted_cols.sort_by_key(|c| c.col_idx);

    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = match line { Ok(l) => l, Err(_) => continue };
        if line.trim().is_empty() { continue; }

        let mut field_iter = line.splitn(max_idx + 2, '|');
        let mut current = 0usize;

        for col in &sorted_cols {
            while current < col.col_idx {
                field_iter.next();
                current += 1;
            }
            if let Some(field) = field_iter.next() {
                let f = field.trim();
                let entry = data.get_mut(&col.col_idx).unwrap();
                if f.is_empty() || f.eq_ignore_ascii_case("null") {
                    entry.null_count += 1;
                } else {
                    entry.values.push(f.to_string());
                }
                current += 1;
            }
        }
    }

    data
}

// ── Result record ─────────────────────────────────────────────────────────────

struct ColumnStat {
    benchmark:    String,
    table:        String,
    column:       String,
    col_idx:      usize,
    total_rows:   u64,
    null_count:   u64,
    avg_len:      f64,
    distinct:     usize,
    prefix_len:   usize,
    suffix_bytes: f64,
    skip_fraction: f64,
    psa_score:    f64,
    /// Up to 500 distinct values (sorted), used to generate benchmark queries.
    sample_distinct: Vec<String>,
}

impl ColumnStat {
    fn from_data(
        benchmark: String, table: String, column: String, col_idx: usize,
        data: &ColData,
    ) -> Self {
        let non_null   = data.values.len() as u64;
        let total_rows = non_null + data.null_count;

        let avg_len = if non_null == 0 { 0.0 } else {
            data.values.iter().map(|v| v.len()).sum::<usize>() as f64 / non_null as f64
        };

        let refs: Vec<&str> = data.values.iter().map(String::as_str).collect();
        let distinct   = refs.iter().collect::<HashSet<_>>().len();
        let prefix_len = estimate_prefix_len(&refs);

        let suffix_bytes  = (avg_len - prefix_len as f64).max(0.0);
        let skip_fraction = if distinct <= 1 { 0.0 } else { 1.0 - 1.0 / distinct as f64 };
        let psa_score     = suffix_bytes * skip_fraction;

        // Collect ALL distinct values, sort them, then sub-sample to at most
        // 1000 evenly-spaced entries for query generation.
        //
        // Why not just take the first 500?
        //   On real data a column can have millions of distinct values.
        //   "First 500" would all cluster near the start of the alphabet —
        //   pick(0.5) would give the median of those 500, not the true median
        //   of the full column.  Instead we keep 1000 entries uniformly spread
        //   across the sorted distinct list so every pick(frac) call maps to
        //   the correct percentile of the *real* distribution regardless of
        //   how many distinct values the column has.
        let mut all_distinct: Vec<String> = {
            let mut seen = HashSet::new();
            refs.iter()
                .filter(|&&v| seen.insert(v))
                .map(|&v| v.to_string())
                .collect()
        };
        all_distinct.sort();

        let sample_distinct: Vec<String> = {
            let n = all_distinct.len();
            if n <= 1000 {
                all_distinct
            } else {
                // Evenly subsample: entry i maps to position
                //   round(i * (n-1) / 999) in the sorted list.
                (0..1000usize)
                    .map(|i| all_distinct[i * (n - 1) / 999].clone())
                    .collect()
            }
        };

        ColumnStat {
            benchmark, table, column, col_idx,
            total_rows, null_count: data.null_count,
            avg_len, distinct, prefix_len,
            suffix_bytes, skip_fraction, psa_score,
            sample_distinct,
        }
    }
}

// ── Benchmark definition structs (JSON-serialisable) ─────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct QueryDef {
    pub id:         usize,
    /// "Eq" | "Ne" | "In" | "Range"
    #[serde(rename = "type")]
    pub query_type: String,
    /// Human-readable label for output tables.
    pub label:      String,
    /// EQ/NE: [value]   IN: [v1,v2,...]   RANGE: [low, high]
    /// For one-sided range, the missing bound is "".
    pub values:     Vec<String>,
    /// Only meaningful for Range queries.
    #[serde(default)]
    pub low_inc:    bool,
    #[serde(default)]
    pub high_inc:   bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BenchmarkDef {
    pub rank:       usize,
    pub benchmark:  String,
    pub table:      String,
    pub column:     String,
    pub col_idx:    usize,
    pub avg_len:    f64,
    pub prefix_len: usize,
    pub psa_score:  f64,
    pub queries:    Vec<QueryDef>,
}

// ── Query generation ──────────────────────────────────────────────────────────

/// Generate exactly 10 filter queries from a sorted list of distinct values.
///
/// Mix:
///   3 × EQ   — at 10 %, 50 %, 90 % of the distinct-value distribution
///   1 × NE   — at 50 %
///   3 × IN   — IN(2) at [20%,80%]; IN(3) at [10%,50%,90%]; IN(5) at spread
///   3 × RANGE — narrow [25%,75%]; wide [10%,90%]; one-sided >= 50%
fn generate_queries(distinct_sorted: &[String]) -> Vec<QueryDef> {
    let n = distinct_sorted.len();
    if n == 0 { return vec![]; }

    // Pick a value at a fractional position in the sorted distinct list.
    let pick = |frac: f64| -> String {
        let idx = ((frac * (n - 1) as f64).round() as usize).min(n - 1);
        distinct_sorted[idx].clone()
    };

    // Shorten a value for display in a label.
    let label_val = |v: &str| -> String {
        if v.len() > 20 {
            let end = v.char_indices().nth(19).map_or(v.len(), |(i, _)| i);
            format!("{}…", &v[..end])
        } else {
            v.to_string()
        }
    };

    let mut qs: Vec<QueryDef> = Vec::with_capacity(10);

    // — EQ queries (3) —
    for (i, frac) in [(1, 0.1f64), (2, 0.5), (3, 0.9)] {
        let v = pick(frac);
        qs.push(QueryDef {
            id: i,
            query_type: "Eq".into(),
            label: format!("EQ '{}'", label_val(&v)),
            values: vec![v],
            low_inc: false, high_inc: false,
        });
    }

    // — NE query (1) —
    {
        let v = pick(0.5);
        qs.push(QueryDef {
            id: 4,
            query_type: "Ne".into(),
            label: format!("NE '{}'", label_val(&v)),
            values: vec![v],
            low_inc: false, high_inc: false,
        });
    }

    // — IN queries (3) —
    {
        let v0 = pick(0.2); let v1 = pick(0.8);
        qs.push(QueryDef {
            id: 5,
            query_type: "In".into(),
            label: format!("IN(2) '{}','{}'", label_val(&v0), label_val(&v1)),
            values: vec![v0, v1],
            low_inc: false, high_inc: false,
        });
    }
    {
        let v0 = pick(0.1); let v1 = pick(0.5); let v2 = pick(0.9);
        qs.push(QueryDef {
            id: 6,
            query_type: "In".into(),
            label: format!("IN(3) '{}',…", label_val(&v0)),
            values: vec![v0, v1, v2],
            low_inc: false, high_inc: false,
        });
    }
    {
        let vals: Vec<String> = [0.0f64, 0.25, 0.5, 0.75, 1.0].iter().map(|&f| pick(f)).collect();
        qs.push(QueryDef {
            id: 7,
            query_type: "In".into(),
            label: format!("IN(5) '{}',…", label_val(&vals[0])),
            values: vals,
            low_inc: false, high_inc: false,
        });
    }

    // — RANGE queries (3) —
    {
        let lo = pick(0.25); let hi = pick(0.75);
        qs.push(QueryDef {
            id: 8,
            query_type: "Range".into(),
            label: format!("RANGE ['{}','{}']", label_val(&lo), label_val(&hi)),
            values: vec![lo, hi],
            low_inc: true, high_inc: true,
        });
    }
    {
        let lo = pick(0.1); let hi = pick(0.9);
        qs.push(QueryDef {
            id: 9,
            query_type: "Range".into(),
            label: format!("RANGE wide ['{}','{}']", label_val(&lo), label_val(&hi)),
            values: vec![lo, hi],
            low_inc: true, high_inc: true,
        });
    }
    {
        let lo = pick(0.5);
        qs.push(QueryDef {
            id: 10,
            query_type: "Range".into(),
            label: format!("RANGE >= '{}'", label_val(&lo)),
            values: vec![lo, String::new()],
            low_inc: true, high_inc: false,
        });
    }

    qs
}

// ── Main scan ─────────────────────────────────────────────────────────────────

fn collect_stats(cfg: &Config) -> Vec<ColumnStat> {
    let mut all: Vec<ColumnStat> = Vec::new();

    let mut bench_dirs: Vec<PathBuf> = match std::fs::read_dir(&cfg.benchmark_root) {
        Ok(e) => e.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect(),
        Err(e) => { eprintln!("Cannot read {}: {}", cfg.benchmark_root.display(), e); return all; }
    };
    bench_dirs.sort();

    for bench_dir in &bench_dirs {
        let bench_name = bench_dir.file_name().unwrap().to_string_lossy().to_string();
        let tables_dir = bench_dir.join("tables");
        let data_dir   = if cfg.use_samples { bench_dir.join("samples") } else { bench_dir.join("data") };

        if !tables_dir.exists() { continue; }
        if !data_dir.exists() {
            if !cfg.use_samples {
                eprintln!("  [{}] no data/ directory", bench_name);
            }
            continue;
        }

        let mut data_files: HashMap<String, PathBuf> = HashMap::new();
        if let Ok(entries) = std::fs::read_dir(&data_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "csv") {
                    let stem  = path.file_stem().unwrap().to_string_lossy();
                    let tname = stem.trim_end_matches(".sample").to_string();
                    data_files.insert(tname, path);
                }
            }
        }

        let mut schema_paths: Vec<PathBuf> = match std::fs::read_dir(&tables_dir) {
            Ok(e) => e.flatten().map(|e| e.path())
                       .filter(|p| p.extension().map_or(false, |e| e == "sql"))
                       .collect(),
            Err(_) => continue,
        };
        schema_paths.sort();

        for schema_path in &schema_paths {
            let sql   = std::fs::read_to_string(schema_path).unwrap_or_default();
            let stem  = schema_path.file_stem().unwrap().to_string_lossy();
            let tname = stem.trim_end_matches(".table").to_string();

            let cols = parse_schema(&sql);
            if cols.is_empty() { continue; }

            let data_path = match data_files.get(&tname) {
                Some(p) => p,
                None    => {
                    // Try with number suffix (e.g., schema "IGlocations2" → file "IGlocations2_1")
                    match data_files.iter().find(|(k, _)| k.starts_with(&tname)) {
                        Some((_, p)) => p,
                        None => {
                            eprintln!("  [{}] no data file for table {}", bench_name, tname);
                            continue;
                        }
                    }
                }
            };

            eprint!("  Scanning {}/{} … ", bench_name, tname);
            let col_data = scan_file(data_path, &cols);
            if col_data.is_empty() { eprintln!("(no data)"); continue; }

            let total_rows = col_data.values().next().map_or(0, |d| {
                d.values.len() as u64 + d.null_count
            });
            eprintln!("{} rows, {} string cols", total_rows, cols.len());

            for col in &cols {
                if let Some(data) = col_data.get(&col.col_idx) {
                    all.push(ColumnStat::from_data(
                        bench_name.clone(), tname.clone(), col.name.clone(), col.col_idx, data,
                    ));
                }
            }
        }
    }

    all
}

// ── Output ────────────────────────────────────────────────────────────────────

fn sort_key(s: &ColumnStat, by: SortBy) -> f64 {
    match by {
        SortBy::Score        => s.psa_score,
        SortBy::AvgLen       => s.avg_len,
        SortBy::SuffixBytes  => s.suffix_bytes,
        SortBy::SkipFraction => s.skip_fraction,
    }
}

fn filtered_top<'a>(stats: &'a [ColumnStat], cfg: &Config) -> Vec<&'a ColumnStat> {
    let mut filtered: Vec<&ColumnStat> = stats.iter()
        .filter(|s| s.avg_len >= cfg.min_avg_len)
        .filter(|s| s.skip_fraction >= cfg.min_card)
        .collect();
    filtered.sort_by(|a, b| {
        sort_key(b, cfg.sort_by).partial_cmp(&sort_key(a, cfg.sort_by)).unwrap()
    });
    filtered
}

fn print_top(stats: &[ColumnStat], cfg: &Config) {
    let filtered = filtered_top(stats, cfg);

    let sort_label = match cfg.sort_by {
        SortBy::Score        => "PSA score  suffix_bytes × skip_fraction  (default)",
        SortBy::AvgLen       => "avg length",
        SortBy::SuffixBytes  => "suffix bytes saved per skip",
        SortBy::SkipFraction => "skip fraction",
    };

    println!(
        "\nFilters: avg_len >= {:.0} bytes,  skip_fraction >= {:.2}",
        cfg.min_avg_len, cfg.min_card
    );
    println!("Sort:    {sort_label}");
    println!(
        "\n{:<4}  {:<14} {:<16} {:<22} {:>8} {:>6} {:>10} {:>8} {:>9} {:>8}",
        "Rank", "Benchmark", "Table", "Column",
        "AvgLen", "P_len", "SuffixSave", "Skip%", "Score", "Distinct"
    );
    println!("{}", "-".repeat(113));

    for (rank, s) in filtered.iter().take(cfg.top_n).enumerate() {
        println!(
            "{:<4}  {:<14} {:<16} {:<22} {:>8.1} {:>6} {:>10.1} {:>7.1}% {:>9.1} {:>8}",
            rank + 1,
            truncate(&s.benchmark, 14),
            truncate(&s.table, 16),
            truncate(&s.column, 22),
            s.avg_len,
            s.prefix_len,
            s.suffix_bytes,
            s.skip_fraction * 100.0,
            s.psa_score,
            s.distinct,
        );
    }

    println!();
    println!("Columns scanned: {}   After filters: {}   Showing top: {}",
        stats.len(), filtered.len(), cfg.top_n.min(filtered.len()));
    println!();
    println!("Column headers:");
    println!("  AvgLen     — average byte length of non-null values");
    println!("  P_len      — prefix_len PSA would choose (90% threshold)");
    println!("  SuffixSave — avg_len - prefix_len: bytes skipped per eliminated row");
    println!("  Skip%      — (1 - 1/distinct) × 100: % of rows eliminated by prefix for a typical needle");
    println!("  Score      — SuffixSave × Skip%/100: expected bytes saved per row scanned");
    println!("  Distinct   — number of distinct non-null values seen");
}

// ── Benchmark file generation ─────────────────────────────────────────────────

/// Write one JSON benchmark-definition file per top-N column.
/// Each file contains column metadata + 10 auto-generated filter queries.
fn generate_benchmarks(stats: &[ColumnStat], cfg: &Config) {
    let top = filtered_top(stats, cfg);
    let to_write: Vec<(usize, &&ColumnStat)> = top.iter()
        .take(cfg.top_n)
        .enumerate()
        .collect();

    if to_write.is_empty() {
        eprintln!("No columns qualify for benchmark generation.");
        return;
    }

    std::fs::create_dir_all(&cfg.benchmarks_dir).unwrap_or_else(|e| {
        eprintln!("Cannot create benchmarks dir {}: {}", cfg.benchmarks_dir.display(), e);
    });

    let mut written = 0usize;
    for (rank, s) in &to_write {
        let queries = generate_queries(&s.sample_distinct);
        if queries.is_empty() {
            eprintln!("  [rank {}] {} / {} — no distinct values, skipping", rank + 1, s.benchmark, s.column);
            continue;
        }

        let def = BenchmarkDef {
            rank:       rank + 1,
            benchmark:  s.benchmark.clone(),
            table:      s.table.clone(),
            column:     s.column.clone(),
            col_idx:    s.col_idx,
            avg_len:    (s.avg_len * 10.0).round() / 10.0,
            prefix_len: s.prefix_len,
            psa_score:  (s.psa_score * 10.0).round() / 10.0,
            queries,
        };

        // File name: bench_01_Benchmark_Table_Column.json  (sanitised)
        let safe = |s: &str| s.replace(['/', '\\', ' ', '"', '\''], "_");
        let fname = format!(
            "bench_{:02}_{}_{}_{}.json",
            rank + 1,
            safe(&def.benchmark),
            safe(&def.table),
            safe(&def.column),
        );
        let path = cfg.benchmarks_dir.join(&fname);

        match serde_json::to_string_pretty(&def) {
            Ok(json) => {
                std::fs::write(&path, json).unwrap_or_else(|e| {
                    eprintln!("  Cannot write {}: {}", path.display(), e);
                });
                println!("  wrote {}", path.display());
                written += 1;
            }
            Err(e) => eprintln!("  Serialisation error for {}: {}", s.column, e),
        }
    }

    println!();
    println!(
        "Generated {} benchmark definition file(s) in  {}",
        written,
        cfg.benchmarks_dir.display()
    );
    println!();
    println!("To run benchmarks locally (sample data):");
    println!("  cargo run -p arrow-prefix-split --bin bench_runner --release -- \\");
    println!("    --use-samples --all");
    println!();
    println!("To run benchmarks on server (real data):");
    println!("  cargo run -p arrow-prefix-split --bin bench_runner --release -- \\");
    println!("    --data /path/to/public_bi_benchmark --all");
}

fn write_csv(stats: &[ColumnStat], path: &Path) -> std::io::Result<()> {
    let mut sorted: Vec<&ColumnStat> = stats.iter().collect();
    sorted.sort_by(|a, b| b.psa_score.partial_cmp(&a.psa_score).unwrap());

    let mut out = String::new();
    writeln!(out,
        "rank,benchmark,table,column,col_idx,total_rows,null_pct,avg_len,distinct,\
         prefix_len,suffix_bytes,skip_fraction,psa_score"
    ).unwrap();

    for (rank, s) in sorted.iter().enumerate() {
        let null_pct = if s.total_rows > 0 {
            s.null_count as f64 / s.total_rows as f64 * 100.0
        } else { 0.0 };
        writeln!(out,
            "{},{},{},{},{},{},{:.1},{:.2},{},{},{:.2},{:.4},{:.4}",
            rank + 1,
            s.benchmark, s.table, csv_escape(&s.column), s.col_idx,
            s.total_rows, null_pct, s.avg_len, s.distinct,
            s.prefix_len, s.suffix_bytes, s.skip_fraction, s.psa_score,
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

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let cfg = parse_args();

    if !cfg.benchmark_root.exists() {
        eprintln!("Error: benchmark root not found: {}", cfg.benchmark_root.display());
        eprintln!("Pass --data PATH to specify the public_bi_benchmark directory.");
        std::process::exit(1);
    }

    let mode = if cfg.use_samples { "sample files (--use-samples)" } else { "real data files" };
    println!("Mode: {mode}");

    let stats = collect_stats(&cfg);

    if stats.is_empty() {
        eprintln!("No string columns found.");
        std::process::exit(1);
    }

    print_top(&stats, &cfg);

    println!("=== Generating benchmark definition files ===\n");
    generate_benchmarks(&stats, &cfg);

    if let Some(csv_path) = &cfg.csv_out {
        write_csv(&stats, csv_path).expect("Failed to write CSV");
        println!("Full results written to {}", csv_path.display());
    }
}
