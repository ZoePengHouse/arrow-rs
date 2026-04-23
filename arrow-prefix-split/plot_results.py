#!/usr/bin/env python3
"""
Plot PSA Range benchmark results from results.csv.

Usage:
    python3 plot_results.py results.csv --out plots/

Produces one PNG per benchmark column (speedup vs selectivity) plus
a combined summary chart.

Requirements:
    pip install matplotlib pandas
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # no display needed on server
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Colours ───────────────────────────────────────────────────────────────────
C90  = "#e07b39"   # PSA-90%   orange
CPL  = "#2176ae"   # PSA-PL    blue
CFX  = "#57a773"   # PSA-fixed green
CSA  = "#888888"   # baseline  grey

def short_col(row):
    return f"{row['benchmark']} / {row['table']} / {row['column']}"

def marker_for(sp):
    if sp > 1.05:  return "▲"
    if sp < 0.95:  return "▼"
    return " "

# ── Per-benchmark chart ───────────────────────────────────────────────────────
def plot_one(df_bench, out_dir):
    """One chart per benchmark column: speedup vs actual selectivity."""
    df = df_bench.sort_values("actual_sel")
    col_label = short_col(df.iloc[0])
    avg_len   = df.iloc[0]["avg_len"]
    pfx_len   = df.iloc[0]["prefix_len"]
    rank      = df.iloc[0]["rank"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Rank {rank}  ·  {col_label}\n"
        f"avg_len={avg_len:.0f} B   prefix_len={pfx_len} B   "
        f"prefix/avg={pfx_len/avg_len*100:.1f}%",
        fontsize=11, y=1.01
    )

    # ── Left: speedup vs selectivity ─────────────────────────────────────────
    ax = axes[0]
    xs = df["actual_sel"] * 100          # percent

    ax.axhline(1.0, color=CSA, lw=1.2, ls="--", label="SA baseline (1.0×)")
    ax.plot(xs, df["speedup_90"], "o-", color=C90, lw=1.8, ms=5, label="PSA-90%")
    ax.plot(xs, df["speedup_pl"], "s-", color=CPL, lw=2.2, ms=6, label="PSA-plateau")
    ax.plot(xs, df["speedup_fx"], "^-", color=CFX, lw=1.8, ms=5, label="PSA-fixed-3")

    ax.set_xlabel("Actual selectivity  (% rows matched)", fontsize=10)
    ax.set_ylabel("Speedup  (SA time / PSA time)", fontsize=10)
    ax.set_title("Speedup vs Selectivity", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Shade the "PSA wins" region.
    ax.axhspan(1.0, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 3.0,
               alpha=0.06, color=CPL, label="_nolegend_")

    # ── Right: SA ns/row vs selectivity + boundary collision rate ─────────────
    ax2 = axes[1]
    total_rows = df["matches"] / df["actual_sel"].replace(0, np.nan)
    sa_ns_per_row = df["sa_ns"] / total_rows.replace(0, np.nan)

    color_sa = C90
    ax2.plot(xs, sa_ns_per_row, "o-", color=color_sa, lw=2, ms=6,
             label="SA  ns/row")
    ax2.set_xlabel("Actual selectivity  (% rows matched)", fontsize=10)
    ax2.set_ylabel("SA time per row  (ns)", fontsize=10, color=color_sa)
    ax2.tick_params(axis="y", labelcolor=color_sa)
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax2.set_title("SA cost per row & Boundary Collision Rate", fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Secondary axis: boundary collision rate
    ax3 = ax2.twinx()
    ax3.plot(xs, df["bcr"] * 100, "D--", color=CFX, lw=1.5, ms=5,
             label="Boundary collision %")
    ax3.set_ylabel("Boundary collision rate  (%)", fontsize=9, color=CFX)
    ax3.tick_params(axis="y", labelcolor=CFX)
    ax3.set_ylim(bottom=0)

    # Combine legends
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")

    plt.tight_layout()

    safe = lambda s: s.replace("/", "_").replace(" ", "_").replace("#", "")
    fname = f"rank{rank:02d}_{safe(col_label)}.png"
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")
    return path

# ── Summary chart: all benchmarks on one plot ─────────────────────────────────
def plot_summary(df, out_dir):
    """PSA-PL speedup vs selectivity for all benchmarks on one chart."""
    range_df = df[df["query_type"] == "Range"].copy()
    if range_df.empty:
        print("  No Range queries found, skipping summary chart.")
        return

    # Aggregate: mean speedup per (target_sel bucket, benchmark column).
    range_df["sel_pct"] = (range_df["target_sel"] * 20).round() / 20 * 100

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axhline(1.0, color=CSA, lw=1.2, ls="--", label="SA baseline (1.0×)")

    cols = range_df.groupby(["rank", "benchmark", "table", "column"])
    cmap = plt.get_cmap("tab10")
    handles = []
    for idx, ((rank, bench, table, col), grp) in enumerate(cols):
        grp_sorted = grp.sort_values("sel_pct")
        label = f"R{rank}: {bench}/{col[:14]}"
        color = cmap(idx % 10)
        line, = ax.plot(
            grp_sorted["sel_pct"],
            grp_sorted["speedup_pl"],
            "o-", color=color, lw=1.8, ms=5, label=label
        )
        handles.append(line)

    ax.set_xlabel("Target selectivity  (%)", fontsize=11)
    ax.set_ylabel("PSA-PL speedup  (SA / PSA time)", fontsize=11)
    ax.set_title("Range Speedup vs Selectivity — all benchmarks (PSA-plateau)", fontsize=12)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(handles=handles, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    ax.axhspan(1.0, max(ax.get_ylim()[1], 1.5), alpha=0.06, color=CPL)

    plt.tight_layout()
    path = os.path.join(out_dir, "summary_range_speedup.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")

# ── SA ns/row summary: shows which columns are expensive for SA ───────────────
def plot_sa_cost_summary(df, out_dir):
    """SA time per row at each selectivity level, grouped by benchmark."""
    range_df = df[df["query_type"] == "Range"].copy()
    if range_df.empty:
        return

    range_df["sel_pct"]    = (range_df["target_sel"] * 20).round() / 20 * 100
    total_rows = range_df["matches"] / range_df["actual_sel"].replace(0, np.nan)
    range_df["sa_ns_row"]  = range_df["sa_ns"] / total_rows

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")
    cols = range_df.groupby(["rank", "benchmark", "table", "column"])
    for idx, ((rank, bench, table, col), grp) in enumerate(cols):
        grp_sorted = grp.sort_values("sel_pct")
        color = cmap(idx % 10)
        ax.plot(
            grp_sorted["sel_pct"],
            grp_sorted["sa_ns_row"],
            "o-", color=color, lw=1.8, ms=5,
            label=f"R{rank}: {bench}/{col[:14]}"
        )

    ax.set_xlabel("Target selectivity  (%)", fontsize=11)
    ax.set_ylabel("StringArray  ns / row", fontsize=11)
    ax.set_title("StringArray cost per row vs Selectivity\n"
                 "(higher = more bytes compared per row = more opportunity for PSA)", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "summary_sa_cost_per_row.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")

# ── BCR summary ───────────────────────────────────────────────────────────────
def plot_bcr_summary(df, out_dir):
    """Boundary collision rate at each selectivity level."""
    range_df = df[df["query_type"] == "Range"].copy()
    if range_df.empty:
        return

    range_df["sel_pct"] = (range_df["target_sel"] * 20).round() / 20 * 100

    fig, ax = plt.subplots(figsize=(11, 5))
    cmap = plt.get_cmap("tab10")
    cols = range_df.groupby(["rank", "benchmark", "table", "column"])
    for idx, ((rank, bench, table, col), grp) in enumerate(cols):
        grp_sorted = grp.sort_values("sel_pct")
        color = cmap(idx % 10)
        ax.plot(
            grp_sorted["sel_pct"],
            grp_sorted["bcr"] * 100,
            "o-", color=color, lw=1.8, ms=5,
            label=f"R{rank}: {bench}/{col[:14]}"
        )

    ax.set_xlabel("Target selectivity  (%)", fontsize=11)
    ax.set_ylabel("Boundary collision rate  (%)", fontsize=11)
    ax.set_title("Boundary Collision Rate vs Selectivity\n"
                 "(rows where prefix == range bound → need full suffix check)", fontsize=11)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(out_dir, "summary_boundary_collision.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Plot PSA Range benchmark results")
    p.add_argument("csv", help="Path to results.csv produced by bench_runner")
    p.add_argument("--out", default="plots", help="Output directory for PNGs (default: ./plots)")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Benchmarks: {df['benchmark'].unique().tolist()}")
    print(f"Query types: {df['query_type'].unique().tolist()}")
    print()

    # ── Per-benchmark plots ───────────────────────────────────────────────────
    print("Per-benchmark charts:")
    for (rank, bench, table, col), grp in df.groupby(["rank", "benchmark", "table", "column"]):
        range_grp = grp[grp["query_type"] == "Range"]
        if len(range_grp) >= 2:
            plot_one(range_grp.copy(), args.out)

    # ── Summary plots ─────────────────────────────────────────────────────────
    print("\nSummary charts:")
    plot_summary(df, args.out)
    plot_sa_cost_summary(df, args.out)
    plot_bcr_summary(df, args.out)

    print(f"\nAll plots written to {args.out}/")

if __name__ == "__main__":
    main()
