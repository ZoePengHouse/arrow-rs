// Licensed under the Apache License, Version 2.0.

/// Configuration knobs for [`PrefixSplitBuilder`](crate::builder::PrefixSplitBuilder).
#[derive(Debug, Clone)]
pub struct PrefixSplitConfig {
    /// Number of rows per sealed segment.  When a segment accumulates this
    /// many rows it is sealed and a per-segment `prefix_len_bytes` is computed.
    /// The final tail segment (fewer rows) is left unsealed and uses full-string
    /// comparison.
    ///
    /// Default: `1_000`.
    pub segment_size: usize,

    /// Fraction of distinct non-null values in a segment that must be
    /// distinguishable by prefix before we stop extending `prefix_len`.
    ///
    /// Formally: we stop when
    /// `distinct_prefixes(P) / distinct_values >= prefix_distinguish_threshold`.
    ///
    /// Default: `0.9` (90 %).
    pub prefix_distinguish_threshold: f64,

    /// Hard upper bound on the computed `prefix_len_bytes`.
    ///
    /// Default: `64`.
    pub max_prefix_len: usize,

    /// When `Some(n)`, skip the threshold algorithm entirely and use `n` as the
    /// prefix length for every sealed segment.  Useful for ablation studies
    /// ("what if we always use 3 bytes?") and for unit tests.
    ///
    /// Default: `None` (auto-compute from data).
    pub fixed_prefix_len: Option<usize>,

    /// Early-stop fraction for plateau detection.
    ///
    /// At each candidate prefix length `P` we compute the marginal gain:
    /// `gain(P) = distinct_prefixes(P) - distinct_prefixes(P-1)`.
    ///
    /// If ALL of the following hold, we conclude the curve has flattened and
    /// stop at `P` even if `prefix_distinguish_threshold` has not been reached:
    ///
    /// 1. `gain(P) < plateau_min_gain_fraction * total_distinct`
    /// 2. `distinct_prefixes(P) >= plateau_min_progress * target`
    ///    (we have already made meaningful progress toward the threshold)
    ///
    /// Condition 2 prevents premature stops on data with a long shared
    /// lead-in prefix (e.g. URLs "https://host/...") where the curve is flat
    /// near zero not because it has plateaued but because the distinctive
    /// region hasn't started yet.
    ///
    /// Set to `0.0` to disable.  Default: `0.01`.
    pub plateau_min_gain_fraction: f64,

    /// Minimum fraction of the threshold target that must already be reached
    /// before plateau detection is allowed to fire.
    ///
    /// Default: `0.5` (plateau can only trigger once we're ≥ 50 % of the way
    /// to `prefix_distinguish_threshold`).
    pub plateau_min_progress: f64,
}

impl Default for PrefixSplitConfig {
    fn default() -> Self {
        Self {
            segment_size: 1_000,
            prefix_distinguish_threshold: 0.9,
            max_prefix_len: 64,
            plateau_min_gain_fraction: 0.01,
            plateau_min_progress: 0.5,
            fixed_prefix_len: None,
        }
    }
}
