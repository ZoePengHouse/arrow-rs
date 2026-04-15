// Licensed under the Apache License, Version 2.0.

use std::collections::HashSet;

use crate::config::PrefixSplitConfig;

/// Metadata describing one segment of a [`PrefixSplitArray`](crate::array::PrefixSplitArray).
///
/// A segment covers rows `start .. start + len`.  Once it is **sealed**
/// (`sealed == true`) a `prefix_len` has been computed and the corresponding
/// prefix slab is stored in `PrefixSplitArray::prefix_data`.  The last
/// (tail) segment may be unsealed (`sealed == false`, `prefix_len == 0`);
/// those rows fall back to full-string comparison during search.
#[derive(Debug, Clone)]
pub struct SegmentMeta {
    /// Absolute index of the first row in this segment.
    pub start: usize,
    /// Number of rows in this segment.
    pub len: usize,
    /// Computed prefix length in bytes.  `0` when `sealed == false`.
    pub prefix_len: usize,
    /// Whether this segment has been sealed (prefix computed).
    pub sealed: bool,
}

impl SegmentMeta {
    /// Creates a fresh unsealed segment starting at `start`.
    #[allow(dead_code)]
    pub(crate) fn new_unsealed(start: usize) -> Self {
        Self {
            start,
            len: 0,
            prefix_len: 0,
            sealed: false,
        }
    }
}

// ── Prefix length computation ─────────────────────────────────────────────────

/// Compute the minimum prefix length (in bytes) such that the fraction of
/// distinct non-null values in `values` that are distinguishable by their
/// prefix reaches `config.prefix_distinguish_threshold`.
///
/// "Distinguishable by prefix" is measured as:
///
/// ```text
/// count(distinct prefixes of length P) / count(distinct values) >= threshold
/// ```
///
/// We iterate P from 1 up to `config.max_prefix_len`, stopping as soon as
/// the threshold is met.  If the threshold is never met within
/// `max_prefix_len` we return `max_prefix_len`.
///
/// Special cases:
/// * All values are null → returns 1 (no non-null data to distinguish).
/// * Only one distinct value → returns 1 (trivially distinguishable).
pub fn compute_prefix_len(values: &[Option<Vec<u8>>], config: &PrefixSplitConfig) -> usize {
    // Collect references to non-null byte strings.
    let non_null: Vec<&[u8]> = values.iter().filter_map(|v| v.as_deref()).collect();

    if non_null.is_empty() {
        return 1;
    }

    // Count distinct non-null values.
    let distinct_set: HashSet<&[u8]> = non_null.iter().copied().collect();
    let total_distinct = distinct_set.len();

    // One or fewer distinct values: any prefix (≥ 1) is enough.
    if total_distinct <= 1 {
        return 1;
    }

    // Target: number of distinct prefixes we need to reach the threshold.
    // We use ceiling so that, e.g., 10 values at 0.9 threshold requires 9.
    let target = (total_distinct as f64 * config.prefix_distinguish_threshold).ceil() as usize;

    // Minimum absolute gain (in distinct-prefix count) below which we declare
    // the curve has plateaued and stop early.
    let min_gain = if config.plateau_min_gain_fraction > 0.0 {
        ((total_distinct as f64 * config.plateau_min_gain_fraction).ceil() as usize).max(1)
    } else {
        0 // plateau detection disabled
    };

    let mut prev_distinct = 0usize;

    for prefix_len in 1..=config.max_prefix_len {
        let distinct_prefixes = count_distinct_prefixes(&non_null, prefix_len);

        // ① Hit the threshold: done.
        if distinct_prefixes >= target {
            return prefix_len;
        }

        // ② Plateau detection (only after the first step so we have a gain to measure).
        //    Guard: also require minimum progress toward the threshold, so we don't
        //    fire early on data with a long shared lead-in prefix (step-function
        //    curve) where the curve is flat near zero before the unique region starts.
        if min_gain > 0 && prefix_len > 1 {
            let gain = distinct_prefixes - prev_distinct;
            let min_progress_count =
                (target as f64 * config.plateau_min_progress).ceil() as usize;
            if gain < min_gain && distinct_prefixes >= min_progress_count {
                return prefix_len;
            }
        }

        prev_distinct = distinct_prefixes;
    }

    config.max_prefix_len
}

/// Count how many distinct prefixes of `prefix_len` bytes exist across
/// the given (non-null) byte-string slice.  Strings shorter than
/// `prefix_len` are treated as their full content (no zero-padding needed
/// for distinctness counting).
fn count_distinct_prefixes(values: &[&[u8]], prefix_len: usize) -> usize {
    let mut seen: HashSet<&[u8]> = HashSet::with_capacity(values.len());
    for v in values {
        seen.insert(&v[..v.len().min(prefix_len)]);
    }
    seen.len()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn b(s: &str) -> Option<Vec<u8>> {
        Some(s.as_bytes().to_vec())
    }

    #[test]
    fn test_all_null() {
        let cfg = PrefixSplitConfig::default();
        let values = vec![None, None, None];
        assert_eq!(compute_prefix_len(&values, &cfg), 1);
    }

    #[test]
    fn test_single_distinct() {
        let cfg = PrefixSplitConfig::default();
        let values = vec![b("apple"), b("apple"), b("apple")];
        assert_eq!(compute_prefix_len(&values, &cfg), 1);
    }

    #[test]
    fn test_two_distinct_differ_at_byte_0() {
        let cfg = PrefixSplitConfig::default();
        let values = vec![b("apple"), b("banana")];
        // 'a' vs 'b': distinguishable at byte 0 → prefix_len = 1
        assert_eq!(compute_prefix_len(&values, &cfg), 1);
    }

    #[test]
    fn test_two_distinct_differ_at_byte_3() {
        // Disable plateau so we test pure threshold behaviour.
        let cfg = PrefixSplitConfig { plateau_min_gain_fraction: 0.0, ..Default::default() };
        let values = vec![b("apple"), b("appww")];
        // "appl" vs "appw": differ at index 3 → prefix_len = 4
        assert_eq!(compute_prefix_len(&values, &cfg), 4);
    }

    #[test]
    fn test_90_percent_threshold() {
        // 10 distinct values; at prefix_len=1: 9 distinct prefixes (90%) → stop
        let cfg = PrefixSplitConfig {
            prefix_distinguish_threshold: 0.9,
            max_prefix_len: 64,
            ..Default::default()
        };
        // Values: a0..a8 (same prefix 'a') + b0
        // at P=1: {"a","b"} = 2 out of 10 distinct — NOT enough.
        // But among distinct values, "a0".."a8" have prefix "a" (1 group),
        // "b0" has prefix "b" (1 group) → 2 distinct prefixes out of 10 distinct
        // values = 20%.  That won't reach 90%.
        //
        // Let's use values that mostly differ at byte 0:
        // "aa", "ba", "ca", "da", "ea", "fa", "ga", "ha", "ia", "aaz"
        // At P=1: "a","b","c","d","e","f","g","h","i" = 9 distinct out of 10
        // 9/10 = 90% → stop at P=1.
        let values: Vec<Option<Vec<u8>>> = vec![
            b("aa"), b("ba"), b("ca"), b("da"), b("ea"),
            b("fa"), b("ga"), b("ha"), b("ia"), b("aaz"),
        ];
        assert_eq!(compute_prefix_len(&values, &cfg), 1);
    }

    #[test]
    fn test_threshold_requires_longer_prefix() {
        // All strings share the same first byte.  We need to go deeper.
        let cfg = PrefixSplitConfig {
            prefix_distinguish_threshold: 0.9,
            max_prefix_len: 64,
            plateau_min_gain_fraction: 0.0, // disable plateau; test threshold only
            ..Default::default()
        };
        // "apple", "applet", "appleton", "application" → all start with "app"
        // P=1: 1 distinct prefix / 4 distinct → 25% — not enough
        // P=2: 1 distinct prefix (all "ap") → 25% — not enough
        // P=3: 1 distinct prefix "app" → not enough
        // P=4: "appl" vs "appl" vs "appl" vs "appl" — still 1 → not enough
        // P=5: "apple" vs "apple" vs "apple" vs "appli" → 2/4 = 50% — not enough
        // P=6: "applet" vs "applet" vs "applet" vs "applic" — 2/4 — not enough
        // P=7: "applet" vs "appleto" vs "applic" → 3/4=75% — not enough
        // P=8: "applet\0" is shorter — actual: "applet" < 8 → take full
        //       "appleton"[..8] = "appleton", "application"[..8] = "applicat"
        //       → {"apple", "applet", "appleton", "applicat"} = 4/4 = 100% ≥ 90%
        //       Stops at 8.
        let values: Vec<Option<Vec<u8>>> = vec![
            b("apple"),
            b("applet"),
            b("appleton"),
            b("application"),
        ];
        let plen = compute_prefix_len(&values, &cfg);
        // Verify: at plen, we distinguish ≥ 90% of 4 distinct values (i.e., ≥ 4)
        let non_null: Vec<&[u8]> = values.iter().filter_map(|v| v.as_deref()).collect();
        let dp = count_distinct_prefixes(&non_null, plen);
        let total = non_null.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(
            dp as f64 / total as f64 >= 0.9,
            "prefix_len={plen} gives {dp}/{total} distinct"
        );
    }

    #[test]
    fn test_max_prefix_len_cap() {
        let cfg = PrefixSplitConfig {
            prefix_distinguish_threshold: 1.0, // 100% — impossible for many shared prefixes
            max_prefix_len: 3,
            plateau_min_gain_fraction: 0.0, // disable plateau; test cap only
            ..Default::default()
        };
        // "abcdef" and "abcghi" differ at byte 3 — but max_prefix_len=3 means we
        // cap at P=3 even though it doesn't fully distinguish.
        let values: Vec<Option<Vec<u8>>> = vec![b("abcdef"), b("abcghi")];
        assert_eq!(compute_prefix_len(&values, &cfg), 3);
    }
}
