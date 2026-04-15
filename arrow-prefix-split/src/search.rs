// Licensed under the Apache License, Version 2.0.

//! Exact equality search over a [`PrefixSplitArray`].
//!
//! The core search function [`search_eq`] exploits the prefix layout to
//! minimise the number of suffix (long-string) memory accesses.
//!
//! # Search algorithm
//!
//! For each **sealed** segment (with a known `prefix_len = P`):
//!
//! 1. **Length filter** – skip any row whose stored original length differs
//!    from `target.len()`.  This is a trivial integer comparison.
//!
//! 2. **Prefix filter** – compare the first `min(target.len(), P)` bytes of
//!    the stored prefix slab against the same bytes of the target.  Because
//!    all prefixes for a segment live in one contiguous slab, this step is
//!    highly cache-friendly; a cache miss amortises over `P` rows instead of
//!    just one.
//!
//!    * If `target.len() <= P` and the prefix matches, the full value is
//!      contained in the prefix, so equality is confirmed without touching the
//!      suffix at all.
//!
//! 3. **Suffix confirmation** – only rows that survived the prefix filter are
//!    checked by comparing `suffix_data[suffix_offsets[i]..]` against
//!    `target[P..]`.
//!
//! For **unsealed** (tail) segments the search falls back to full
//! reconstruction + comparison (the suffix region stores the entire string).

use crate::array::PrefixSplitArray;

/// Returns the indices of all elements equal to `target`.
///
/// Null elements are always skipped.  The returned indices are in ascending
/// order.
///
/// # Example
///
/// ```rust
/// use arrow_prefix_split::{PrefixSplitBuilder, search_eq};
///
/// let mut b = PrefixSplitBuilder::new();
/// for v in ["cat", "dog", "catfish", "cat", "emu"] {
///     b.append_value(v);
/// }
/// let arr = b.finish();
/// assert_eq!(search_eq(&arr, "cat"), vec![0, 3]);
/// ```
pub fn search_eq(array: &PrefixSplitArray, target: &str) -> Vec<usize> {
    let target_bytes = target.as_bytes();
    let target_len = target_bytes.len();

    let mut result = Vec::new();

    for (seg_idx, seg) in array.segments().iter().enumerate() {
        if !seg.sealed {
            // ── Unsealed tail segment: full comparison ────────────────────
            for row_offset in 0..seg.len {
                let i = seg.start + row_offset;
                if array.is_null(i) {
                    continue;
                }
                if array.byte_length(i) != target_len {
                    continue;
                }
                // The unsealed segment stores the full string in the suffix region.
                if array.suffix_bytes(i) == target_bytes {
                    result.push(i);
                }
            }
        } else {
            // ── Sealed segment: prefix-first filtering ────────────────────
            let p = seg.prefix_len; // guaranteed ≥ 1 for sealed segments
            let t_cmp_len = target_len.min(p);
            let target_prefix = &target_bytes[..t_cmp_len];
            let slab = array.prefix_slab(seg_idx);

            for row_offset in 0..seg.len {
                let i = seg.start + row_offset;

                // ① Skip nulls.
                if array.is_null(i) {
                    continue;
                }

                // ② Length filter: O(1), avoids all memory traffic for misses.
                if array.byte_length(i) != target_len {
                    continue;
                }

                // ③ Prefix comparison (dense slab, cache-friendly).
                let slab_start = row_offset * p;
                let stored_prefix = &slab[slab_start..slab_start + t_cmp_len];
                if stored_prefix != target_prefix {
                    continue;
                }

                // ④ If the target fits entirely within the prefix region, the
                //    prefix+length checks already guarantee equality.
                if target_len <= p {
                    result.push(i);
                    continue;
                }

                // ⑤ Suffix confirmation (only reached by prefix-passing rows).
                if array.suffix_bytes(i) == &target_bytes[p..] {
                    result.push(i);
                }
            }
        }
    }

    result
}

/// Returns the count of elements equal to `target` without allocating a
/// result vector.
pub fn count_eq(array: &PrefixSplitArray, target: &str) -> usize {
    let target_bytes = target.as_bytes();
    let target_len = target_bytes.len();
    let mut count = 0usize;

    for (seg_idx, seg) in array.segments().iter().enumerate() {
        if !seg.sealed {
            for row_offset in 0..seg.len {
                let i = seg.start + row_offset;
                if array.is_null(i) || array.byte_length(i) != target_len {
                    continue;
                }
                if array.suffix_bytes(i) == target_bytes {
                    count += 1;
                }
            }
        } else {
            let p = seg.prefix_len;
            let t_cmp_len = target_len.min(p);
            let target_prefix = &target_bytes[..t_cmp_len];
            let slab = array.prefix_slab(seg_idx);

            for row_offset in 0..seg.len {
                let i = seg.start + row_offset;
                if array.is_null(i) || array.byte_length(i) != target_len {
                    continue;
                }
                let slab_start = row_offset * p;
                if &slab[slab_start..slab_start + t_cmp_len] != target_prefix {
                    continue;
                }
                if target_len <= p {
                    count += 1;
                    continue;
                }
                if array.suffix_bytes(i) == &target_bytes[p..] {
                    count += 1;
                }
            }
        }
    }

    count
}

/// Returns the indices of all elements that match **any** needle in `needles`
/// (i.e. `WHERE col IN ('a', 'b', 'c')`).
///
/// Makes a **single pass** over the array — one length/prefix check per row
/// regardless of how many needles are provided — so it is significantly faster
/// than calling [`search_eq`] once per needle and merging the results.
///
/// Null elements are always skipped.  The returned indices are in ascending
/// order and deduplicated.
///
/// # Example
///
/// ```rust
/// use arrow_prefix_split::{PrefixSplitBuilder, search_in};
///
/// let mut b = PrefixSplitBuilder::new();
/// for v in ["cat", "dog", "bird", "cat", "emu"] {
///     b.append_value(v);
/// }
/// let arr = b.finish();
/// assert_eq!(search_in(&arr, &["cat", "bird"]), vec![0, 2, 3]);
/// ```
pub fn search_in(array: &PrefixSplitArray, needles: &[&str]) -> Vec<usize> {
    if needles.is_empty() {
        return Vec::new();
    }

    // Pre-convert to bytes once, outside the hot loop.
    let needle_bytes: Vec<&[u8]> = needles.iter().map(|n| n.as_bytes()).collect();

    let mut result = Vec::new();

    for (seg_idx, seg) in array.segments().iter().enumerate() {
        if !seg.sealed {
            // ── Unsealed tail: full comparison for each needle ────────────
            for row_offset in 0..seg.len {
                let i = seg.start + row_offset;
                if array.is_null(i) {
                    continue;
                }
                let row_len = array.byte_length(i);
                let row_suffix = array.suffix_bytes(i); // full string in tail
                for nb in &needle_bytes {
                    if nb.len() == row_len && *nb == row_suffix {
                        result.push(i);
                        break;
                    }
                }
            }
        } else {
            // ── Sealed segment: prefix-first filtering for all needles ────
            let p = seg.prefix_len;
            let slab = array.prefix_slab(seg_idx);

            // Pre-slice needle prefixes for this segment's prefix_len.
            // Tuple: (full_len, cmp_len, prefix_slice, full_bytes).
            let info: Vec<(usize, usize, &[u8], &[u8])> = needle_bytes
                .iter()
                .map(|nb| {
                    let cmp = nb.len().min(p);
                    (nb.len(), cmp, &nb[..cmp], *nb)
                })
                .collect();

            for row_offset in 0..seg.len {
                let i = seg.start + row_offset;
                if array.is_null(i) {
                    continue;
                }
                let row_len = array.byte_length(i);
                let slab_start = row_offset * p;
                let row_prefix = &slab[slab_start..slab_start + p];

                for &(needle_len, cmp_len, needle_prefix, needle_full) in &info {
                    // ① Length filter.
                    if row_len != needle_len {
                        continue;
                    }
                    // ② Prefix filter (cmp_len = min(needle_len, p)).
                    if &row_prefix[..cmp_len] != needle_prefix {
                        continue;
                    }
                    // ③ Short needle: entirely within prefix — already confirmed.
                    if needle_len <= p {
                        result.push(i);
                        break;
                    }
                    // ④ Suffix confirmation.
                    if array.suffix_bytes(i) == &needle_full[p..] {
                        result.push(i);
                        break;
                    }
                }
            }
        }
    }

    result
}

/// Returns the indices of all **non-null** elements that are **not** equal to
/// `target` (i.e. `WHERE col != 'value'`).
///
/// Makes a single pass; rows are emitted in ascending order.
///
/// # Example
///
/// ```rust
/// use arrow_prefix_split::{PrefixSplitBuilder, search_ne};
///
/// let mut b = PrefixSplitBuilder::new();
/// for v in ["cat", "dog", "cat"] {
///     b.append_value(v);
/// }
/// b.append_null();
/// let arr = b.finish();
/// assert_eq!(search_ne(&arr, "cat"), vec![1]); // "dog"; null at idx 3 excluded
/// ```
pub fn search_ne(array: &PrefixSplitArray, target: &str) -> Vec<usize> {
    let target_bytes = target.as_bytes();
    let target_len = target_bytes.len();

    let mut result = Vec::new();

    for (seg_idx, seg) in array.segments().iter().enumerate() {
        if !seg.sealed {
            for row_offset in 0..seg.len {
                let i = seg.start + row_offset;
                if array.is_null(i) {
                    continue;
                }
                // Not equal if length differs OR full bytes differ.
                let row_len = array.byte_length(i);
                if row_len != target_len || array.suffix_bytes(i) != target_bytes {
                    result.push(i);
                }
            }
        } else {
            let p = seg.prefix_len;
            let t_cmp_len = target_len.min(p);
            let target_prefix = &target_bytes[..t_cmp_len];
            let slab = array.prefix_slab(seg_idx);

            for row_offset in 0..seg.len {
                let i = seg.start + row_offset;
                if array.is_null(i) {
                    continue;
                }

                // ① Length filter: if lengths differ → definitely not equal.
                if array.byte_length(i) != target_len {
                    result.push(i);
                    continue;
                }

                // ② Prefix filter: if prefixes differ → definitely not equal.
                let slab_start = row_offset * p;
                let stored_prefix = &slab[slab_start..slab_start + t_cmp_len];
                if stored_prefix != target_prefix {
                    result.push(i);
                    continue;
                }

                // ③ Short target fits in prefix: prefix + length match → equal → skip.
                if target_len <= p {
                    continue;
                }

                // ④ Suffix check: equal only when suffix also matches.
                if array.suffix_bytes(i) != &target_bytes[p..] {
                    result.push(i);
                }
            }
        }
    }

    result
}

/// Returns the indices of all **non-null** elements in the range
/// `[low, high]` / `(low, high]` / `[low, high)` / `(low, high)`
/// (i.e. `WHERE col >= 'a' AND col <= 'z'`).
///
/// Comparison is byte-lexicographic, which matches Unicode code-point order
/// for valid UTF-8 strings.
///
/// The prefix slab is used as a three-way filter per row:
/// * Prefix **strictly below** low → skip (no reconstruction needed).
/// * Prefix **strictly above** high → skip (no reconstruction needed).
/// * Prefix **strictly between** low and high → include directly (no reconstruction).
/// * Prefix **equals** one of the boundaries → fall through to full `value(i)`
///   reconstruction for the exact check.
///
/// Returns indices in ascending order.
///
/// # Example
///
/// ```rust
/// use arrow_prefix_split::{PrefixSplitBuilder, search_range};
///
/// let mut b = PrefixSplitBuilder::new();
/// for v in ["apple", "banana", "cherry", "date"] {
///     b.append_value(v);
/// }
/// let arr = b.finish();
/// // "banana" <= v <= "cherry"
/// assert_eq!(search_range(&arr, "banana", true, "cherry", true), vec![1, 2]);
/// ```
pub fn search_range(
    array: &PrefixSplitArray,
    low: &str,
    low_inclusive: bool,
    high: &str,
    high_inclusive: bool,
) -> Vec<usize> {
    let low_bytes = low.as_bytes();
    let high_bytes = high.as_bytes();

    let mut result = Vec::new();

    for (seg_idx, seg) in array.segments().iter().enumerate() {
        if !seg.sealed {
            // ── Unsealed tail: full comparison ────────────────────────────
            for row_offset in 0..seg.len {
                let i = seg.start + row_offset;
                if array.is_null(i) {
                    continue;
                }
                // In the unsealed tail, suffix_bytes holds the full string.
                let v = array.suffix_bytes(i);
                if bytes_in_range(v, low_bytes, low_inclusive, high_bytes, high_inclusive) {
                    result.push(i);
                }
            }
        } else {
            // ── Sealed segment: prefix-accelerated range filter ───────────
            let p = seg.prefix_len;
            let slab = array.prefix_slab(seg_idx);

            for row_offset in 0..seg.len {
                let i = seg.start + row_offset;
                if array.is_null(i) {
                    continue;
                }

                let row_len = array.byte_length(i);
                // Actual prefix bytes (remainder of slab slot is zero-padding).
                let effective = row_len.min(p);
                let row_pfx = &slab[row_offset * p..row_offset * p + effective];

                // ── Compare prefix against lower bound ────────────────────
                let vs_low = if low_bytes.is_empty() {
                    std::cmp::Ordering::Greater // every value passes an empty lower bound
                } else {
                    let k = effective.min(low_bytes.len());
                    row_pfx[..k].cmp(&low_bytes[..k])
                };

                // ── Compare prefix against upper bound ────────────────────
                let vs_high = if high_bytes.is_empty() {
                    std::cmp::Ordering::Less // every value passes an empty upper bound
                } else {
                    let k = effective.min(high_bytes.len());
                    row_pfx[..k].cmp(&high_bytes[..k])
                };

                // Prefix is definitively below low: skip.
                if vs_low == std::cmp::Ordering::Less {
                    continue;
                }
                // Prefix is definitively above high: skip.
                if vs_high == std::cmp::Ordering::Greater {
                    continue;
                }

                // Prefix is strictly between low and high: definitely in range.
                if vs_low == std::cmp::Ordering::Greater
                    && vs_high == std::cmp::Ordering::Less
                {
                    result.push(i);
                    continue;
                }

                // Boundary case: prefix touches one of the bounds.
                // Must reconstruct to get the exact value.
                let full = array.value(i);
                let vb = full.as_bytes();
                if bytes_in_range(vb, low_bytes, low_inclusive, high_bytes, high_inclusive) {
                    result.push(i);
                }
            }
        }
    }

    result
}

/// Helper: byte-lexicographic range check.
#[inline]
fn bytes_in_range(v: &[u8], low: &[u8], low_inc: bool, high: &[u8], high_inc: bool) -> bool {
    let ok_low = if low.is_empty() {
        true
    } else if low_inc {
        v >= low
    } else {
        v > low
    };
    let ok_high = if high.is_empty() {
        true
    } else if high_inc {
        v <= high
    } else {
        v < high
    };
    ok_low && ok_high
}

/// Returns the first index of an element equal to `target`, or `None`.
pub fn find_first_eq(array: &PrefixSplitArray, target: &str) -> Option<usize> {
    let target_bytes = target.as_bytes();
    let target_len = target_bytes.len();

    for (seg_idx, seg) in array.segments().iter().enumerate() {
        if !seg.sealed {
            for row_offset in 0..seg.len {
                let i = seg.start + row_offset;
                if array.is_null(i) || array.byte_length(i) != target_len {
                    continue;
                }
                if array.suffix_bytes(i) == target_bytes {
                    return Some(i);
                }
            }
        } else {
            let p = seg.prefix_len;
            let t_cmp_len = target_len.min(p);
            let target_prefix = &target_bytes[..t_cmp_len];
            let slab = array.prefix_slab(seg_idx);

            for row_offset in 0..seg.len {
                let i = seg.start + row_offset;
                if array.is_null(i) || array.byte_length(i) != target_len {
                    continue;
                }
                let slab_start = row_offset * p;
                if &slab[slab_start..slab_start + t_cmp_len] != target_prefix {
                    continue;
                }
                if target_len <= p {
                    return Some(i);
                }
                if array.suffix_bytes(i) == &target_bytes[p..] {
                    return Some(i);
                }
            }
        }
    }

    None
}
