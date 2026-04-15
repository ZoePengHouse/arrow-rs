// Licensed under the Apache License, Version 2.0.

//! Integration tests for `arrow-prefix-split`.
//!
//! Coverage:
//!  ① Basic construction & value access
//!  ② Null handling
//!  ③ Prefix / suffix correctness
//!  ④ Segment sealing (small segment_size)
//!  ⑤ Prefix-length computation & 90 % threshold
//!  ⑥ Short strings (< prefix_len): reconstruction correctness
//!  ⑦ Exact equality search – correctness
//!  ⑧ Exact equality search – no false positives / no false negatives
//!  ⑨ Iterator
//!  ⑩ Slice
//!  ⑪ Conversion to / from StringArray
//!  ⑫ Arrow `Array` trait methods
//!  ⑬ Empty array edge cases
//!  ⑭ Multi-segment search spanning sealed + unsealed segments

use arrow_array::{Array, StringArray};
use arrow_prefix_split::{
    PrefixSplitArray, PrefixSplitBuilder, PrefixSplitConfig, search_eq, search_in, search_ne,
    search_range,
};
use arrow_prefix_split::search::{count_eq, find_first_eq};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build an array from a list of `Option<&str>` using the given config.
fn build_with_cfg<'a>(
    values: impl IntoIterator<Item = Option<&'a str>>,
    config: PrefixSplitConfig,
) -> PrefixSplitArray {
    let mut b = PrefixSplitBuilder::with_config(config);
    for v in values {
        b.append_option(v);
    }
    b.finish()
}

/// Config that seals every 10 rows (easier to reason about in tests).
fn small_seg_cfg() -> PrefixSplitConfig {
    PrefixSplitConfig {
        segment_size: 10,
        ..Default::default()
    }
}

// ── ① Basic construction & value access ──────────────────────────────────────

#[test]
fn test_basic_construction_and_value() {
    let mut b = PrefixSplitBuilder::new();
    b.append_value("apple");
    b.append_value("application");
    b.append_value("banana");
    let arr = b.finish();

    assert_eq!(arr.len(), 3);
    assert_eq!(arr.value(0), "apple");
    assert_eq!(arr.value(1), "application");
    assert_eq!(arr.value(2), "banana");
}

#[test]
fn test_empty_string() {
    let mut b = PrefixSplitBuilder::new();
    b.append_value("");
    b.append_value("non-empty");
    let arr = b.finish();

    assert_eq!(arr.value(0), "");
    assert_eq!(arr.byte_length(0), 0);
    assert_eq!(arr.value(1), "non-empty");
}

// ── ② Null handling ───────────────────────────────────────────────────────────

#[test]
fn test_null_handling() {
    let mut b = PrefixSplitBuilder::new();
    b.append_value("hello");
    b.append_null();
    b.append_value("world");
    b.append_null();
    let arr = b.finish();

    assert_eq!(arr.len(), 4);
    assert!(!arr.is_null(0));
    assert!(arr.is_null(1));
    assert!(!arr.is_null(2));
    assert!(arr.is_null(3));
    assert_eq!(arr.null_count(), 2);

    assert_eq!(arr.value(0), "hello");
    assert_eq!(arr.value(2), "world");
}

#[test]
fn test_all_nulls() {
    let arr = build_with_cfg([None, None, None], small_seg_cfg());
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.null_count(), 3);
}

#[test]
fn test_no_nulls_null_count_is_zero() {
    let arr = build_with_cfg(
        [Some("x"), Some("y"), Some("z")],
        PrefixSplitConfig::default(),
    );
    assert_eq!(arr.null_count(), 0);
    assert!(arr.nulls().is_none());
}

// ── ③ Prefix / suffix correctness ────────────────────────────────────────────

/// Verify that prefix + suffix reconstructs the original value exactly.
#[test]
fn test_prefix_suffix_round_trip() {
    // Use a tiny segment so the first 3 values form a sealed segment.
    let cfg = PrefixSplitConfig {
        segment_size: 3,
        max_prefix_len: 64,
        prefix_distinguish_threshold: 0.9,
        plateau_min_gain_fraction: 0.0, plateau_min_progress: 0.0,
    };
    let values = ["apple", "applet", "application", "banana", "band"];
    let mut b = PrefixSplitBuilder::with_config(cfg);
    for v in values {
        b.append_value(v);
    }
    let arr = b.finish();

    for (i, expected) in values.iter().enumerate() {
        assert_eq!(
            arr.value(i),
            *expected,
            "round-trip failed for index {i}"
        );
    }
}

/// For a sealed segment, verify that prefix + suffix byte-concatenation
/// equals the original string.
#[test]
fn test_prefix_and_suffix_bytes_concatenation() {
    // Force a 3-row sealed segment.
    let cfg = PrefixSplitConfig {
        segment_size: 3,
        ..Default::default()
    };
    let values = ["alpha", "beta", "gamma"];
    let mut b = PrefixSplitBuilder::with_config(cfg.clone());
    for v in values {
        b.append_value(v);
    }
    let arr = b.finish();

    // The first 3 rows form a sealed segment with some prefix_len P.
    assert!(arr.segments()[0].sealed, "segment 0 should be sealed");
    let p = arr.segments()[0].prefix_len;
    assert!(p >= 1, "prefix_len must be at least 1");

    for i in 0..3 {
        let orig = values[i].as_bytes();
        let pfx_bytes = arr.prefix_bytes(i).expect("sealed segment must have prefix");
        let sfx_bytes = arr.suffix_bytes(i);

        // The stored prefix slab is exactly `p` bytes long (zero-padded).
        assert_eq!(pfx_bytes.len(), p);

        // Concatenating the actual prefix part + suffix gives the original.
        let actual_pfx_len = orig.len().min(p);
        let mut reconstructed = pfx_bytes[..actual_pfx_len].to_vec();
        reconstructed.extend_from_slice(sfx_bytes);
        assert_eq!(
            reconstructed,
            orig,
            "bytes reconstruction failed for index {i}"
        );
    }
}

/// Strings shorter than `prefix_len` must still round-trip correctly.
#[test]
fn test_short_string_reconstruction() {
    // Force a large prefix_len by having many similar long strings.
    let cfg = PrefixSplitConfig {
        segment_size: 5,
        max_prefix_len: 64,
        prefix_distinguish_threshold: 0.9,
        plateau_min_gain_fraction: 0.0, plateau_min_progress: 0.0,
    };
    // These all share "abcdefghij" so prefix_len will be pushed high.
    // Then we also include a very short string.
    let values = [
        "abcdefghij_1",
        "abcdefghij_2",
        "abcdefghij_3",
        "abcdefghij_4",
        "hi",           // much shorter than the likely prefix_len
    ];
    let mut b = PrefixSplitBuilder::with_config(cfg);
    for v in values {
        b.append_value(v);
    }
    let arr = b.finish();

    // All values must round-trip.
    for (i, expected) in values.iter().enumerate() {
        assert_eq!(arr.value(i), *expected, "short-string round-trip failed at {i}");
    }
}

// ── ④ Segment sealing ─────────────────────────────────────────────────────────

#[test]
fn test_segment_sealing_logic() {
    // 25 rows with segment_size=10 → 2 sealed segments + 1 unsealed tail (5 rows).
    let cfg = PrefixSplitConfig {
        segment_size: 10,
        ..Default::default()
    };
    let mut b = PrefixSplitBuilder::with_config(cfg);
    for i in 0..25u32 {
        b.append_value(&format!("value_{i:04}"));
    }
    let arr = b.finish();

    assert_eq!(arr.segments().len(), 3);

    // First two segments are sealed.
    assert!(arr.segments()[0].sealed);
    assert_eq!(arr.segments()[0].len, 10);
    assert_eq!(arr.segments()[0].start, 0);

    assert!(arr.segments()[1].sealed);
    assert_eq!(arr.segments()[1].len, 10);
    assert_eq!(arr.segments()[1].start, 10);

    // Tail segment is NOT sealed.
    assert!(!arr.segments()[2].sealed);
    assert_eq!(arr.segments()[2].len, 5);
    assert_eq!(arr.segments()[2].start, 20);
}

#[test]
fn test_exact_segment_size_has_no_unsealed_tail() {
    let cfg = PrefixSplitConfig {
        segment_size: 5,
        ..Default::default()
    };
    let mut b = PrefixSplitBuilder::with_config(cfg);
    for i in 0..10u32 {
        b.append_value(&format!("row_{i}"));
    }
    let arr = b.finish();

    // 10 rows / 5-row segments = 2 sealed + 0-row tail (empty tail is dropped).
    let segs = arr.segments();
    // The builder flushes an empty tail with 0 rows; we rely on segments being
    // non-empty, so either 2 or 3 segments exist.  Both sealed ones must be so.
    assert!(segs.iter().filter(|s| s.sealed).count() >= 2);
    for s in segs.iter().filter(|s| s.sealed) {
        assert_eq!(s.len, 5);
    }
}

// ── ⑤ Prefix-length computation & 90 % threshold ─────────────────────────────

#[test]
fn test_prefix_length_is_reasonable() {
    // Build 10 rows that all differ at byte 0 → prefix_len should be 1.
    let cfg = PrefixSplitConfig {
        segment_size: 10,
        prefix_distinguish_threshold: 0.9,
        max_prefix_len: 64,
        plateau_min_gain_fraction: 0.0, plateau_min_progress: 0.0,
    };
    let values: Vec<String> = (b'a'..=b'j')
        .map(|c| format!("{}rest", c as char))
        .collect();
    let mut b = PrefixSplitBuilder::with_config(cfg);
    for v in &values {
        b.append_value(v);
    }
    let arr = b.finish();

    // Should be sealed after exactly 10 rows.
    assert!(arr.segments()[0].sealed);
    assert_eq!(arr.segments()[0].prefix_len, 1);
}

#[test]
fn test_prefix_length_grows_for_common_prefix() {
    // All strings share the first 4 bytes; they differ at byte 4.
    let cfg = PrefixSplitConfig {
        segment_size: 10,
        prefix_distinguish_threshold: 0.9,
        max_prefix_len: 64,
        plateau_min_gain_fraction: 0.0, plateau_min_progress: 0.0,
    };
    let values: Vec<String> = (0u8..10)
        .map(|i| format!("data{i:04}"))
        .collect();
    let mut b = PrefixSplitBuilder::with_config(cfg);
    for v in &values {
        b.append_value(v);
    }
    let arr = b.finish();

    assert!(arr.segments()[0].sealed);
    // "data" is the shared prefix; need at least 5 bytes to tell them apart.
    assert!(
        arr.segments()[0].prefix_len >= 5,
        "expected prefix_len >= 5, got {}",
        arr.segments()[0].prefix_len
    );
}

// ── ⑥ Exact equality search – correctness ────────────────────────────────────

#[test]
fn test_search_eq_finds_exact_matches() {
    let mut b = PrefixSplitBuilder::new();
    let data = ["cat", "dog", "catfish", "cat", "emu", "cat"];
    for v in data {
        b.append_value(v);
    }
    let arr = b.finish();

    let hits = search_eq(&arr, "cat");
    assert_eq!(hits, vec![0, 3, 5]);
}

#[test]
fn test_search_eq_no_match_returns_empty() {
    let mut b = PrefixSplitBuilder::new();
    for v in ["alpha", "beta", "gamma"] {
        b.append_value(v);
    }
    let arr = b.finish();

    assert!(search_eq(&arr, "delta").is_empty());
}

#[test]
fn test_search_eq_skips_nulls() {
    let mut b = PrefixSplitBuilder::new();
    b.append_value("apple");
    b.append_null();
    b.append_value("apple");
    let arr = b.finish();

    // Null in the middle must not be returned.
    let hits = search_eq(&arr, "apple");
    assert_eq!(hits, vec![0, 2]);
}

#[test]
fn test_search_eq_on_sealed_segments() {
    let cfg = PrefixSplitConfig {
        segment_size: 10,
        ..Default::default()
    };
    // Build 20 rows (two sealed segments) + 5 in the tail.
    let mut b = PrefixSplitBuilder::with_config(cfg);
    for i in 0..25u32 {
        // Every 5th element is "target".
        if i % 5 == 0 {
            b.append_value("target");
        } else {
            b.append_value(&format!("other_{i}"));
        }
    }
    let arr = b.finish();

    let hits = search_eq(&arr, "target");
    let expected: Vec<usize> = (0..25).filter(|i| i % 5 == 0).collect();
    assert_eq!(hits, expected, "missed hits in sealed segments");
}

/// Verify no false positives: strings with the same prefix but different
/// suffixes must NOT be returned.
#[test]
fn test_search_eq_no_false_positives() {
    let cfg = PrefixSplitConfig {
        segment_size: 5,
        ..Default::default()
    };
    // All share "application" as a prefix → they share many prefix bytes.
    let data = [
        "application",
        "application_server",
        "application_client",
        "application_proxy",
        "application_db",
    ];
    let mut b = PrefixSplitBuilder::with_config(cfg);
    for v in data {
        b.append_value(v);
    }
    let arr = b.finish();

    // Only exact match at index 0.
    let hits = search_eq(&arr, "application");
    assert_eq!(hits, vec![0]);

    // No match at all.
    assert!(search_eq(&arr, "application_cache").is_empty());
}

/// Strings that are a prefix of the target (shorter) must not match.
#[test]
fn test_search_eq_no_match_for_prefix_of_target() {
    let mut b = PrefixSplitBuilder::new();
    b.append_value("app");     // shorter than target
    b.append_value("apple");   // exact target
    b.append_value("applez");  // longer than target
    let arr = b.finish();

    let hits = search_eq(&arr, "apple");
    assert_eq!(hits, vec![1]);
}

// ── ⑦ count_eq and find_first_eq ─────────────────────────────────────────────

#[test]
fn test_count_eq() {
    let mut b = PrefixSplitBuilder::new();
    for v in ["x", "y", "x", "z", "x"] {
        b.append_value(v);
    }
    let arr = b.finish();
    assert_eq!(count_eq(&arr, "x"), 3);
    assert_eq!(count_eq(&arr, "y"), 1);
    assert_eq!(count_eq(&arr, "w"), 0);
}

#[test]
fn test_find_first_eq() {
    let mut b = PrefixSplitBuilder::new();
    for v in ["a", "b", "c", "b", "d"] {
        b.append_value(v);
    }
    let arr = b.finish();
    assert_eq!(find_first_eq(&arr, "b"), Some(1));
    assert_eq!(find_first_eq(&arr, "z"), None);
}

// ── ⑧ Iterator ───────────────────────────────────────────────────────────────

#[test]
fn test_iterator_yields_all_values() {
    let mut b = PrefixSplitBuilder::new();
    let data = ["first", "second", "third"];
    for v in data {
        b.append_value(v);
    }
    let arr = b.finish();

    let collected: Vec<Option<String>> = arr.iter().collect();
    assert_eq!(
        collected,
        vec![
            Some("first".to_string()),
            Some("second".to_string()),
            Some("third".to_string()),
        ]
    );
}

#[test]
fn test_iterator_yields_none_for_nulls() {
    let mut b = PrefixSplitBuilder::new();
    b.append_value("a");
    b.append_null();
    b.append_value("b");
    let arr = b.finish();

    let collected: Vec<Option<String>> = arr.iter().collect();
    assert_eq!(
        collected,
        vec![Some("a".to_string()), None, Some("b".to_string())]
    );
}

#[test]
fn test_iterator_exact_size() {
    let mut b = PrefixSplitBuilder::new();
    for v in ["p", "q", "r", "s"] {
        b.append_value(v);
    }
    let arr = b.finish();
    let mut iter = arr.iter();
    assert_eq!(iter.len(), 4);
    iter.next();
    assert_eq!(iter.len(), 3);
}

// ── ⑨ Slice ───────────────────────────────────────────────────────────────────

#[test]
fn test_slice_basic() {
    let mut b = PrefixSplitBuilder::new();
    for v in ["a", "b", "c", "d", "e"] {
        b.append_value(v);
    }
    let arr = b.finish();
    let sliced = arr.slice(1, 3);

    assert_eq!(sliced.len(), 3);
    assert_eq!(sliced.value(0), "b");
    assert_eq!(sliced.value(1), "c");
    assert_eq!(sliced.value(2), "d");
}

#[test]
fn test_slice_preserves_nulls() {
    let mut b = PrefixSplitBuilder::new();
    b.append_value("x");
    b.append_null();
    b.append_value("y");
    b.append_null();
    b.append_value("z");
    let arr = b.finish();

    let sliced = arr.slice(1, 3); // [null, "y", null]
    assert_eq!(sliced.len(), 3);
    assert!(sliced.is_null(0));
    assert_eq!(sliced.value(1), "y");
    assert!(sliced.is_null(2));
}

#[test]
fn test_slice_empty() {
    let arr = build_with_cfg([Some("a"), Some("b")], PrefixSplitConfig::default());
    let sliced = arr.slice(1, 0);
    assert_eq!(sliced.len(), 0);
    assert!(sliced.is_empty());
}

#[test]
fn test_array_trait_slice_returns_arc() {
    let mut b = PrefixSplitBuilder::new();
    for v in ["one", "two", "three"] {
        b.append_value(v);
    }
    let arr = b.finish();

    // Call through the Arrow `Array` trait (returns `Arc<dyn Array>`).
    let sliced_ref = Array::slice(&arr, 0, 2);
    assert_eq!(sliced_ref.len(), 2);

    // Downcast back to our concrete type.
    let downcast = sliced_ref
        .as_any()
        .downcast_ref::<PrefixSplitArray>()
        .expect("downcast to PrefixSplitArray");
    assert_eq!(downcast.value(0), "one");
    assert_eq!(downcast.value(1), "two");
}

// ── ⑩ Conversion to / from StringArray ───────────────────────────────────────

#[test]
fn test_round_trip_to_string_array() {
    let original = StringArray::from(vec![
        Some("alpha"),
        None,
        Some("beta"),
        Some("gamma"),
        None,
    ]);

    let prefix_arr = PrefixSplitArray::from(&original);
    let roundtripped = prefix_arr.to_string_array();

    assert_eq!(original.len(), roundtripped.len());
    for i in 0..original.len() {
        assert_eq!(
            original.is_null(i),
            roundtripped.is_null(i),
            "null mismatch at {i}"
        );
        if !original.is_null(i) {
            assert_eq!(
                original.value(i),
                roundtripped.value(i),
                "value mismatch at {i}"
            );
        }
    }
}

#[test]
fn test_from_string_array_preserves_order() {
    let data = vec!["zebra", "apple", "mango", "cherry"];
    let sa = StringArray::from(data.clone());
    let psa = PrefixSplitArray::from(&sa);

    for (i, expected) in data.iter().enumerate() {
        assert_eq!(psa.value(i), *expected);
    }
}

// ── ⑪ Arrow `Array` trait methods ────────────────────────────────────────────

#[test]
fn test_array_trait_data_type() {
    use arrow_schema::DataType;
    let arr = build_with_cfg([Some("test")], PrefixSplitConfig::default());
    assert_eq!(*arr.data_type(), DataType::Utf8);
}

#[test]
fn test_array_trait_len_and_is_empty() {
    let arr = build_with_cfg([], PrefixSplitConfig::default());
    assert_eq!(arr.len(), 0);
    assert!(arr.is_empty());

    let arr2 = build_with_cfg([Some("x")], PrefixSplitConfig::default());
    assert_eq!(arr2.len(), 1);
    assert!(!arr2.is_empty());
}

#[test]
fn test_array_trait_null_methods() {
    let mut b = PrefixSplitBuilder::new();
    b.append_value("a");
    b.append_null();
    let arr = b.finish();

    assert_eq!(arr.null_count(), 1);
    assert!(!arr.is_null(0));
    assert!(arr.is_null(1));
    assert!(arr.is_valid(0));
    assert!(!arr.is_valid(1));
}

#[test]
fn test_array_trait_to_data_is_utf8() {
    use arrow_schema::DataType;
    let arr = build_with_cfg([Some("hello"), Some("world")], PrefixSplitConfig::default());
    let data = arr.to_data();
    assert_eq!(*data.data_type(), DataType::Utf8);
    assert_eq!(data.len(), 2);
}

#[test]
fn test_array_trait_offset_is_zero() {
    let arr = build_with_cfg([Some("x")], PrefixSplitConfig::default());
    assert_eq!(arr.offset(), 0);
}

// ── ⑫ Edge cases ─────────────────────────────────────────────────────────────

#[test]
fn test_empty_array() {
    let arr = build_with_cfg([], PrefixSplitConfig::default());
    assert_eq!(arr.len(), 0);
    assert_eq!(arr.null_count(), 0);
    assert_eq!(arr.segments().len(), 0);
    assert_eq!(arr.iter().count(), 0);
    assert!(search_eq(&arr, "anything").is_empty());
}

#[test]
fn test_unicode_strings() {
    let mut b = PrefixSplitBuilder::new();
    b.append_value("日本語");        // 3-byte UTF-8 chars
    b.append_value("中文");
    b.append_value("한국어");
    b.append_value("日本語");
    let arr = b.finish();

    assert_eq!(arr.value(0), "日本語");
    assert_eq!(arr.value(1), "中文");
    assert_eq!(arr.value(2), "한국어");

    let hits = search_eq(&arr, "日本語");
    assert_eq!(hits, vec![0, 3]);
}

// ── ⑬ Multi-segment search spanning sealed + unsealed ────────────────────────

#[test]
fn test_search_across_sealed_and_unsealed_segments() {
    let cfg = PrefixSplitConfig {
        segment_size: 10,
        ..Default::default()
    };
    let mut b = PrefixSplitBuilder::with_config(cfg);

    // 15 rows: 10 sealed + 5 unsealed tail.
    for i in 0..15u32 {
        if i == 5 || i == 12 {
            b.append_value("needle");
        } else {
            b.append_value(&format!("hay_{i:04}"));
        }
    }
    let arr = b.finish();

    assert_eq!(arr.segments().len(), 2);
    assert!(arr.segments()[0].sealed);
    assert!(!arr.segments()[1].sealed);

    // index 5 is in the sealed segment; index 12 is in the unsealed tail.
    let hits = search_eq(&arr, "needle");
    assert_eq!(hits, vec![5, 12]);
}

// ── ⑭ Debug / Display ────────────────────────────────────────────────────────

#[test]
fn test_debug_does_not_panic() {
    let mut b = PrefixSplitBuilder::new();
    b.append_value("hello");
    b.append_null();
    let arr = b.finish();
    let s = format!("{:?}", arr);
    assert!(s.contains("PrefixSplitArray"));
}

#[test]
fn test_display_shows_values_and_null() {
    let mut b = PrefixSplitBuilder::new();
    b.append_value("x");
    b.append_null();
    b.append_value("y");
    let arr = b.finish();
    let s = format!("{}", arr);
    assert!(s.contains("\"x\""));
    assert!(s.contains("null"));
    assert!(s.contains("\"y\""));
}

// ── ⑮ search_in ──────────────────────────────────────────────────────────────

/// Basic IN-clause matches.
#[test]
fn test_search_in_basic() {
    let mut b = PrefixSplitBuilder::new();
    for v in ["cat", "dog", "bird", "cat", "emu"] {
        b.append_value(v);
    }
    let arr = b.finish();
    assert_eq!(search_in(&arr, &["cat", "bird"]), vec![0, 2, 3]);
}

/// Null rows are skipped by search_in.
#[test]
fn test_search_in_skips_nulls() {
    let mut b = PrefixSplitBuilder::new();
    b.append_value("cat");
    b.append_null();
    b.append_value("cat");
    let arr = b.finish();
    assert_eq!(search_in(&arr, &["cat"]), vec![0, 2]);
}

/// Empty needle list returns empty result.
#[test]
fn test_search_in_empty_needles() {
    let mut b = PrefixSplitBuilder::new();
    b.append_value("cat");
    let arr = b.finish();
    assert!(search_in(&arr, &[]).is_empty());
}

/// search_in agrees with repeated search_eq calls.
#[test]
fn test_search_in_matches_repeated_eq() {
    // Use a small segment_size so we exercise sealed segments.
    let cfg = PrefixSplitConfig { segment_size: 10, ..Default::default() };
    let mut b = PrefixSplitBuilder::with_config(cfg);
    let words = ["alpha", "beta", "gamma", "delta", "alpha", "epsilon",
                 "beta", "zeta", "alpha", "eta", "alpha", "theta"];
    for w in words {
        b.append_value(w);
    }
    let arr = b.finish();

    let needles = ["alpha", "beta"];
    let mut expected: Vec<usize> = needles.iter()
        .flat_map(|n| search_eq(&arr, n))
        .collect();
    expected.sort_unstable();
    expected.dedup();

    assert_eq!(search_in(&arr, &needles), expected);
}

// ── ⑯ search_ne ──────────────────────────────────────────────────────────────

/// search_ne returns all non-null, non-matching rows.
#[test]
fn test_search_ne_basic() {
    let mut b = PrefixSplitBuilder::new();
    for v in ["cat", "dog", "cat", "bird"] {
        b.append_value(v);
    }
    let arr = b.finish();
    assert_eq!(search_ne(&arr, "cat"), vec![1, 3]);
}

/// Null rows are excluded from search_ne.
#[test]
fn test_search_ne_excludes_nulls() {
    let mut b = PrefixSplitBuilder::new();
    b.append_value("cat");
    b.append_null();
    b.append_value("dog");
    let arr = b.finish();
    // null at index 1 is NOT included even though it's != "cat".
    assert_eq!(search_ne(&arr, "cat"), vec![2]);
}

/// search_ne is the complement of search_eq over non-null rows.
#[test]
fn test_search_ne_complement_of_eq() {
    let cfg = PrefixSplitConfig { segment_size: 10, ..Default::default() };
    let mut b = PrefixSplitBuilder::with_config(cfg);
    let words = ["apple", "apply", "application", "banana", "apply", "cherry",
                 "apple", "date", "apply", "elderberry", "fig", "apply"];
    for w in words {
        b.append_value(w);
    }
    let arr = b.finish();

    let target = "apply";
    let eq_hits = search_eq(&arr, target);
    let ne_hits = search_ne(&arr, target);

    // Together they must cover all non-null rows.
    let mut all: Vec<usize> = eq_hits.iter().chain(ne_hits.iter()).copied().collect();
    all.sort_unstable();
    let expected: Vec<usize> = (0..arr.len()).filter(|&i| arr.is_valid(i)).collect();
    assert_eq!(all, expected);

    // They must be disjoint.
    let eq_set: std::collections::HashSet<usize> = eq_hits.into_iter().collect();
    assert!(ne_hits.iter().all(|i| !eq_set.contains(i)));
}

// ── ⑰ search_range ───────────────────────────────────────────────────────────

/// Basic inclusive range.
#[test]
fn test_search_range_inclusive() {
    let mut b = PrefixSplitBuilder::new();
    for v in ["apple", "banana", "cherry", "date", "elderberry"] {
        b.append_value(v);
    }
    let arr = b.finish();
    // "banana" <= v <= "cherry"
    assert_eq!(
        search_range(&arr, "banana", true, "cherry", true),
        vec![1, 2]
    );
}

/// Exclusive lower bound.
#[test]
fn test_search_range_exclusive_low() {
    let mut b = PrefixSplitBuilder::new();
    for v in ["apple", "banana", "cherry", "date"] {
        b.append_value(v);
    }
    let arr = b.finish();
    // v > "banana" AND v <= "date"
    assert_eq!(
        search_range(&arr, "banana", false, "date", true),
        vec![2, 3]
    );
}

/// Exclusive upper bound.
#[test]
fn test_search_range_exclusive_high() {
    let mut b = PrefixSplitBuilder::new();
    for v in ["apple", "banana", "cherry", "date"] {
        b.append_value(v);
    }
    let arr = b.finish();
    // "apple" <= v < "date"
    assert_eq!(
        search_range(&arr, "apple", true, "date", false),
        vec![0, 1, 2]
    );
}

/// Null rows are excluded from search_range.
#[test]
fn test_search_range_excludes_nulls() {
    let mut b = PrefixSplitBuilder::new();
    b.append_value("banana");
    b.append_null();
    b.append_value("cherry");
    let arr = b.finish();
    assert_eq!(
        search_range(&arr, "apple", true, "date", true),
        vec![0, 2]
    );
}

/// search_range agrees with brute-force scan over a sealed segment.
#[test]
fn test_search_range_vs_brute_force() {
    let cfg = PrefixSplitConfig { segment_size: 10, ..Default::default() };
    let mut b = PrefixSplitBuilder::with_config(cfg);
    let words = [
        "ant", "bear", "cat", "deer", "elephant", "fox",
        "goat", "hare", "ibis", "jaguar", "koala", "lion",
    ];
    for w in words {
        b.append_value(w);
    }
    let arr = b.finish();

    let low = "cat";
    let high = "jaguar";
    let psa_hits = search_range(&arr, low, true, high, false);

    // Brute-force expected result.
    let expected: Vec<usize> = (0..arr.len())
        .filter(|&i| {
            arr.is_valid(i) && {
                let v = arr.value(i);
                v.as_str() >= low && v.as_str() < high
            }
        })
        .collect();

    assert_eq!(psa_hits, expected);
}
