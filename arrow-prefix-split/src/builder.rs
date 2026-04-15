// Licensed under the Apache License, Version 2.0.

use arrow_buffer::NullBufferBuilder;

use crate::array::PrefixSplitArray;
use crate::config::PrefixSplitConfig;
use crate::segment::{SegmentMeta, compute_prefix_len};

/// Builder for [`PrefixSplitArray`].
///
/// Values are appended one at a time.  Internally the builder accumulates a
/// **pending** buffer for the segment currently being built.  When the pending
/// buffer reaches `config.segment_size` rows the segment is *sealed*:
///
/// 1. `compute_prefix_len` determines the optimal prefix length (≥ 90 %
///    threshold by default).
/// 2. For each value in the segment the first `prefix_len` bytes are written
///    into a fixed-width slab; remaining bytes go into the shared suffix
///    buffers.
///
/// When [`finish`](Self::finish) is called the remaining (tail) pending values
/// are flushed as an *unsealed* segment (prefix_len = 0, full strings stored
/// in the suffix region).
///
/// # Example
///
/// ```rust
/// use arrow_prefix_split::PrefixSplitBuilder;
///
/// let mut b = PrefixSplitBuilder::new();
/// b.append_value("hello");
/// b.append_null();
/// b.append_value("world");
/// let array = b.finish();
/// assert_eq!(array.len(), 3);
/// assert_eq!(array.value(0), "hello");
/// assert!(array.is_null(1));
/// ```
pub struct PrefixSplitBuilder {
    config: PrefixSplitConfig,

    // ── Completed (sealed + unsealed-tail) segment data ───────────────────
    segments: Vec<SegmentMeta>,
    /// One entry per segment: `seg.len * seg.prefix_len` bytes.
    /// Empty vec for unsealed segments.
    prefix_data: Vec<Vec<u8>>,
    /// Original byte length of every element across all flushed segments.
    lengths: Vec<i32>,
    /// Concatenated suffix bytes of all flushed elements.
    suffix_data: Vec<u8>,
    /// Suffix offset table: `suffix_offsets[i]..suffix_offsets[i+1]` is the
    /// suffix for element `i`.  Length = total_flushed + 1.
    suffix_offsets: Vec<i32>,
    null_builder: NullBufferBuilder,
    total_len: usize,

    // ── Pending: current in-progress segment (not yet sealed) ─────────────
    /// Raw string bytes (or None for null) for the rows not yet flushed.
    pending: Vec<Option<Vec<u8>>>,
}

impl PrefixSplitBuilder {
    /// Creates a builder with default configuration (1 000 rows / segment,
    /// 90 % threshold, 64-byte max prefix).
    pub fn new() -> Self {
        Self::with_config(PrefixSplitConfig::default())
    }

    /// Creates a builder with a custom configuration.
    pub fn with_config(config: PrefixSplitConfig) -> Self {
        Self {
            config,
            segments: Vec::new(),
            prefix_data: Vec::new(),
            lengths: Vec::new(),
            suffix_data: Vec::new(),
            suffix_offsets: vec![0i32],
            null_builder: NullBufferBuilder::new(0),
            total_len: 0,
            pending: Vec::new(),
        }
    }

    // ── Public append API ─────────────────────────────────────────────────

    /// Appends a non-null UTF-8 string.
    pub fn append_value(&mut self, v: &str) {
        self.null_builder.append_non_null();
        self.pending.push(Some(v.as_bytes().to_vec()));
        self.total_len += 1;
        self.maybe_seal();
    }

    /// Appends a null element.
    pub fn append_null(&mut self) {
        self.null_builder.append_null();
        self.pending.push(None);
        self.total_len += 1;
        self.maybe_seal();
    }

    /// Appends `Some(v)` or null.
    pub fn append_option(&mut self, v: Option<&str>) {
        match v {
            Some(s) => self.append_value(s),
            None => self.append_null(),
        }
    }

    /// Appends multiple non-null strings.
    pub fn extend<'a, I: IntoIterator<Item = &'a str>>(&mut self, iter: I) {
        for s in iter {
            self.append_value(s);
        }
    }

    /// Returns the number of elements appended so far.
    pub fn len(&self) -> usize {
        self.total_len
    }

    /// Returns `true` if no elements have been appended.
    pub fn is_empty(&self) -> bool {
        self.total_len == 0
    }

    // ── Build ─────────────────────────────────────────────────────────────

    /// Finalizes the array.
    ///
    /// Any pending (tail) rows are flushed as an **unsealed** segment.
    pub fn finish(mut self) -> PrefixSplitArray {
        self.flush_pending(false);

        let nulls = self.null_builder.build();

        PrefixSplitArray {
            len: self.total_len,
            nulls,
            segments: self.segments,
            prefix_data: self.prefix_data,
            lengths: self.lengths,
            suffix_data: self.suffix_data,
            suffix_offsets: self.suffix_offsets,
            config: self.config,
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    fn maybe_seal(&mut self) {
        if self.pending.len() == self.config.segment_size {
            self.flush_pending(true);
        }
    }

    /// Flush all rows in `self.pending` into the flat storage.
    ///
    /// If `compute_prefix` is `true` the segment is sealed: a `prefix_len` is
    /// computed and each value is split into prefix slab + suffix.
    ///
    /// If `compute_prefix` is `false` (tail segment) `prefix_len = 0` and the
    /// entire value is written into the suffix region.
    fn flush_pending(&mut self, compute_prefix: bool) {
        if self.pending.is_empty() {
            return;
        }

        let seg_start = self.total_len - self.pending.len();
        let seg_len = self.pending.len();

        let prefix_len = if compute_prefix {
            compute_prefix_len(&self.pending, &self.config)
        } else {
            0
        };

        let mut seg_prefix_data = vec![0u8; seg_len * prefix_len];

        for (row, val) in self.pending.iter().enumerate() {
            match val {
                None => {
                    self.lengths.push(0);
                    let cur = *self.suffix_offsets.last().unwrap();
                    self.suffix_offsets.push(cur);
                }
                Some(bytes) => {
                    let total_len = bytes.len();
                    self.lengths.push(total_len as i32);

                    let actual_prefix = total_len.min(prefix_len);
                    if prefix_len > 0 {
                        let slab_start = row * prefix_len;
                        seg_prefix_data[slab_start..slab_start + actual_prefix]
                            .copy_from_slice(&bytes[..actual_prefix]);
                    }

                    let suffix = if total_len > prefix_len {
                        &bytes[prefix_len..]
                    } else {
                        &[]
                    };
                    let cur = *self.suffix_offsets.last().unwrap();
                    self.suffix_data.extend_from_slice(suffix);
                    self.suffix_offsets.push(cur + suffix.len() as i32);
                }
            }
        }

        self.segments.push(SegmentMeta {
            start: seg_start,
            len: seg_len,
            prefix_len,
            sealed: compute_prefix,
        });
        self.prefix_data.push(seg_prefix_data);
        self.pending.clear();
    }
}

impl Default for PrefixSplitBuilder {
    fn default() -> Self {
        Self::new()
    }
}
