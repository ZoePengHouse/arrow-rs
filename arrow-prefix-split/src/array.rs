// Licensed under the Apache License, Version 2.0.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, StringArray};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType;

use crate::config::PrefixSplitConfig;
use crate::segment::SegmentMeta;

static UTF8_DATA_TYPE: DataType = DataType::Utf8;

/// A string array that stores each value split into a **fixed-width prefix
/// slab** and a variable-length **suffix region**, organised into segments.
///
/// See the [crate-level documentation](crate) for design rationale and layout
/// details.
///
/// # Constructing
///
/// Use [`PrefixSplitBuilder`](crate::builder::PrefixSplitBuilder) to build
/// one value at a time, or the `From<&StringArray>` impl for bulk conversion.
///
/// # Accessing values
///
/// Because the original string is split across two separate byte buffers,
/// element access requires reconstruction and therefore **returns an owned
/// `String`** rather than a borrowed `&str`.  Use the suffix/prefix byte
/// accessors if you need zero-copy partial views.
pub struct PrefixSplitArray {
    /// Total number of logical elements (including nulls).
    pub(crate) len: usize,

    /// Physical null bitmap; `None` means all elements are valid.
    pub(crate) nulls: Option<NullBuffer>,

    /// One [`SegmentMeta`] per segment, in row order.
    pub(crate) segments: Vec<SegmentMeta>,

    /// Per-segment prefix slabs.
    ///
    /// `prefix_data[s]` holds `seg.len * seg.prefix_len` bytes for segment `s`.
    /// The prefix of the *k*-th row within segment `s` is at byte range
    /// `k * prefix_len .. (k+1) * prefix_len`.
    /// Empty for unsealed segments (prefix_len == 0).
    pub(crate) prefix_data: Vec<Vec<u8>>,

    /// Original byte length of every element.
    /// For null elements the length is stored as 0.
    pub(crate) lengths: Vec<i32>,

    /// Concatenated suffix bytes across all elements.
    pub(crate) suffix_data: Vec<u8>,

    /// Suffix offset table: element `i`'s suffix is
    /// `suffix_data[suffix_offsets[i] .. suffix_offsets[i+1]]`.
    /// Length = `len + 1`; `suffix_offsets[0] == 0`.
    pub(crate) suffix_offsets: Vec<i32>,

    /// Configuration used to build this array (kept for slicing / re-building).
    pub(crate) config: PrefixSplitConfig,
}

// ── Core accessors ────────────────────────────────────────────────────────────

impl PrefixSplitArray {
    /// Number of logical elements (including nulls).
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if the array has no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// `true` if element `i` is null.
    #[inline]
    pub fn is_null(&self, i: usize) -> bool {
        assert!(i < self.len, "index {i} out of bounds (len={})", self.len);
        self.nulls.as_ref().map_or(false, |n| n.is_null(i))
    }

    /// `true` if element `i` is non-null.
    #[inline]
    pub fn is_valid(&self, i: usize) -> bool {
        !self.is_null(i)
    }

    /// Number of null elements.
    #[inline]
    pub fn null_count(&self) -> usize {
        self.nulls.as_ref().map_or(0, |n| n.null_count())
    }

    /// The optional null buffer.
    #[inline]
    pub fn nulls(&self) -> Option<&NullBuffer> {
        self.nulls.as_ref()
    }

    /// Original byte length of element `i`.  Returns 0 for null elements.
    #[inline]
    pub fn byte_length(&self, i: usize) -> usize {
        assert!(i < self.len, "index {i} out of bounds (len={})", self.len);
        self.lengths[i] as usize
    }

    /// Reconstructs and returns the full string value for element `i`.
    ///
    /// # Panics
    /// Panics if `i` is out of bounds **or** if element `i` is null.
    pub fn value(&self, i: usize) -> String {
        assert!(i < self.len, "index {i} out of bounds (len={})", self.len);
        assert!(!self.is_null(i), "element {i} is null");

        let seg_idx = self.segment_index_for(i);
        let seg = &self.segments[seg_idx];
        let total_len = self.lengths[i] as usize;

        let mut result = Vec::with_capacity(total_len);

        if seg.prefix_len > 0 {
            let row_in_seg = i - seg.start;
            let p = seg.prefix_len;
            let actual_prefix_len = total_len.min(p);
            let slab_start = row_in_seg * p;
            result.extend_from_slice(
                &self.prefix_data[seg_idx][slab_start..slab_start + actual_prefix_len],
            );
        }

        let s = self.suffix_offsets[i] as usize;
        let e = self.suffix_offsets[i + 1] as usize;
        result.extend_from_slice(&self.suffix_data[s..e]);

        // SAFETY: we only ever store valid UTF-8 (input is always `&str`).
        unsafe { String::from_utf8_unchecked(result) }
    }

    /// Returns the raw prefix slab bytes for element `i`, or `None` if the
    /// segment containing `i` is not sealed.
    ///
    /// The returned slice is always exactly `prefix_len` bytes long, including
    /// any zero-padding for strings shorter than `prefix_len`.
    pub fn prefix_bytes(&self, i: usize) -> Option<&[u8]> {
        assert!(i < self.len, "index {i} out of bounds");
        let seg_idx = self.segment_index_for(i);
        let seg = &self.segments[seg_idx];
        if !seg.sealed {
            return None;
        }
        let row_in_seg = i - seg.start;
        let p = seg.prefix_len;
        Some(&self.prefix_data[seg_idx][row_in_seg * p..(row_in_seg + 1) * p])
    }

    /// Returns the raw suffix bytes for element `i`.
    ///
    /// For sealed segments this is `value[prefix_len..]`; for unsealed
    /// segments this is the full string.
    pub fn suffix_bytes(&self, i: usize) -> &[u8] {
        assert!(i < self.len, "index {i} out of bounds");
        let s = self.suffix_offsets[i] as usize;
        let e = self.suffix_offsets[i + 1] as usize;
        &self.suffix_data[s..e]
    }

    /// Returns the segment metadata slice.
    pub fn segments(&self) -> &[SegmentMeta] {
        &self.segments
    }

    /// Returns the prefix slab for segment `seg_idx`.
    ///
    /// Useful for bulk SIMD-style prefix scanning in search routines.
    #[inline]
    pub(crate) fn prefix_slab(&self, seg_idx: usize) -> &[u8] {
        &self.prefix_data[seg_idx]
    }

    // ── Derived helpers ───────────────────────────────────────────────────

    /// Binary-search the segment that contains logical row `i`.
    #[inline]
    pub(crate) fn segment_index_for(&self, i: usize) -> usize {
        debug_assert!(!self.segments.is_empty());
        let mut lo = 0usize;
        let mut hi = self.segments.len() - 1;
        while lo < hi {
            let mid = lo + (hi - lo + 1) / 2;
            if self.segments[mid].start <= i {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    }
}

// ── Slicing ───────────────────────────────────────────────────────────────────

impl PrefixSplitArray {
    /// Returns a new `PrefixSplitArray` covering rows `[offset, offset+length)`.
    ///
    /// This is a **copy-slice**: the returned array is independent and its
    /// segments are rebuilt from scratch using the same configuration.
    pub fn slice(&self, offset: usize, length: usize) -> PrefixSplitArray {
        assert!(
            offset + length <= self.len,
            "slice [{offset}, {offset}+{length}) out of bounds (len={})",
            self.len
        );

        if length == 0 {
            return PrefixSplitArray {
                len: 0,
                nulls: None,
                segments: vec![],
                prefix_data: vec![],
                lengths: vec![],
                suffix_data: vec![],
                suffix_offsets: vec![0],
                config: self.config.clone(),
            };
        }

        let mut builder = crate::builder::PrefixSplitBuilder::with_config(self.config.clone());
        for i in offset..offset + length {
            if self.is_null(i) {
                builder.append_null();
            } else {
                builder.append_value(&self.value(i));
            }
        }
        builder.finish()
    }

    fn do_slice(&self, offset: usize, length: usize) -> Self {
        self.slice(offset, length)
    }
}

// ── Arrow interoperability ────────────────────────────────────────────────────

impl PrefixSplitArray {
    /// Converts this array to a standard Arrow [`StringArray`].
    ///
    /// This is an `O(n)` operation that reconstructs every value.
    pub fn to_string_array(&self) -> StringArray {
        let strings: Vec<Option<String>> = (0..self.len)
            .map(|i| {
                if self.is_null(i) {
                    None
                } else {
                    Some(self.value(i))
                }
            })
            .collect();
        StringArray::from(strings)
    }
}

impl From<&StringArray> for PrefixSplitArray {
    /// Converts a standard Arrow [`StringArray`] using default configuration.
    fn from(arr: &StringArray) -> Self {
        let mut builder = crate::builder::PrefixSplitBuilder::new();
        for i in 0..arr.len() {
            if arr.is_null(i) {
                builder.append_null();
            } else {
                builder.append_value(arr.value(i));
            }
        }
        builder.finish()
    }
}

// ── Arrow `Array` trait ───────────────────────────────────────────────────────

// SAFETY: We uphold the Arrow Array contract by delegating to_data /
// into_data to a proper StringArray (standard Utf8 layout).  All methods
// that operate on logical indices do bounds-checking.
unsafe impl Array for PrefixSplitArray {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn to_data(&self) -> arrow_data::ArrayData {
        self.to_string_array().to_data()
    }

    fn into_data(self) -> arrow_data::ArrayData {
        self.to_string_array().into_data()
    }

    fn data_type(&self) -> &DataType {
        &UTF8_DATA_TYPE
    }

    fn slice(&self, offset: usize, length: usize) -> ArrayRef {
        Arc::new(self.do_slice(offset, length))
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn offset(&self) -> usize {
        0
    }

    fn nulls(&self) -> Option<&NullBuffer> {
        self.nulls.as_ref()
    }

    fn get_buffer_memory_size(&self) -> usize {
        let prefix: usize = self.prefix_data.iter().map(|b| b.len()).sum();
        let suffix = self.suffix_data.len();
        let offsets = (self.len + 1) * size_of::<i32>();
        let lengths = self.len * size_of::<i32>();
        let nulls = self.nulls.as_ref().map_or(0, |n| n.buffer().len());
        prefix + suffix + offsets + lengths + nulls
    }

    fn get_array_memory_size(&self) -> usize {
        self.get_buffer_memory_size() + size_of::<Self>()
    }
}

// ── Iterators ─────────────────────────────────────────────────────────────────

/// Iterator over a [`PrefixSplitArray`] that yields `Option<String>`.
pub struct PrefixSplitIter<'a> {
    array: &'a PrefixSplitArray,
    pos: usize,
}

impl<'a> PrefixSplitIter<'a> {
    pub(crate) fn new(array: &'a PrefixSplitArray) -> Self {
        Self { array, pos: 0 }
    }
}

impl Iterator for PrefixSplitIter<'_> {
    type Item = Option<String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.array.len() {
            return None;
        }
        let i = self.pos;
        self.pos += 1;
        if self.array.is_null(i) {
            Some(None)
        } else {
            Some(Some(self.array.value(i)))
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.array.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for PrefixSplitIter<'_> {}

impl PrefixSplitArray {
    /// Returns an iterator that yields `Option<String>` for each element.
    pub fn iter(&self) -> PrefixSplitIter<'_> {
        PrefixSplitIter::new(self)
    }
}

// ── Display / Debug ───────────────────────────────────────────────────────────

impl fmt::Debug for PrefixSplitArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PrefixSplitArray(len={}, null_count={}, segments=[",
            self.len,
            self.null_count()
        )?;
        for (idx, seg) in self.segments.iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            if seg.sealed {
                write!(
                    f,
                    "Seg{{start={}, len={}, prefix_len={}}}",
                    seg.start, seg.len, seg.prefix_len
                )?;
            } else {
                write!(
                    f,
                    "Seg{{start={}, len={}, unsealed}}",
                    seg.start, seg.len
                )?;
            }
        }
        write!(f, "])")
    }
}

impl fmt::Display for PrefixSplitArray {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..self.len {
            if i > 0 {
                write!(f, ", ")?;
            }
            if self.is_null(i) {
                write!(f, "null")?;
            } else {
                write!(f, "{:?}", self.value(i))?;
            }
        }
        write!(f, "]")
    }
}
