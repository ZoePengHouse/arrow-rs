// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! # `arrow-prefix-split`
//!
//! A prefix-split string array layout for Apache Arrow that is optimized for
//! **exact equality search** (`WHERE col = 'value'`).
//!
//! ## Motivation
//!
//! Standard Arrow `StringArray` stores all string bytes contiguously. When
//! scanning for equality, each cache miss brings in bytes that may belong to
//! only one (long) string. This module exploits the observation that, in many
//! real-world string columns, the first few bytes already distinguish most
//! values: by storing all prefixes together in a dense fixed-width region,
//! a single cache line carries distinguishing bits for many more candidates,
//! reducing the number of full-string reads needed during a scan.
//!
//! ## Layout
//!
//! The array is divided into **segments** (default 1 000 rows each).
//!
//! Once a segment reaches `segment_size` rows it is **sealed**: a
//! `prefix_len_bytes` value is computed so that the prefixes can distinguish
//! ≥ 90 % of the distinct non-null strings in that segment (configurable
//! threshold).  The tail segment (< `segment_size` rows) is left *unsealed*
//! and falls back to full-string comparison.
//!
//! For every element the value is split into:
//! * **prefix** – first `min(len, prefix_len)` bytes, stored in a
//!   fixed-width slab (`prefix_len` bytes per row, zero-padded for short
//!   strings).
//! * **suffix** – remaining bytes `value[prefix_len..]`, stored with
//!   Arrow-style offset/data buffers.
//! * **length** – original byte length stored separately so short strings
//!   (< `prefix_len`) can be reconstructed exactly.
//!
//! ## Usage
//!
//! ```rust
//! use arrow_prefix_split::{PrefixSplitArray, PrefixSplitBuilder, PrefixSplitConfig, search_eq};
//!
//! let mut builder = PrefixSplitBuilder::new();
//! builder.append_value("apple");
//! builder.append_value("application");
//! builder.append_value("banana");
//! builder.append_null();
//! let array = builder.finish();
//!
//! assert_eq!(array.value(0), "apple");
//! assert!(array.is_null(3));
//!
//! let hits = search_eq(&array, "banana");
//! assert_eq!(hits, vec![2]);
//! ```

pub mod array;
pub mod builder;
pub mod config;
pub mod search;
pub mod segment;

pub use array::{PrefixSplitArray, PrefixSplitIter};
pub use builder::PrefixSplitBuilder;
pub use config::PrefixSplitConfig;
pub use search::{search_eq, search_in, search_ne, search_range};
pub use segment::SegmentMeta;
