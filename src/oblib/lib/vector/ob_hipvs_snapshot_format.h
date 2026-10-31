// SERVER-SIDE copy of the hipVS HNSW snapshot wire format.
// MUST stay byte-for-byte compatible with tools/hipvs_hnsw/snapshot_format.h
// (the GPU worker's copy); both implement the same 96-byte layout.
/*
 * Copyright (c) 2026 seekdb competition entry.
 * Licensed under the Apache License, Version 2.0.
 *
 * Binary snapshot layout shared by the hipVS GPU worker (writer) and the
 * in-server offload loader (validator).
 *
 * The layout follows the on-disk format of VSAG's HNSW index (a hnswlib
 * derivative) so that a GPU-built snapshot can be loaded directly through
 * Index::Deserialize() without any conversion step inside the server:
 *
 *   header (96 bytes)
 *     [ 0] u64  offset_level0, always 0
 *     [ 8] u64  max_elements (capacity, == count for a full snapshot)
 *     [16] u64  cur_element_count
 *     [24] u64  size_data_per_element (row stride in bytes)
 *     [32] u64  label_offset (offset of the u64 label inside a row)
 *     [40] u64  offset_data (offset of the vector inside a row)
 *     [48] i32  max_level
 *     [52] u32  enterpoint node id
 *     [56] u64  max_m (upper-layer degree cap, == M)
 *     [64] u64  max_m0 (base-layer degree cap, == 2M)
 *     [72] u64  M
 *     [80] f64  level multiplier
 *     [88] u64  ef_construction
 *
 *   base rows, count rows at 96 + i * stride
 *     [..] u32  link count (<= 2M)
 *     [..] 2M x u32 neighbor slots (ids are row ids, unused tail slots are 0)
 *     [..] dim x f32 vector data
 *     [..] u64  label
 *
 *   upper-layer section, count records in row order
 *     [..] u32  byte length of the record (0 when the row has no upper level)
 *     for every level of that row:
 *       [..] u32 degree (<= M)
 *       [..] M x u32 neighbor slots (row ids, unused tail slots are 0)
 */
#ifndef OB_HIPVS_SNAPSHOT_FORMAT_H
#define OB_HIPVS_SNAPSHOT_FORMAT_H

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace obhipvs {

constexpr uint64_t SNAPSHOT_HEADER_SIZE = 96;
constexpr uint64_t MAX_BASE_DEGREE = 128;   // hard cap for M, base cap is 2M

struct SnapshotHeader
{
  uint64_t offset_level0;
  uint64_t max_elements;
  uint64_t count;
  uint64_t row_stride;
  uint64_t label_offset;
  uint64_t data_offset;
  int32_t max_level;
  uint32_t enterpoint;
  uint64_t max_m;
  uint64_t max_m0;
  uint64_t m;
  double level_mult;
  uint64_t ef_construction;
};

inline uint64_t compute_row_stride(uint64_t dim, uint64_t m)
{
  return (1 + 2 * m) * 4 + dim * 4 + 8;
}

template <typename T>
inline void write_pod(std::string &out, const T &value)
{
  out.append(reinterpret_cast<const char *>(&value), sizeof(T));
}

inline void write_bytes(std::string &out, const void *data, size_t size)
{
  out.append(static_cast<const char *>(data), size);
}

// Reads a little-endian POD at an absolute offset; throws when truncated.
template <typename T>
inline T read_pod(const char *bytes, size_t size, size_t offset)
{
  if (offset + sizeof(T) > size) {
    throw std::runtime_error("truncated snapshot read");
  }
  T value;
  std::memcpy(&value, bytes + offset, sizeof(T));
  return value;
}

// Structural validation of the 96-byte header. Returns the expected row count.
// Only checks invariants, never trusts derived values for memory access.
inline uint64_t validate_header(const char *bytes, size_t size,
    uint64_t expect_count, uint64_t expect_dim, uint64_t expect_m)
{
  if (size < SNAPSHOT_HEADER_SIZE) {
    throw std::runtime_error("snapshot shorter than its header");
  }
  SnapshotHeader h{};
  h.offset_level0   = read_pod<uint64_t>(bytes, size, 0);
  h.max_elements    = read_pod<uint64_t>(bytes, size, 8);
  h.count           = read_pod<uint64_t>(bytes, size, 16);
  h.row_stride      = read_pod<uint64_t>(bytes, size, 24);
  h.label_offset    = read_pod<uint64_t>(bytes, size, 32);
  h.data_offset     = read_pod<uint64_t>(bytes, size, 40);
  h.max_level       = read_pod<int32_t>(bytes, size, 48);
  h.enterpoint      = read_pod<uint32_t>(bytes, size, 52);
  h.max_m           = read_pod<uint64_t>(bytes, size, 56);
  h.max_m0          = read_pod<uint64_t>(bytes, size, 64);
  h.m               = read_pod<uint64_t>(bytes, size, 72);
  h.level_mult      = read_pod<double>(bytes, size, 80);
  h.ef_construction = read_pod<uint64_t>(bytes, size, 88);

  const uint64_t stride = compute_row_stride(expect_dim, expect_m);
  if (h.offset_level0 != 0 || h.count != expect_count ||
      h.max_m != expect_m || h.max_m0 != 2 * expect_m || h.m != expect_m ||
      h.data_offset != (1 + expect_m * 2) * 4 ||
      h.label_offset != h.data_offset + expect_dim * 4 ||
      h.row_stride != stride ||
      h.max_level < 0 || h.enterpoint >= expect_count) {
    throw std::runtime_error("snapshot header does not match the requested shape");
  }
  // overflow guard: last row must end inside the buffer
  if (expect_count != 0 &&
      (SNAPSHOT_HEADER_SIZE > size ||
       (size - SNAPSHOT_HEADER_SIZE) / h.row_stride < expect_count)) {
    throw std::runtime_error("snapshot base rows exceed the buffer");
  }
  return h.count;
}

}  // namespace obhipvs

#endif  // OB_HIPVS_SNAPSHOT_FORMAT_H
