/*
 * Copyright (c) 2026 seekdb competition entry.
 * Licensed under the Apache License, Version 2.0.
 *
 * hipVS HNSW bulk-build worker.
 *
 * A standalone executable that builds the base layer of an HNSW index on an
 * AMD GPU through hipVS (the ROCm port of the cuVS C API) and serializes a
 * VSAG-loadable HNSW snapshot to stdout:
 *
 *   1. read the vector matrix from stdin (own 64-byte framed protocol);
 *   2. build a CAGRA kNN graph on the GPU (NN-Descent or IVF-PQ based);
 *   3. lift component entries plus a deterministic sample of nodes into
 *      routing layers with a host-side HNSW (hnsw_upper_layers.h), which also
 *      splices cross-component links into the base layer;
 *   4. verify that every vector is reachable from the snapshot entry point;
 *   5. emit the snapshot in the VSAG hnsw on-disk layout.
 *
 * It runs as a separate process on purpose: hipVS and VSAG expose overlapping
 * C++ symbols, so both must never be linked into one binary. Only bytes cross
 * the process boundary.
 *
 * Build (AMD ROCm environment): see build_worker.sh
 * Usage:
 *   hipvs_hnsw_worker build    < vectors.fbin > snapshot.bin
 *   hipvs_hnsw_worker selftest < vectors.fbin   (build + recall self check)
 */
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "hnsw_upper_layers.h"
#include "snapshot_format.h"

namespace {

constexpr uint64_t PROTOCOL_MAGIC = 0x5748554154564842ULL;  // "BHVTUAHW" LE
constexpr uint32_t PROTOCOL_VERSION = 1;
constexpr uint32_t PROTOCOL_HEADER_SIZE = 64;
constexpr uint64_t MIN_ROWS = 256;
constexpr uint64_t MAX_ROWS = 1ULL << 32;
constexpr uint64_t MAX_DIM = 65536;

struct BuildRequest
{
  uint64_t n = 0;
  uint64_t dim = 0;
  uint64_t m = 0;
  uint64_t ef_construction = 0;
  uint64_t flags = 0;  // bit 0: run the recall self-test after building
};

std::vector<float> read_request(BuildRequest &req)
{
  uint64_t header[8] = {};
  if (fread(header, sizeof(header), 1, stdin) != 1) {
    throw std::runtime_error("cannot read the 64-byte request header");
  }
  if (header[0] != PROTOCOL_MAGIC) {
    throw std::runtime_error("bad protocol magic");
  }
  if (header[1] != PROTOCOL_VERSION || header[2] != PROTOCOL_HEADER_SIZE) {
    throw std::runtime_error("unsupported protocol version");
  }
  req.n = header[3];
  req.dim = header[4];
  req.m = header[5];
  req.ef_construction = header[6];
  req.flags = header[7];
  if (req.n < MIN_ROWS || req.n > MAX_ROWS || req.dim < 1 || req.dim > MAX_DIM ||
      req.m < 4 || req.m > obhipvs::MAX_BASE_DEGREE ||
      req.ef_construction < req.m || req.ef_construction > 1000000 ||
      req.n <= 4 * req.m) {
    throw std::runtime_error("unsupported build shape (need n >= 4M + 1, 4 <= M <= 128)");
  }
  if (req.n > (std::numeric_limits<size_t>::max)() / req.dim / sizeof(float)) {
    throw std::runtime_error("vector matrix exceeds the address space");
  }
  std::vector<float> vectors(static_cast<size_t>(req.n) * req.dim);
  if (fread(vectors.data(), sizeof(float), vectors.size(), stdin) != vectors.size()) {
    throw std::runtime_error("truncated vector payload");
  }
  for (float v : vectors) {
    if (!std::isfinite(v)) {
      throw std::runtime_error("non-finite value in the vector payload");
    }
  }
  return vectors;
}

struct GpuGraph
{
  std::vector<uint32_t> links;  // n x degree, row-major
  uint64_t degree = 0;
  double gpu_ms = 0.0;
};

// Builds the flat kNN base graph on the GPU through the hipVS (cuVS) C API.
GpuGraph build_gpu_graph(const std::vector<float> &vectors, const BuildRequest &req)
{
  int device_count = 0;
  if (hipGetDeviceCount(&device_count) != hipSuccess || device_count < 1) {
    throw std::runtime_error("no AMD GPU is visible to HIP");
  }

  cuvsResources_t resources = 0;
  if (cuvsResourcesCreate(&resources) != CUVS_SUCCESS) {
    throw std::runtime_error(std::string("cuvsResourcesCreate: ") + cuvsGetLastErrorText());
  }

  cuvsCagraIndexParams_t params = nullptr;
  cuvsCagraIndex_t index = nullptr;
  GpuGraph out;
  out.degree = 2 * req.m;

  try {
    if (cuvsCagraIndexParamsCreate(&params) != CUVS_SUCCESS) {
      throw std::runtime_error(std::string("cuvsCagraIndexParamsCreate: ") + cuvsGetLastErrorText());
    }
    params->graph_degree = static_cast<uint32_t>(out.degree);
    // NN-Descent has a fixed startup cost that dominates small snapshots;
    // IVF-PQ's refined candidate graph wins at larger cardinalities.
    const bool use_nn_descent = req.n < (1ULL << 18);
    params->build_algo = use_nn_descent ? NN_DESCENT : IVF_PQ;
    params->nn_descent_niter = 20;
    params->intermediate_graph_degree = static_cast<uint32_t>(
        out.degree * (use_nn_descent ? 2 : 4));
    if (!use_nn_descent) {
      // The C API owns graph_build_params after the build; hand it a heap copy.
      if (params->graph_build_params == nullptr) {
        params->graph_build_params = new cuvsIvfPqParams{};
      }
      params->graph_build_params->refinement_rate = 4.0f;
    }
    if (cuvsCagraIndexCreate(&index) != CUVS_SUCCESS) {
      throw std::runtime_error(std::string("cuvsCagraIndexCreate: ") + cuvsGetLastErrorText());
    }

    int64_t shape[2] = {static_cast<int64_t>(req.n), static_cast<int64_t>(req.dim)};
    DLManagedTensor dataset{};
    dataset.dl_tensor.data = const_cast<float *>(vectors.data());
    dataset.dl_tensor.device.device_type = kDLCPU;
    dataset.dl_tensor.device.device_id = 0;
    dataset.dl_tensor.ndim = 2;
    dataset.dl_tensor.dtype.code = kDLFloat;
    dataset.dl_tensor.dtype.bits = 32;
    dataset.dl_tensor.dtype.lanes = 1;
    dataset.dl_tensor.shape = shape;
    dataset.dl_tensor.strides = nullptr;
    dataset.dl_tensor.byte_offset = 0;

    const auto build_begin = std::chrono::steady_clock::now();
    if (cuvsCagraBuild(resources, params, &dataset, index) != CUVS_SUCCESS) {
      throw std::runtime_error(std::string("cuvsCagraBuild: ") + cuvsGetLastErrorText());
    }
    if (hipDeviceSynchronize() != hipSuccess) {
      throw std::runtime_error("GPU build did not synchronize cleanly");
    }
    const auto build_end = std::chrono::steady_clock::now();
    out.gpu_ms = std::chrono::duration<double, std::milli>(build_end - build_begin).count();

    DLManagedTensor graph_view{};
    if (cuvsCagraIndexGetGraph(index, &graph_view) != CUVS_SUCCESS) {
      throw std::runtime_error(std::string("cuvsCagraIndexGetGraph: ") + cuvsGetLastErrorText());
    }
    if (graph_view.dl_tensor.ndim != 2 ||
        graph_view.dl_tensor.shape[0] != static_cast<int64_t>(req.n) ||
        graph_view.dl_tensor.shape[1] != static_cast<int64_t>(out.degree) ||
        graph_view.dl_tensor.dtype.bits != 32) {
      throw std::runtime_error("GPU graph view has an unexpected shape");
    }
    out.links.resize(static_cast<size_t>(req.n) * out.degree);
    const hipError_t copy = hipMemcpy(out.links.data(), graph_view.dl_tensor.data,
        out.links.size() * sizeof(uint32_t), hipMemcpyDeviceToHost);
    if (graph_view.deleter != nullptr) {
      graph_view.deleter(&graph_view);
    }
    if (copy != hipSuccess) {
      throw std::runtime_error("hipMemcpy of the GPU graph failed");
    }
    for (uint32_t v : out.links) {
      if (v >= req.n) {
        throw std::runtime_error("GPU graph contains an out-of-range neighbor");
      }
    }
  } catch (...) {
    if (index != nullptr) {
      cuvsCagraIndexDestroy(index);
    }
    if (params != nullptr) {
      cuvsCagraIndexParamsDestroy(params);
    }
    cuvsResourcesDestroy(resources);
    throw;
  }
  if (index != nullptr) {
    cuvsCagraIndexDestroy(index);
  }
  if (params != nullptr) {
    cuvsCagraIndexParamsDestroy(params);
  }
  cuvsResourcesDestroy(resources);
  return out;
}

// One representative per weakly-connected component of the directed GPU graph.
// Treats edges as undirected; every component must stay searchable, so each
// one contributes its lowest-id member as a routing entry.
std::vector<uint32_t> component_entries(const GpuGraph &graph, uint64_t n)
{
  const uint64_t deg = graph.degree;
  std::vector<uint64_t> offset(n + 1, 0);
  for (uint32_t v : graph.links) {
    ++offset[static_cast<size_t>(v) + 1];
  }
  for (uint64_t i = 0; i < n; ++i) {
    offset[i + 1] += offset[i];
  }
  std::vector<uint64_t> cursor(offset);
  std::vector<uint32_t> reverse(graph.links.size());
  for (uint64_t i = 0; i < n; ++i) {
    for (uint64_t j = 0; j < deg; ++j) {
      reverse[cursor[graph.links[i * deg + j]]++] = static_cast<uint32_t>(i);
    }
  }
  std::vector<bool> visited(n, false);
  std::vector<uint32_t> entries;
  std::vector<uint32_t> stack;
  for (uint64_t start = 0; start < n; ++start) {
    if (visited[start]) {
      continue;
    }
    entries.push_back(static_cast<uint32_t>(start));
    visited[start] = true;
    stack.push_back(static_cast<uint32_t>(start));
    while (!stack.empty()) {
      const uint32_t node = stack.back();
      stack.pop_back();
      for (uint64_t j = 0; j < deg; ++j) {
        const uint32_t next = graph.links[node * deg + j];
        if (!visited[next]) {
          visited[next] = true;
          stack.push_back(next);
        }
      }
      for (uint64_t p = offset[node]; p < offset[node + 1]; ++p) {
        const uint32_t prev = reverse[p];
        if (!visited[prev]) {
          visited[prev] = true;
          stack.push_back(prev);
        }
      }
    }
  }
  return entries;
}

// Deterministic extra sample so routing layers have healthy density.
std::vector<uint32_t> select_entries(const GpuGraph &graph, uint64_t n, uint64_t m)
{
  std::vector<uint32_t> entries = component_entries(graph, n);
  std::unordered_set<uint32_t> chosen(entries.begin(), entries.end());
  const uint64_t target = std::max<uint64_t>(n / m, entries.size());
  const uint64_t stride = std::max<uint64_t>(n / (target + 1), 1);
  for (uint64_t i = 0; i < n && chosen.size() < target; i += stride) {
    chosen.insert(static_cast<uint32_t>(i));
  }
  entries.assign(chosen.begin(), chosen.end());
  std::sort(entries.begin(), entries.end());
  return entries;
}

struct SnapshotLinks
{
  std::vector<uint32_t> base;                    // n x 2M, row-major
  std::vector<uint32_t> row_level;               // real level per row (0 = base only)
  std::vector<std::vector<std::vector<uint32_t>>> upper;  // [real level - 1][row][ids]
  uint32_t max_level = 1;
  uint32_t enterpoint = 0;
};

SnapshotLinks assemble_layers(const GpuGraph &graph, const std::vector<float> &vectors,
    const BuildRequest &req, const std::vector<uint32_t> &entries)
{
  const uint64_t n = req.n;
  const uint64_t deg = graph.degree;
  SnapshotLinks out;
  out.base = graph.links;
  out.row_level.assign(n, 0);
  out.max_level = 1;

  obhipvs::UpperHnsw upper(vectors.data(), static_cast<uint32_t>(req.dim),
      static_cast<uint32_t>(req.m), static_cast<uint32_t>(req.ef_construction), 20260101ULL);
  for (uint32_t gid : entries) {
    const uint32_t level = upper.insert(gid);
    out.row_level[gid] = level;
    out.max_level = std::max(out.max_level, level);
  }
  out.enterpoint = upper.entry();

  // Upper adjacency indexed by real level (level - 1); row dimension is dense.
  out.upper.resize(out.max_level);
  for (auto &level_rows : out.upper) {
    level_rows.resize(n);
  }
  for (uint32_t gid : entries) {
    const uint32_t level = out.row_level[gid];
    for (uint32_t l = 1; l <= level; ++l) {
      std::vector<uint32_t> ids;
      upper.links_at(gid, l, ids);
      out.upper[l - 1][gid] = ids;
    }
  }

  // Splice the aux level-0 links of entry rows into the base layer so that a
  // level-0 traversal can cross component boundaries through the entries.
  std::vector<uint32_t> row;
  for (uint32_t gid : entries) {
    row.clear();
    std::vector<uint32_t> aux;
    upper.links_at(gid, 1, aux);
    row.insert(row.end(), aux.begin(), aux.end());
    const uint32_t *gpu_row = graph.links.data() + static_cast<size_t>(gid) * deg;
    for (uint64_t j = 0; j < deg && row.size() < deg; ++j) {
      if (gpu_row[j] != gid && std::find(row.begin(), row.end(), gpu_row[j]) == row.end()) {
        row.push_back(gpu_row[j]);
      }
    }
    std::copy(row.begin(), row.end(), out.base.begin() + static_cast<size_t>(gid) * deg);
  }
  return out;
}

// Every vector must be reachable from the enterpoint through the base layer
// (the upper layers route into every component by construction).
void verify_reachability(const SnapshotLinks &links, uint64_t n, uint64_t deg)
{
  std::vector<bool> seen(n, false);
  std::vector<uint32_t> stack{links.enterpoint};
  seen[links.enterpoint] = true;
  uint64_t reached = 0;
  while (!stack.empty()) {
    const uint32_t node = stack.back();
    stack.pop_back();
    ++reached;
    for (uint64_t j = 0; j < deg; ++j) {
      const uint32_t next = links.base[node * deg + j];
      if (next < n && !seen[next]) {
        seen[next] = true;
        stack.push_back(next);
      }
    }
  }
  if (reached != n) {
    throw std::runtime_error("routing check failed: base layer does not reach every vector");
  }
}

std::string serialize_snapshot(const std::vector<float> &vectors, const BuildRequest &req,
    const SnapshotLinks &links)
{
  const uint64_t n = req.n;
  const uint64_t dim = req.dim;
  const uint64_t m = req.m;
  const uint64_t deg = 2 * m;
  std::string out;
  out.reserve(96 + static_cast<size_t>(n) * obhipvs::compute_row_stride(dim, m));

  obhipvs::write_pod(out, static_cast<uint64_t>(0));                       // offset_level0
  obhipvs::write_pod(out, n);                                              // max_elements
  obhipvs::write_pod(out, n);                                              // cur_element_count
  obhipvs::write_pod(out, obhipvs::compute_row_stride(dim, m));            // row stride
  obhipvs::write_pod(out, (1 + deg) * 4 + dim * 4);                        // label_offset
  obhipvs::write_pod(out, (1 + deg) * 4);                                  // offset_data
  obhipvs::write_pod(out, static_cast<int32_t>(links.max_level));          // max_level
  obhipvs::write_pod(out, links.enterpoint);                               // enterpoint
  obhipvs::write_pod(out, m);                                              // max_m
  obhipvs::write_pod(out, deg);                                            // max_m0
  obhipvs::write_pod(out, m);                                              // M
  obhipvs::write_pod(out, 1.0 / std::log(static_cast<double>(m)));         // mult
  obhipvs::write_pod(out, req.ef_construction);                            // ef_construction

  std::vector<uint32_t> padded(deg, 0);
  for (uint64_t i = 0; i < n; ++i) {
    obhipvs::write_pod(out, static_cast<uint32_t>(deg));
    const uint32_t *row = links.base.data() + i * deg;
    obhipvs::write_bytes(out, row, deg * sizeof(uint32_t));
    obhipvs::write_bytes(out, vectors.data() + i * dim, dim * sizeof(float));
    obhipvs::write_pod(out, static_cast<uint64_t>(i));                     // label == row id
  }
  for (uint64_t i = 0; i < n; ++i) {
    const uint32_t level = links.row_level[i];
    if (level <= 1) {
      obhipvs::write_pod(out, static_cast<uint32_t>(0));
      continue;
    }
    const uint32_t record = level * (4 + 4 * m);
    obhipvs::write_pod(out, record);
    for (uint32_t l = 1; l <= level; ++l) {
      const auto &ids = links.upper[l - 1][i];
      obhipvs::write_pod(out, static_cast<uint32_t>(ids.size()));
      padded.assign(m, 0);
      for (size_t k = 0; k < ids.size() && k < m; ++k) {
        padded[k] = ids[k];
      }
      obhipvs::write_bytes(out, padded.data(), m * sizeof(uint32_t));
    }
  }
  return out;
}

// ---- self-test: layered search + recall against brute force ----------------

float l2(const std::vector<float> &v, uint64_t dim, uint64_t a, const float *q)
{
  const float *pa = v.data() + a * dim;
  float sum = 0.0f;
  for (uint64_t d = 0; d < dim; ++d) {
    const float diff = pa[d] - q[d];
    sum += diff * diff;
  }
  return sum;
}

double run_selftest(const std::vector<float> &vectors, const BuildRequest &req,
    const SnapshotLinks &links)
{
  const uint64_t n = req.n;
  const uint64_t dim = req.dim;
  const uint64_t deg = 2 * req.m;
  std::mt19937 rng(777);
  std::uniform_int_distribution<uint64_t> pick(0, n - 1);
  const uint32_t queries = 100;
  const uint32_t topk = 10;
  uint64_t hits = 0;

  std::vector<float> query(dim);
  for (uint32_t q = 0; q < queries; ++q) {
    const uint64_t target = pick(rng);
    std::copy_n(vectors.data() + target * dim, dim, query.begin());
    // brute-force ground truth (exact top-10, excluding the self row)
    std::vector<std::pair<float, uint64_t>> truth;
    truth.reserve(n);
    for (uint64_t i = 0; i < n; ++i) {
      truth.emplace_back(l2(vectors, dim, i, query.data()), i);
    }
    std::partial_sort(truth.begin(), truth.begin() + topk + 1, truth.end());
    std::unordered_set<uint64_t> expected;
    for (uint32_t k = 0; k < topk + 1; ++k) {
      if (truth[k].second != target) {
        expected.insert(truth[k].second);
      }
      if (expected.size() == topk) {
        break;
      }
    }
    // layered HNSW search
    uint32_t cur = links.enterpoint;
    float cur_dist = l2(vectors, dim, cur, query.data());
    for (int level = links.max_level; level >= 1; --level) {
      bool changed = true;
      while (changed) {
        changed = false;
        if (level - 1 < static_cast<int>(links.upper.size())) {
          for (uint32_t nb : links.upper[level - 1][cur]) {
            const float d = l2(vectors, dim, nb, query.data());
            if (d < cur_dist) {
              cur_dist = d;
              cur = nb;
              changed = true;
            }
          }
        }
      }
    }
    // beam search at the base layer (ef = max(efc, 64))
    const uint32_t ef = std::max<uint32_t>(
        static_cast<uint32_t>(req.ef_construction), 64);
    std::unordered_set<uint32_t> visited{cur};
    auto cmp = [](const std::pair<float, uint32_t> &a, const std::pair<float, uint32_t> &b) {
      return a.first > b.first;
    };
    std::priority_queue<std::pair<float, uint32_t>, std::vector<std::pair<float, uint32_t>>, decltype(cmp)> cand(cmp);
    std::priority_queue<std::pair<float, uint32_t>> best;
    cand.push({cur_dist, cur});
    best.push({cur_dist, cur});
    if (best.size() > ef) {
      best.pop();
    }
    while (!cand.empty()) {
      const auto c = cand.top();
      cand.pop();
      if (!best.empty() && c.first > best.top().first && best.size() >= ef) {
        break;
      }
      for (uint64_t j = 0; j < deg; ++j) {
        const uint32_t nb = links.base[c.second * deg + j];
        if (nb < n && visited.insert(nb).second) {
          const float d = l2(vectors, dim, nb, query.data());
          if (best.size() < ef || d < best.top().first) {
            best.push({d, nb});
            if (best.size() > ef) {
              best.pop();
            }
            cand.push({d, nb});
          }
        }
      }
    }
    uint32_t found = 0;
    std::vector<uint64_t> ids;
    while (!best.empty()) {
      ids.push_back(best.top().second);
      best.pop();
    }
    for (uint64_t id : ids) {
      if (expected.count(id) != 0) {
        ++found;
      }
    }
    hits += found;
  }
  return static_cast<double>(hits) / (static_cast<double>(queries) * topk);
}

}  // namespace

int main(int argc, char **argv)
{
  const bool selftest = argc > 1 && std::strcmp(argv[1], "selftest") == 0;
  if (argc > 1 && std::strcmp(argv[1], "build") != 0 && !selftest) {
    std::fprintf(stderr, "usage: %s [build|selftest] < vectors.fbin > snapshot.bin\n", argv[0]);
    return 2;
  }
  const auto total_begin = std::chrono::steady_clock::now();
  try {
    BuildRequest req;
    const std::vector<float> vectors = read_request(req);
    std::fprintf(stderr, "[HIPVS] input n=%llu dim=%llu m=%llu efc=%llu\n",
        static_cast<unsigned long long>(req.n), static_cast<unsigned long long>(req.dim),
        static_cast<unsigned long long>(req.m), static_cast<unsigned long long>(req.ef_construction));

    GpuGraph graph = build_gpu_graph(vectors, req);
    std::fprintf(stderr, "[HIPVS] gpu graph built in %.1f ms (degree=%llu, algo=%s)\n",
        graph.gpu_ms, static_cast<unsigned long long>(graph.degree),
        req.n < (1ULL << 18) ? "nn_descent" : "ivf_pq");

    const std::vector<uint32_t> entries = select_entries(graph, req.n, req.m);
    std::fprintf(stderr, "[HIPVS] routing entries: %zu\n", entries.size());

    SnapshotLinks links = assemble_layers(graph, vectors, req, entries);
    verify_reachability(links, req.n, graph.degree);
    std::fprintf(stderr, "[HIPVS] routing layers assembled, max_level=%u, enterpoint=%u\n",
        links.max_level, links.enterpoint);

    const std::string snapshot = serialize_snapshot(vectors, req, links);
    const double recall = selftest ? run_selftest(vectors, req, links) : -1.0;

    if (std::fwrite(snapshot.data(), 1, snapshot.size(), stdout) != snapshot.size() ||
        std::fflush(stdout) != 0) {
      throw std::runtime_error("cannot write the snapshot to stdout");
    }
    const double total_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - total_begin).count();
    std::fprintf(stderr,
        "[HIPVS_RESULT] rows=%llu dim=%llu m=%llu gpu_ms=%.1f total_ms=%.1f snapshot_bytes=%zu recall@10=%.4f\n",
        static_cast<unsigned long long>(req.n), static_cast<unsigned long long>(req.dim),
        static_cast<unsigned long long>(req.m), graph.gpu_ms, total_ms, snapshot.size(), recall);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[HIPVS_FAILED] %s\n", e.what());
    return 1;
  }
}
