/*
 * Copyright (c) 2026 seekdb competition entry.
 * Licensed under the Apache License, Version 2.0.
 *
 * A compact, dependency-free HNSW used for the routing layers of a GPU-built
 * snapshot. The GPU (hipVS/CAGRA) produces the flat base-layer kNN graph;
 * this class lifts a small subset of entry nodes into a layered hierarchy so
 * the resulting snapshot can be traversed the usual HNSW way:
 *
 *   - aux level L of this class becomes real level L + 1 of the snapshot;
 *   - the aux level-0 links of entry nodes are spliced into the snapshot's
 *     base layer, which makes the base layer span every connected component
 *     of the GPU graph (a CAGRA graph may be disconnected).
 *
 * Distances are computed on the host: the entry subset is small (about n/M
 * nodes), so this step is negligible next to the GPU graph construction.
 */
#ifndef OB_HIPVS_HNSW_UPPER_LAYERS_H
#define OB_HIPVS_HNSW_UPPER_LAYERS_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace obhipvs {

class UpperHnsw
{
public:
  // vectors: row-major float table, dim elements per row; row ids are the
  // global (snapshot) row ids passed to insert().
  UpperHnsw(const float *vectors, uint32_t dim, uint32_t m,
      uint32_t ef_construction, uint64_t seed)
      : vectors_(vectors), dim_(dim), m_(m ? m : 8),
        efc_(ef_construction ? ef_construction : 200),
        rng_(seed)
  {
    if (vectors == nullptr || dim_ == 0) {
      throw std::invalid_argument("UpperHnsw: invalid vector table");
    }
    mult_ = 1.0 / std::log(static_cast<double>(m_));
  }

  uint32_t max_level() const { return cur_max_ + 1; }  // real level semantics
  uint32_t entry() const { return gids_.at(entry_); }

  // Real level (>= 1) of a global id; 0 when the node has no upper level.
  uint32_t real_level(uint32_t gid) const
  {
    auto it = gid2local_.find(gid);
    if (it == gid2local_.end()) {
      throw std::runtime_error("UpperHnsw: unknown global id");
    }
    return static_cast<uint32_t>(levels_[it->second]) + 1;
  }

  // Neighbor global ids of gid at the given real level (>= 1), at most m.
  void links_at(uint32_t gid, uint32_t real_level, std::vector<uint32_t> &out) const
  {
    out.clear();
    auto it = gid2local_.find(gid);
    if (it == gid2local_.end() || real_level < 1) {
      throw std::runtime_error("UpperHnsw: bad links_at request");
    }
    const uint32_t local = it->second;
    // real level L maps to aux level L - 1; the node exists up to
    // aux level levels_[local], i.e. real level levels_[local] + 1.
    if (static_cast<int>(real_level) > levels_[local] + 1) {
      return;  // node does not exist at that level
    }
    const auto &adj = adj_[real_level - 1][local];
    out.reserve(adj.size());
    for (uint32_t nb : adj) {
      out.push_back(gids_[nb]);
    }
  }

  // Inserts one point; returns its real level (aux level + 1).
  uint32_t insert(uint32_t gid)
  {
    if (gid2local_.count(gid) != 0) {
      throw std::runtime_error("UpperHnsw: duplicate insert");
    }
    const uint32_t local = static_cast<uint32_t>(gids_.size());
    gid2local_.emplace(gid, local);
    gids_.push_back(gid);

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const int level = static_cast<int>(
        -std::log(std::max(uniform(rng_), std::numeric_limits<double>::min())) * mult_);
    levels_.push_back(level);
    if (adj_.size() < static_cast<size_t>(level) + 1) {
      adj_.resize(static_cast<size_t>(level) + 1);
    }
    for (int l = 0; l <= level; ++l) {
      adj_[l].emplace_back();
    }

    if (entry_ == UINT32_MAX) {  // first node
      entry_ = local;
      cur_max_ = level;
      return static_cast<uint32_t>(level + 1);
    }

    uint32_t cur = entry_;
    float cur_dist = distance(local, cur);
    // Greedy descent through the levels above the insertion level.
    for (int l = cur_max_; l > level; --l) {
      bool changed = true;
      while (changed) {
        changed = false;
        for (uint32_t nb : adj_[l][cur]) {
          const float d = distance(local, nb);
          if (d < cur_dist) {
            cur_dist = d;
            cur = nb;
            changed = true;
          }
        }
      }
    }
    // Search + link on every level from min(level, cur_max) down to 0.
    for (int l = std::min(level, cur_max_); l >= 0; --l) {
      std::vector<Cand> result;
      search_layer(local, cur, l, efc_, result);
      select_neighbors(result, m_);
      for (const Cand &c : result) {
        adj_[l][local].push_back(c.id);
        adj_[l][c.id].push_back(local);
        // Keep the degree cap on the other endpoint too.
        if (adj_[l][c.id].size() > m_) {
          shrink(l, c.id);
        }
      }
      if (!result.empty()) {
        cur = result.front().id;
      }
    }
    if (level > cur_max_) {
      cur_max_ = level;
      entry_ = local;
    }
    return static_cast<uint32_t>(level + 1);
  }

private:
  struct Cand
  {
    float dist;
    uint32_t id;
    bool operator<(const Cand &o) const { return dist < o.dist; }
  };

  float distance(uint32_t a, uint32_t b) const
  {
    const float *pa = vectors_ + static_cast<size_t>(gids_[a]) * dim_;
    const float *pb = vectors_ + static_cast<size_t>(gids_[b]) * dim_;
    float sum = 0.0f;
    for (uint32_t d = 0; d < dim_; ++d) {
      const float diff = pa[d] - pb[d];
      sum += diff * diff;
    }
    return sum;
  }

  // Beam search on one level starting from the entry point `start`.
  void search_layer(uint32_t query, uint32_t start, int level, uint32_t ef,
      std::vector<Cand> &result) const
  {
    std::unordered_set<uint32_t> visited;
    std::priority_queue<Cand> top;                 // max-heap on distance
    std::priority_queue<Cand, std::vector<Cand>, std::greater<>> cand;
    const float start_dist = distance(query, start);
    visited.insert(start);
    top.push({start_dist, start});
    cand.push({start_dist, start});
    while (!cand.empty()) {
      const Cand c = cand.top();
      cand.pop();
      if (!top.empty() && c.dist > top.top().dist && top.size() >= ef) {
        break;
      }
      for (uint32_t nb : adj_[level][c.id]) {
        if (visited.insert(nb).second) {
          const float d = distance(query, nb);
          if (top.size() < ef || d < top.top().dist) {
            top.push({d, nb});
            if (top.size() > ef) {
              top.pop();
            }
            cand.push({d, nb});
          }
        }
      }
    }
    result.clear();
    while (!top.empty()) {
      result.push_back(top.top());
      top.pop();
    }
    std::sort(result.begin(), result.end());
  }

  static void select_neighbors(std::vector<Cand> &result, uint32_t m)
  {
    if (result.size() > m) {
      result.resize(m);
    }
  }

  // Drops the farthest neighbor when a node exceeds the degree cap.
  void shrink(int level, uint32_t local)
  {
    auto &adj = adj_[level][local];
    while (adj.size() > m_) {
      float worst = -1.0f;
      size_t worst_idx = 0;
      for (size_t i = 0; i < adj.size(); ++i) {
        const float d = distance(local, adj[i]);
        if (d > worst) {
          worst = d;
          worst_idx = i;
        }
      }
      const uint32_t victim = adj[worst_idx];
      adj.erase(adj.begin() + static_cast<long>(worst_idx));
      auto &rev = adj_[level][victim];
      rev.erase(std::remove(rev.begin(), rev.end(), local), rev.end());
    }
  }

  const float *vectors_;
  uint32_t dim_;
  uint32_t m_;
  uint32_t efc_;
  std::mt19937 rng_;
  double mult_;
  std::unordered_map<uint32_t, uint32_t> gid2local_;
  std::vector<uint32_t> gids_;
  std::vector<int> levels_;                                   // aux levels
  std::vector<std::vector<std::vector<uint32_t>>> adj_;       // [aux level][local][neighbors]
  int cur_max_ = -1;
  uint32_t entry_ = UINT32_MAX;
};

}  // namespace obhipvs

#endif  // OB_HIPVS_HNSW_UPPER_LAYERS_H
