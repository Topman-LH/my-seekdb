/*
 * Copyright (c) 2026 seekdb competition entry.
 * Licensed under the Apache License, Version 2.0.
 *
 * GPU offload for HNSW bulk build through hipVS.
 *
 * HNSW bulk build is dominated by the O(n log n) neighbour-selection work on
 * the base layer. This offload hands the whole vector matrix to a standalone
 * worker process (tools/hipvs_hnsw/hipvs_hnsw_worker) which builds the base
 * kNN graph on an AMD GPU through hipVS (the ROCm port of the cuVS C API) and
 * returns a VSAG-loadable HNSW snapshot on stdout. The caller deserializes
 * that snapshot into the already-created VSAG index - the same code path used
 * by fdeserialize - so search behaviour is unchanged.
 *
 * The worker is a separate process on purpose: hipVS and VSAG expose
 * overlapping C++ symbols and must never be linked into one binary.
 *
 * Activation (all runtime, no schema change):
 *   OB_HIPVS_WORKER=/path/to/hipvs_hnsw_worker   required, enables offload
 *   OB_HIPVS_OFFLOAD_MIN_ROWS=65536              optional gate (default 65536)
 *
 * Failure policy: any failure (GPU missing, worker crash, snapshot rejected)
 * logs a warning and returns non-OB_SUCCESS; build_index falls back to the
 * regular CPU VSAG build.
 */
#ifndef OB_HIPVS_OFFLOAD_H
#define OB_HIPVS_OFFLOAD_H

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>

#include "lib/oblog/ob_log.h"
// Snapshot wire-format header lives INSIDE this package (lib/vector) so it is
// covered by oblib's Bazel glob (lib/**/*.h) and stays inside the hermetic
// build sandbox. Do NOT include the tools/hipvs_hnsw copy from here: it is
// outside the package and breaks Bazel (file-not-found + layering_check).
// Keep the two copies' 96-byte layout in sync.
#include "ob_hipvs_snapshot_format.h"

// Framed stdin protocol shared with tools/hipvs_hnsw/hipvs_hnsw_worker.cpp
// (the worker keeps its own copy inside its translation unit).
namespace obhipvs_offload
{
constexpr uint64_t PROTOCOL_MAGIC = 0x5748554154564842ULL;  // "BHVTUAHW" LE
constexpr uint32_t PROTOCOL_VERSION = 1;
constexpr uint32_t PROTOCOL_HEADER_SIZE = 64;

struct OffloadConfig
{
  std::string worker_path;
  uint64_t min_rows = 65536;
};

// Offload is enabled only when OB_HIPVS_WORKER points to an executable.
inline bool load_config(OffloadConfig &cfg)
{
  const char *path = std::getenv("OB_HIPVS_WORKER");
  if (path == nullptr || path[0] == '\0') {
    return false;
  }
  if (0 != access(path, X_OK)) {
    _OB_LOG(WARN, "[HIPVS] OB_HIPVS_WORKER is not executable, offload disabled");
    return false;
  }
  cfg.worker_path = path;
  const char *min_rows = std::getenv("OB_HIPVS_OFFLOAD_MIN_ROWS");
  if (min_rows != nullptr) {
    uint64_t v = std::strtoull(min_rows, nullptr, 10);
    if (v > 0) {
      cfg.min_rows = v;
    }
  }
  return true;
}

// Writes the 64-byte framed request plus the vector matrix to the worker's
// stdin and reads the snapshot from its stdout. Returns OB_SUCCESS only when
// the child exited 0 and the snapshot passes structural validation.
inline int gpu_build_snapshot(const float *vectors, uint64_t count, uint64_t dim,
    uint64_t m, uint64_t ef_construction, const OffloadConfig &cfg,
    std::string &snapshot)
{
  int ret = OB_SUCCESS;
  int in_pipe[2] = {-1, -1};
  int out_pipe[2] = {-1, -1};
  pid_t pid = -1;

  if (OB_ISNULL(vectors) || count == 0 || dim == 0 ||
      m < 4 || m > obhipvs::MAX_BASE_DEGREE || count <= 4 * m) {
    ret = OB_INVALID_ARGUMENT;
  } else if (OB_FAIL(::pipe(in_pipe)) || OB_FAIL(::pipe(out_pipe))) {
    ret = OB_ERR_SYS;
    _OB_LOG(WARN, "[HIPVS] cannot create pipes for the GPU worker");
  } else if ((pid = fork()) < 0) {
    ret = OB_ERR_SYS;
    _OB_LOG(WARN, "[HIPVS] fork failed");
  } else if (pid == 0) {
    // child: stdin <- in_pipe[0], stdout -> out_pipe[1]
    close(in_pipe[1]);
    close(out_pipe[0]);
    dup2(in_pipe[0], 0);
    dup2(out_pipe[1], 1);
    execl(cfg.worker_path.c_str(), cfg.worker_path.c_str(), "build",
        static_cast<char *>(nullptr));
    _exit(127);
  }

  if (OB_SUCC(ret)) {
    close(in_pipe[0]);  in_pipe[0] = -1;
    close(out_pipe[1]); out_pipe[1] = -1;

    // The worker may exit early (bad GPU, bad shape); writes must fail with
    // EPIPE instead of raising SIGPIPE, which would kill the server.
    sigset_t pipe_mask, old_mask;
    sigemptyset(&pipe_mask);
    sigaddset(&pipe_mask, SIGPIPE);
    sigprocmask(SIG_BLOCK, &pipe_mask, &old_mask);

    const uint64_t header[8] = {
        PROTOCOL_MAGIC, PROTOCOL_VERSION,
        PROTOCOL_HEADER_SIZE, count, dim, m, ef_construction, 0};
    const uint64_t total_floats = count * dim;
    bool io_ok = (sizeof(header) ==
        static_cast<size_t>(write(in_pipe[1], header, sizeof(header))));
    const float *cur = vectors;
    uint64_t left = total_floats;
    while (io_ok && left > 0) {
      const uint64_t take = std::min<uint64_t>(left, 2UL << 20);
      const ssize_t w = write(in_pipe[1], cur, take * sizeof(float));
      io_ok = (w == static_cast<ssize_t>(take * sizeof(float)));
      cur += take;
      left -= take;
    }
    close(in_pipe[1]);
    in_pipe[1] = -1;

    char buf[1 << 20];
    ssize_t r = 0;
    while (io_ok && (r = read(out_pipe[0], buf, sizeof(buf))) > 0) {
      snapshot.append(buf, static_cast<size_t>(r));
    }
    close(out_pipe[0]);
    out_pipe[0] = -1;

    sigprocmask(SIG_SETMASK, &old_mask, nullptr);

    int status = 0;
    if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status) ||
        WEXITSTATUS(status) != 0) {
      ret = OB_ERR_UNEXPECTED;
      _OB_LOG(WARN, "[HIPVS] GPU worker did not exit cleanly");
    } else if (!io_ok) {
      ret = OB_IO_ERROR;
      _OB_LOG(WARN, "[HIPVS] pipe IO with the GPU worker failed");
    } else {
      try {
        obhipvs::validate_header(snapshot.data(), snapshot.size(), count, dim, m);
      } catch (const std::exception &e) {
        ret = OB_ERR_UNEXPECTED;
        _OB_LOG(WARN, "[HIPVS] snapshot rejected: %s", e.what());
      }
    }
  }
  if (in_pipe[0] >= 0) close(in_pipe[0]);
  if (in_pipe[1] >= 0) close(in_pipe[1]);
  if (out_pipe[0] >= 0) close(out_pipe[0]);
  if (out_pipe[1] >= 0) close(out_pipe[1]);
  return ret;
}

}  // namespace obhipvs_offload

#endif  // OB_HIPVS_OFFLOAD_H
