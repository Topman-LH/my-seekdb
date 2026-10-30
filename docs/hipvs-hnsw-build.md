# GPU-accelerated HNSW bulk build with hipVS (AMD ROCm)

Use the AMD GPU (through **hipVS**, the ROCm distribution of the cuVS C API)
to accelerate the construction of HNSW vector indexes in seekdb, without
changing the on-disk format, the search path, or any user-visible behaviour.

## Design

Bulk HNSW build is dominated by the O(n log n) neighbour selection on the
base layer. That is a large, batched, data-parallel workload - exactly what a
GPU is good at - while the upper routing layers only cover ~n/M points and
stay on the host.

```
                       seekdb server (CPU, VSAG)
  HnswIndexHandler::build_index(base)
        │  OB_HIPVS_WORKER set and rows >= gate?
        │
        ▼  framed stdin (64-byte header + float32 matrix, streamed)
  ┌───────────────────────── worker process ─────────────────────────┐
  │ 1. cuvsCagraBuild        → flat kNN graph on the GPU (NN-Descent │
  │    (hipVS / cuVS C API)    for < 2^18 rows, IVF-PQ above)        │
  │ 2. host-side UpperHnsw   → lifts ~n/M sampled entries (plus one   │
  │    (hnsw_upper_layers.h)   per connected component) into routing  │
  │                            layers; splices cross-component links  │
  │                            into the base layer                    │
  │ 3. verify_reachability   → every vector reachable from entry      │
  │ 4. serialize_snapshot    → VSAG hnswlib on-disk layout            │
  └──────────────────────────┬───────────────────────────────────────┘
                             ▼  stdout (validated snapshot)
        index_->Deserialize(snapshot)   → same index, GPU-built graph
        (any failure → fallback to the regular CPU Build())
```

Key properties:

- **Process isolation** - hipVS and VSAG expose overlapping C++ symbols, so
  the GPU part lives in a standalone binary; only bytes cross the boundary.
- **Format-compatible** - the worker emits the VSAG hnswlib on-disk layout,
  so the server loads it with a plain `Deserialize`; no conversion step, no
  format fork. The 96-byte header is structurally validated before use.
- **Correctness guards** - connected components of the (possibly
  disconnected) CAGRA graph are bridged through their entry nodes; the worker
  refuses to emit a snapshot unless every vector is reachable from the entry
  point; an optional built-in recall self-test (`selftest` mode) checks
  top-10 recall against brute force.
- **Zero-risk fallback** - everything is runtime-gated by the
  `OB_HIPVS_WORKER` environment variable. No GPU, no worker, or a rejected
  snapshot simply falls back to the unchanged CPU build path.

## Repository layout

```
tools/hipvs_hnsw/hipvs_hnsw_worker.cpp   GPU worker (hipVS/cuVS C API + HIP)
tools/hipvs_hnsw/hnsw_upper_layers.h     host-side routing-layer HNSW
tools/hipvs_hnsw/snapshot_format.h       shared snapshot layout + validation
tools/hipvs_hnsw/build_worker.sh         hipcc build script
tools/hipvs_hnsw/benchmark/benchmark.py  throughput / recall benchmark
src/oblib/lib/vector/ob_hipvs_offload.h  server-side offload (fork/exec + pipes)
docs/integration_patch.md                exact build_index() wiring
docs/hipvs-hnsw-build.md                 this file
```

## Build & run (AMD Radeon Cloud notebook / ROCm host)

```bash
cd tools/hipvs_hnsw
./build_worker.sh ./hipvs_hnsw_worker

# quick correctness check (builds + 100-query recall self-test):
python3 benchmark/benchmark.py --worker ./hipvs_hnsw_worker \
    --rows 100000 --dim 128 --m 16 --efc 200 --selftest

# throughput at scale:
python3 benchmark/benchmark.py --worker ./hipvs_hnsw_worker \
    --rows 1000000 --dim 128 --m 16 --efc 200
```

Enable in seekdb:

```bash
export OB_HIPVS_WORKER=/path/to/hipvs_hnsw_worker
export OB_HIPVS_OFFLOAD_MIN_ROWS=65536   # optional, default 65536
```

Wiring inside the server: see `docs/integration_patch.md` (two edits in
`src/oblib/lib/vector/ob_vsag_adaptor.cpp`).

## Parameter mapping

| seekdb / VSAG        | worker build request      | snapshot header        |
| -------------------- | ------------------------- | ---------------------- |
| dim                  | dim                       | size_data_per_element  |
| max_degree (M)       | m                         | max_m = M, max_m0 = 2M |
| ef_construction      | ef_construction           | ef_construction        |
| row count            | n                         | max_elements / count   |

## Verification line

OB-P:6emIAkt5rPOgNl3c__UZjg
