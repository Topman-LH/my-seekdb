# Integration patch: ob_vsag_adaptor.cpp

Two minimal edits wire the GPU offload into the existing HNSW build path.
Everything is runtime-gated; without `OB_HIPVS_WORKER` set the binary behaves
exactly as before.

## 1) Add the include

`src/oblib/lib/vector/ob_vsag_adaptor.cpp`, after the existing includes
(near line 29):

```cpp
#include "lib/vector/ob_hipvs_offload.h"
```

## 2) Add the offload branch to HnswIndexHandler::build_index

`src/oblib/lib/vector/ob_vsag_adaptor.cpp`, function
`int HnswIndexHandler::build_index(const vsag::DatasetPtr &base)`
(around line 218). Insert the marked block between the `try {` and the
existing `index_->Build(base)` call:

```cpp
int HnswIndexHandler::build_index(const vsag::DatasetPtr &base)
{
  int ret = OB_SUCCESS;
  try {
    // ---- hipVS GPU offload (added) --------------------------------------
    obhipvs_offload::OffloadConfig cfg;
    if (obhipvs_offload::load_config(cfg) &&
        base->GetNumElements() >= cfg.min_rows) {
      std::string snapshot;
      if (OB_FAIL(obhipvs_offload::gpu_build_snapshot(
              base->GetFloat32Vectors(),
              static_cast<uint64_t>(base->GetNumElements()),
              static_cast<uint64_t>(base->GetDim()),
              static_cast<uint64_t>(get_max_degree()),
              static_cast<uint64_t>(get_ef_construction()), cfg, snapshot))) {
        _OB_LOG(WARN, "[HIPVS] GPU offload failed, falling back to the CPU build");
      } else {
        std::stringstream stream(snapshot);
        tl::expected<void, Error> loaded = index_->Deserialize(stream);
        if (loaded.has_value()) {
          _OB_LOG(INFO, "[HIPVS] HNSW built by the hipVS GPU worker and loaded "
                        "from its snapshot", K(base->GetNumElements()), K_(dim));
          return OB_SUCCESS;
        }
        _OB_LOG(WARN, "[HIPVS] snapshot rejected by VSAG, falling back to the CPU build",
                K(loaded.error().type));
      }
    }
    // ---- end hipVS GPU offload ------------------------------------------

    tl::expected<std::vector<int64_t>, Error> result = index_->Build(base);
    if (result.has_value()) {
    } else {
      ret = vsag_errcode2ob(result.error().type);
    }
  } catch (const std::exception &e) {
    ...
```

### Why `index_->Deserialize()` is safe here

`index_` is the VSAG index created by the create phase with exactly the
parameters the worker baked into the snapshot header (dim, max_degree = M,
ef_construction). `fdeserialize()` uses the identical
`Factory::CreateIndex` + `Deserialize` flow for the replay path, so loading a
GPU-built snapshot through `Deserialize` on the existing empty index leaves
search behaviour untouched: same index type, same search parameters, same
label space (label == row id, matching what `Build()` assigns).

### Failure policy

Every offload failure path falls through to the original
`index_->Build(base)` CPU build, so a missing GPU, a crashed worker, or a
rejected snapshot can never fail a user DDL.
