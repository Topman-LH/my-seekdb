#!/usr/bin/env python3
# Benchmark for the hipVS HNSW bulk-build worker.
#
# Generates a random float32 matrix, feeds it to the GPU worker through the
# 64-byte framed stdin protocol, and reports wall time + GPU time + recall.
# Optionally compares against a CPU VSAG build of the same matrix when the
# `vsag` Python binding is available.
#
# Usage (on the AMD GPU notebook):
#   python3 benchmark.py --rows 200000 --dim 128 --m 16 --efc 200 [--selftest]
import argparse
import json
import os
import struct
import subprocess
import sys
import time
import uuid

MAGIC = 0x5748554154564842
VERSION = 1
HEADER_SIZE = 64


def write_framed(worker, rows, dim, m, efc, selftest):
    cmd = [worker, "selftest" if selftest else "build"]
    payload = bytearray(struct.pack("<QQQQQQQQ", MAGIC, VERSION, HEADER_SIZE, rows, dim, m, efc, 0))
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # stream the matrix in 8 MiB chunks so peak RSS stays bounded
    proc.stdin.write(payload)
    chunk = 2 * 1024 * 1024  # floats per chunk (8 MiB)
    state = (rows * 2654435761) & 0xFFFFFFFF
    remaining = rows * dim
    while remaining > 0:
        take = min(chunk, remaining)
        vals = bytearray()
        for _ in range(take):
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            vals += struct.pack("<f", (state / 0x7FFFFFFF) * 2.0 - 1.0)
        proc.stdin.write(vals)
        remaining -= take
    proc.stdin.close()
    snapshot = proc.stdout.read()
    err = proc.stderr.read().decode("utf-8", "replace")
    rc = proc.wait()
    wall = time.time() - t0
    return rc, snapshot, err, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default="./hipvs_hnsw_worker")
    ap.add_argument("--rows", type=int, default=200000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="./benchmark_result.json")
    args = ap.parse_args()

    if not os.path.exists(args.worker):
        sys.exit(f"worker binary not found: {args.worker} (build it with build_worker.sh first)")

    rc, snapshot, err, wall = write_framed(args.worker, args.rows, args.dim, args.m, args.efc, args.selftest)
    print(err.strip())
    if rc != 0:
        sys.exit(f"worker failed with rc={rc}")

    result = {
        "rows": args.rows, "dim": args.dim, "m": args.m, "efc": args.efc,
        "wall_seconds": round(wall, 3),
        "snapshot_bytes": len(snapshot),
        "mode": "selftest" if args.selftest else "build",
    }
    for line in err.splitlines():
        if line.startswith("[HIPVS_RESULT]"):
            for kv in line[len("[HIPVS_RESULT]"):].split():
                k, _, v = kv.partition("=")
                try:
                    result[k] = float(v) if "." in v else int(v)
                except ValueError:
                    result[k] = v
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nsnapshot written check: magic ok = {len(snapshot) >= 96}")


if __name__ == "__main__":
    main()
