"""Fork-vs-upstream openEMS throughput matrix.

Drives the `openEMS` *binaries* of two builds over identical XML models, so the
comparison never depends on which Python bindings happen to be installed. Each
model is written once with `openEMS.Write2XML()` and contains nothing a stock
upstream build cannot parse: a uniform Cartesian grid, one excitation box, and
a boundary condition. No probes, no dumps, no post-processing.

The reported figure is the binary's own final `Speed: <x> MCells/s`, which
covers the time-stepping loop only. Everything outside that loop (operator
build, engine/GPU init) is reported separately as `setup_s`.

Usage (from the repo root, with the venv that has CSXCAD/openEMS importable):

    python python/Tests/benchmark_fork_vs_upstream.py \
        --fork-build build --upstream-build ../openEMS-upstream-bench/build \
        --out /tmp/bench.json

`--only` filters runs by a substring of "<config>/<engine label>".
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSXCAD_LIB = os.path.expanduser("~/opt/lib")

SPEED_RE = re.compile(r"^Speed:\s+([0-9.eE+-]+)\s+MCells/s", re.M)
# The cell count prints as an integer on short runs and as "262144.00" once
# an interval line has put the stream into fixed notation.
ITER_RE = re.compile(r"Time for \d+ iterations with [\d.]+ cells :\s+([0-9.eE+-]+)\s+sec", re.M)
CELLS_RE = re.compile(r"-->\s+([\d.]+)\s+FDTD cells", re.M)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

def write_model(nx, ny, nz, boundary, timesteps, out_path):
    """Write one benchmark model to `out_path` as openEMS XML."""
    sys.path.insert(0, os.path.join(REPO, "python"))
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS

    bc = [0] * 6
    if boundary == "pml":
        bc = ["PML_8"] * 6
    elif boundary == "mur":
        bc = ["MUR"] * 6

    fdtd = openEMS(NrTS=timesteps, EndCriteria=1.0e-300)
    fdtd.SetGaussExcite(1.0e9, 0.5e9)
    fdtd.SetBoundaryCond(bc)

    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(1.0e-3)
    mesh.SetLines("x", list(range(nx)))
    mesh.SetLines("y", list(range(ny)))
    mesh.SetLines("z", list(range(nz)))

    exc = csx.AddExcitation("benchmark_excitation", exc_type=0, exc_val=[1, 0, 0])
    exc.AddBox([nx // 2, 9, 9], [nx // 2 + 1, ny - 10, nz - 10], priority=10)

    fdtd.Write2XML(out_path)


# --------------------------------------------------------------------------
# runs
# --------------------------------------------------------------------------

def run_once(build, xml, engine, env_extra=None, num_threads=0, cpus=None,
             timeout=3600):
    binary = os.path.join(build, "openEMS")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join([build, CSXCAD_LIB])
    env.update(env_extra or {})

    cmd = []
    if cpus:
        # This host is a hybrid CPU. Left to the scheduler, the same run lands
        # on P-cores or E-cores from one repeat to the next and the result
        # swings by up to 2.6x, which swamps every effect being measured. Rows
        # that compare engines therefore pin to one hardware thread per P-core.
        cmd += ["taskset", "-c", cpus]
    cmd += [binary, xml, "--engine=" + engine]
    if num_threads:
        cmd.append("--numThreads=%d" % num_threads)

    with tempfile.TemporaryDirectory(prefix="oebench_") as cwd:
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True)
        wall = time.time() - t0

    out = proc.stdout
    if proc.returncode != 0:
        return {"ok": False, "error": "exit %d" % proc.returncode, "tail": out[-1500:]}
    m = SPEED_RE.search(out)
    if not m:
        return {"ok": False, "error": "no Speed line", "tail": out[-1500:]}

    iters = ITER_RE.search(out)
    cells = CELLS_RE.search(out)
    loop_s = float(iters.group(1)) if iters else None
    return {
        "ok": True,
        "mcells_s": float(m.group(1)),
        "loop_s": loop_s,
        # everything the run spent outside the stepping loop: operator build,
        # engine/GPU init, teardown.
        "setup_s": round(wall - loop_s, 2) if loop_s is not None else None,
        "cells": int(float(cells.group(1))) if cells else None,
        "wall_s": wall,
    }


def run_best(repeats, **kw):
    """Best-of-N. Throughput noise on a shared desktop is one-sided."""
    results = [run_once(**kw) for _ in range(repeats)]
    good = [r for r in results if r["ok"]]
    if not good:
        return results[0]
    best = max(good, key=lambda r: r["mcells_s"])
    best["samples"] = [round(r["mcells_s"], 1) for r in good]
    return best


# --------------------------------------------------------------------------
# matrix
# --------------------------------------------------------------------------

# (name, nx, ny, nz, boundary, timesteps, repeats)
#
# The 64x64x64 grid is L3-resident and, on this host, bimodal: at fixed core,
# fixed 5.3 GHz and identical hugepage backing it returns either ~480 or
# ~170-310 MC/s from run to run. The DRAM-bound grids are stable to 0.5%, which
# points at the uncore/ring clock -- one active core does not hold it up, and
# only L3-resident work notices. Hence more repeats there; best-of-N picks the
# un-throttled mode.
CONFIGS = [
    ("64x64x64 pec",     64,  64,  64,  "pec", 40000, 6),
    ("160x128x192 pec",  160, 128, 192, "pec", 3000,  3),
    ("224x224x224 pec",  224, 224, 224, "pec", 1200,  3),
    ("160x128x192 pml",  160, 128, 192, "pml", 1500,  3),
    ("224x224x224 pml",  224, 224, 224, "pml", 600,   3),
    ("224x224x224 mur",  224, 224, 224, "mur", 1000,  3),
]


# One hardware thread on each of the eight P-cores; CPUs 16-31 are E-cores.
PIN_1 = "0"
PIN_8 = "0,2,4,6,8,10,12,14"


def engine_matrix(args):
    """(label, build, engine, env, num_threads, cpus)

    Two families of row. The *pinned* rows fix the thread count and the core
    placement identically on both sides, and are what the engine-vs-engine
    speedups are computed from. The *default* rows run exactly as a user gets
    them -- no pinning, no --numThreads -- and so also measure each build's own
    thread auto-tune.
    """
    blk = {"OPENEMS_AVX2_TEMPORAL_BLOCK": str(args.block_k)}
    t = args.tuned_threads
    return [
        ("upstream sse 1t",      args.upstream_build, "sse",                {}, 0, PIN_1),
        ("upstream mt 8t",       args.upstream_build, "multithreaded",      {}, t, PIN_8),
        ("upstream mt default",  args.upstream_build, "multithreaded",      {}, 0, None),
        ("fork avx2 1t",         args.fork_build,     "avx2",               {}, 0, PIN_1),
        ("fork avx2-mt 8t",      args.fork_build,     "avx2-multithreaded", {}, t, PIN_8),
        ("fork avx2-mt default", args.fork_build,     "avx2-multithreaded", {}, 0, None),
        ("fork tblock 8t",       args.fork_build,     "avx2-multithreaded", blk, t, PIN_8),
        ("fork tblock default",  args.fork_build,     "avx2-multithreaded", blk, 0, None),
        ("fork gpu rx6800",      args.fork_build,     "gpu", {"OPENEMS_GPU_INDEX": "0"}, 0, None),
        ("fork gpu uhd770",      args.fork_build,     "gpu", {"OPENEMS_GPU_INDEX": "1"}, 0, None),
    ]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fork-build", default=os.path.join(REPO, "build-bench"))
    p.add_argument("--upstream-build",
                   default=os.path.abspath(os.path.join(REPO, "..", "openEMS-upstream-bench", "build")))
    p.add_argument("--out", default="/tmp/oebench/results.json")
    p.add_argument("--repeats", type=int, default=3,
                   help="floor on the per-config repeat count in CONFIGS")
    p.add_argument("--block-k", default="16")
    p.add_argument("--tuned-threads", type=int, default=8)
    p.add_argument("--only", default=None)
    p.add_argument("--xml-dir", default="/tmp/oebench/xml")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.xml_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    results = []
    if os.path.exists(args.out):
        results = json.load(open(args.out))

    def already(cfg, label):
        return any(r["config"] == cfg and r["engine"] == label for r in results)

    for cfg_name, nx, ny, nz, boundary, ts, reps in CONFIGS:
        xml = os.path.join(args.xml_dir, cfg_name.replace(" ", "_") + ".xml")
        if not os.path.exists(xml):
            write_model(nx, ny, nz, boundary, ts, xml)

        for label, build, engine, env, threads, cpus in engine_matrix(args):
            key = "%s/%s" % (cfg_name, label)
            if args.only and args.only not in key:
                continue
            if already(cfg_name, label):
                continue
            print("### %-46s " % key, end="", flush=True)
            r = run_best(max(reps, args.repeats), build=build, xml=xml, engine=engine,
                         env_extra=env, num_threads=threads, cpus=cpus)
            r.update(config=cfg_name, engine=label, cells=nx * ny * nz,
                     timesteps=ts, boundary=boundary, grid=[nx, ny, nz],
                     cpus=cpus, num_threads=threads)
            results.append(r)
            json.dump(results, open(args.out, "w"), indent=1)
            print(("%8.1f MC/s  %s" % (r["mcells_s"], r.get("samples")))
                  if r["ok"] else "FAILED: " + r["error"])

    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
