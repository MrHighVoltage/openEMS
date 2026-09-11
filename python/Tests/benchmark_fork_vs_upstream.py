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

Everything host-specific is an option, because the same matrix is run on more
than one machine: `--pin-single` / `--pin-tuned` / `--tuned-threads` for the
core layout, `--gpu label:index[:k]` (repeatable, `--gpu none` for CPU-only)
for the devices, `--csxcad-lib` for where libCSXCAD.so lives.
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
DEFAULT_CSXCAD_LIB = os.path.expanduser("~/opt/lib")

SPEED_RE = re.compile(r"^Speed:\s+([0-9.eE+-]+)\s+MCells/s", re.M)
# The cell count's format depends on what the stream printed before it: an
# integer on a short run, "262144.00" once an interval line has switched to
# fixed notation, and "1.12394e+07" on a run fast enough to print no interval
# line at all -- which is every GPU run on a large grid. Accept all three, or
# setup_s silently becomes None on exactly the rows that have the most setup.
ITER_RE = re.compile(r"Time for \d+ iterations with [0-9.eE+-]+ cells :\s+([0-9.eE+-]+)\s+sec", re.M)
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
             timeout=3600, csxcad_lib=DEFAULT_CSXCAD_LIB):
    binary = os.path.join(build, "openEMS")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join([build, csxcad_lib])
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
                              # universal_newlines, not text=: the remote hosts
                              # this matrix also runs on ship Python 3.6, where
                              # text= does not exist yet.
                              universal_newlines=True)
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
    # The schedule announces itself, and announces what made it refuse. Keep
    # that line: a tblock row that quietly fell back to the flat sweep is
    # otherwise indistinguishable from one that engaged and gained nothing.
    notes = [ln.strip() for ln in out.splitlines() if "temporal blocking" in ln]
    return {
        "ok": True,
        "mcells_s": float(m.group(1)),
        "loop_s": loop_s,
        "notes": notes,
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


# Default pinning for the i9-13900K this matrix was first run on: one hardware
# thread on each of the eight P-cores; CPUs 16-31 are E-cores. Other hosts pass
# --pin-single / --pin-tuned / --tuned-threads.
DEFAULT_PIN_1 = "0"
DEFAULT_PIN_N = "0,2,4,6,8,10,12,14"

# label:index[:k] -- k is the GPU block depth for that device's +tblock row.
DEFAULT_GPUS = ["rx6800:0:16", "uhd770:1:6"]


def parse_gpu_spec(spec):
    """Parse "label:index[:k]". k is None when unset; the caller defaults it."""
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError("--gpu wants label:index[:k], got %r" % spec)
    return parts[0], parts[1], (parts[2] if len(parts) == 3 else None)


def engine_matrix(args):
    """(label, build, engine, env, num_threads, cpus)

    Three families of row. The *pinned* rows fix the thread count and the core
    placement identically on both sides, and are what the engine-vs-engine
    speedups are computed from. The *default* rows run exactly as a user gets
    them -- no pinning, no --numThreads -- and so also measure each build's own
    thread auto-tune. The *GPU* rows are neither: the CPU is not the resource
    under test, so pinning it would only add a variable.
    """
    blk = {"OPENEMS_AVX2_TEMPORAL_BLOCK": str(args.block_k)}
    t = args.tuned_threads
    p1, pn = args.pin_single, args.pin_tuned
    rows = [
        ("upstream sse 1t",       args.upstream_build, "sse",                {}, 0, p1),
        ("upstream mt %dt" % t,   args.upstream_build, "multithreaded",      {}, t, pn),
        ("upstream mt default",   args.upstream_build, "multithreaded",      {}, 0, None),
        ("fork avx2 1t",          args.fork_build,     "avx2",               {}, 0, p1),
        ("fork avx2-mt %dt" % t,  args.fork_build,     "avx2-multithreaded", {}, t, pn),
        ("fork avx2-mt default",  args.fork_build,     "avx2-multithreaded", {}, 0, None),
        ("fork tblock %dt" % t,   args.fork_build,     "avx2-multithreaded", blk, t, pn),
        ("fork tblock default",   args.fork_build,     "avx2-multithreaded", blk, 0, None),
    ]
    for spec in args.gpu:
        label, index, k = parse_gpu_spec(spec)
        k = k or args.gpu_block_k
        rows.append(("fork gpu %s" % label, args.fork_build, "gpu",
                     {"OPENEMS_GPU_INDEX": index}, 0, None))
        # Same device, same run, with the trapezoidal schedule turned on. The
        # engine prints why it refused when it does, and the flat row above is
        # the control that says what the refusal cost.
        rows.append(("fork gpu %s tblock" % label, args.fork_build, "gpu",
                     {"OPENEMS_GPU_INDEX": index,
                      "OPENEMS_GPU_TEMPORAL_BLOCK": k}, 0, None))
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fork-build", default=os.path.join(REPO, "build-bench"))
    p.add_argument("--upstream-build",
                   default=os.path.abspath(os.path.join(REPO, "..", "openEMS-upstream-bench", "build")))
    p.add_argument("--out", default="/tmp/oebench/results.json")
    p.add_argument("--repeats", type=int, default=3,
                   help="floor on the per-config repeat count in CONFIGS")
    p.add_argument("--block-k", default="16",
                   help="AVX2 block depth (OPENEMS_AVX2_TEMPORAL_BLOCK)")
    p.add_argument("--gpu-block-k", default="16",
                   help="GPU block depth for --gpu entries that do not name one")
    p.add_argument("--gpu", action="append", default=None, metavar="LABEL:INDEX[:K]",
                   help="repeatable; default: " + " ".join(DEFAULT_GPUS)
                        + ". Pass --gpu none for a CPU-only matrix.")
    p.add_argument("--tuned-threads", type=int, default=8)
    p.add_argument("--pin-single", default=DEFAULT_PIN_1,
                   help="taskset CPU list for the 1-thread rows")
    p.add_argument("--pin-tuned", default=DEFAULT_PIN_N,
                   help="taskset CPU list for the --tuned-threads rows")
    p.add_argument("--csxcad-lib", default=DEFAULT_CSXCAD_LIB,
                   help="directory holding libCSXCAD.so, added to LD_LIBRARY_PATH")
    p.add_argument("--only", default=None)
    p.add_argument("--xml-dir", default="/tmp/oebench/xml")
    args = p.parse_args()
    if args.gpu is None:
        args.gpu = list(DEFAULT_GPUS)
    elif args.gpu == ["none"]:
        args.gpu = []
    return args


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
                         env_extra=env, num_threads=threads, cpus=cpus,
                         csxcad_lib=args.csxcad_lib)
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
