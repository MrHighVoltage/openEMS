"""Controlled openEMS field-dump benchmark.

Measures wall-clock time of a run whose cost is dominated by full-domain
time-domain field dumps, rather than by the FDTD update itself.  Used to
A/B the host-side dump path: the asynchronous ring-buffer writer
(`Common/async_field_writer.*`, gated by OPENEMS_ASYNC_DUMP_BUFFERS), the
persistent HDF5 handle and the threaded field extraction.

    # asynchronous writer (default, 4 ring slots) vs. synchronous writes
    python python/Tests/benchmark_field_dump.py --dump-dir ~/scratch
    OPENEMS_ASYNC_DUMP_BUFFERS=0 python python/Tests/benchmark_field_dump.py \
        --dump-dir ~/scratch

Point --dump-dir at a real filesystem; the default temporary directory is
tmpfs on many hosts, which measures memcpy rather than storage.
"""

import argparse
import os
import shutil
import sys
import tempfile
import time


def _bootstrap_local_openems_runtime():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    python_dir = os.path.join(repo_root, "python")
    build_dir = os.path.join(repo_root, "build")

    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)

    libopenems = os.path.join(build_dir, "libopenEMS.so")
    if not os.path.exists(libopenems):
        raise RuntimeError(f"Local openEMS library not found: {libopenems}")

    paths = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
    if paths and os.path.abspath(paths[0]) == os.path.abspath(build_dir):
        return

    paths = [path for path in paths if os.path.abspath(path) != os.path.abspath(build_dir)]
    os.environ["LD_LIBRARY_PATH"] = ":".join([build_dir] + paths)
    os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)


_bootstrap_local_openems_runtime()

from CSXCAD import ContinuousStructure
from openEMS import openEMS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs=3, type=int, metavar=("NX", "NY", "NZ"),
                        default=(128, 128, 128))
    parser.add_argument("--timesteps", type=int, default=400)
    parser.add_argument("--engine", default="avx2-multithreaded")
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--file-type", choices=("hdf5", "vtk"), default="hdf5")
    parser.add_argument("--dump-dir", default=None,
                        help="parent directory for the simulation output")
    parser.add_argument("--keep-output", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    nx, ny, nz = args.cells

    # OverSampling above the Nyquist rate forces a dump on every timestep,
    # which is the worst case for the dump path and the point of the fixture.
    fdtd = openEMS(NrTS=args.timesteps, EndCriteria=1.0e-300,
                   OverSampling=args.timesteps)
    fdtd.SetGaussExcite(1.0e9, 0.5e9)
    fdtd.SetBoundaryCond([0] * 6)

    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(1.0e-3)
    mesh.SetLines("x", list(range(nx)))
    mesh.SetLines("y", list(range(ny)))
    mesh.SetLines("z", list(range(nz)))

    excitation = csx.AddExcitation("benchmark_excitation", exc_type=0, exc_val=[1, 0, 0])
    excitation.AddBox([nx // 2, 9, 9], [nx // 2 + 1, ny - 10, nz - 10], priority=10)

    dump = csx.AddDump("Et", dump_type=0, dump_mode=0,
                       file_type=1 if args.file_type == "hdf5" else 0)
    dump.AddBox([0, 0, 0], [nx - 1, ny - 1, nz - 1])

    parent = args.dump_dir or tempfile.gettempdir()
    sim_path = os.path.join(os.path.expanduser(parent), "openems_field_dump_benchmark")
    shutil.rmtree(sim_path, ignore_errors=True)

    cells = nx * ny * nz
    print(
        "BENCHMARK_CONFIG"
        f" cells={nx}x{ny}x{nz}"
        f" timesteps={args.timesteps}"
        f" engine={args.engine}"
        f" file_type={args.file_type}"
        f" dump_bytes_per_ts={3 * cells * 4}"
        f" async_buffers={os.environ.get('OPENEMS_ASYNC_DUMP_BUFFERS', 'default(4)')}"
        f" path={sim_path}",
        flush=True,
    )

    run_options = {"cleanup": True, "engine": args.engine}
    if args.num_threads:
        run_options["numThreads"] = args.num_threads

    start = time.perf_counter()
    fdtd.Run(sim_path, **run_options)
    wall = time.perf_counter() - start

    dump_file = os.path.join(sim_path, "Et.h5")
    size = os.path.getsize(dump_file) if os.path.exists(dump_file) else 0
    print(
        f"BENCHMARK_RESULT wall_s={wall:.2f}"
        f" MC_per_s={cells * args.timesteps / wall / 1e6:.1f}"
        f" dump_file_MB={size / 1e6:.1f}",
        flush=True,
    )

    if not args.keep_output:
        shutil.rmtree(sim_path, ignore_errors=True)


if __name__ == "__main__":
    main()
