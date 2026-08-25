# -*- coding: utf-8 -*-
"""Shared helpers for GPU-vs-CPU regression tests (test_gpu_engine.py).

openEMS silently falls back to a CPU engine when `engine="gpu"` is requested
but the library wasn't built with WITH_GPU (or GPU init fails) -- the C++ CLI
option parser only registers the "gpu" engine value under #ifdef WITH_GPU, so
an unrecognized value simply leaves the default engine selected, no exception
raised. To reliably detect whether the GPU engine actually ran, we capture the
process's real stdout (fd 1) around a minimal run and look for the banner
openEMS prints the moment it selects the Vulkan engine:
    "openEMS - enabled Vulkan GPU engine"      (openems.cpp)
This only appears when WITH_GPU was compiled in and GPU init succeeded.
"""

import contextlib
import os
import tempfile

import numpy as np


@contextlib.contextmanager
def capture_c_stdout():
    """Capture everything written to OS-level stdout (fd 1), including from C/C++.

    `contextlib.redirect_stdout` only redirects Python's `sys.stdout` object; it
    does not affect writes openEMS's C++ core makes directly to fd 1 via `cout`.
    This redirects the fd itself so C++ output is captured too.
    """
    stdout_fd = 1
    saved_fd = os.dup(stdout_fd)
    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        os.dup2(tmp.fileno(), stdout_fd)
        try:
            yield tmp
        finally:
            os.dup2(saved_fd, stdout_fd)
    finally:
        os.close(saved_fd)
    tmp.seek(0)


_gpu_available_cache = None


def gpu_engine_available():
    """Return True iff the local openEMS build can actually run engine="gpu".

    Runs a single-cell, single-timestep simulation and checks the captured
    stdout for the GPU-engine-selected banner. Cached after the first call.
    """
    global _gpu_available_cache
    if _gpu_available_cache is not None:
        return _gpu_available_cache

    try:
        from CSXCAD import ContinuousStructure
        from openEMS import openEMS

        sim_path = os.path.join(tempfile.gettempdir(), "gpu_avail_probe")

        with capture_c_stdout() as captured:
            FDTD = openEMS(NrTS=1, EndCriteria=1e-300)
            FDTD.SetGaussExcite(1e9, 0.5e9)
            FDTD.SetBoundaryCond([0] * 6)

            CSX = ContinuousStructure()
            FDTD.SetCSX(CSX)
            mesh = CSX.GetGrid()
            mesh.SetDeltaUnit(1e-3)
            mesh.AddLine("x", list(range(8)))
            mesh.AddLine("y", list(range(8)))
            mesh.AddLine("z", list(range(8)))

            exc = CSX.AddExcitation("probe_exc", exc_type=0, exc_val=[0, 0, 1])
            exc.AddBox([3, 3, 3], [4, 4, 4])

            FDTD.Run(sim_path, cleanup=True, engine="gpu")

        text = captured.read().decode("utf-8", errors="replace")
        _gpu_available_cache = "enabled Vulkan GPU engine" in text
    except Exception:
        _gpu_available_cache = False

    return _gpu_available_cache


def assert_engines_agree(a, b, *, rtol=2e-2, atol=None, atol_frac=3e-2, label=""):
    """Compare two engines' numeric results, raising a readable AssertionError.

    GPU and CPU results are not bit-identical: different floating-point
    accumulation order, and the GPU shaders run in single precision while the
    CPU engine runs in double, so their respective "zero" noise floors differ
    by several orders of magnitude. Near a zero-crossing this makes a purely
    relative tolerance blow up on values that are both, in absolute terms,
    negligible. So this uses numpy's combined criterion
    (|a-b| <= atol + rtol*|b|) with `atol` defaulted to a fraction of the
    signal's own peak amplitude, rather than a near-zero constant.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape, (
        "{}: shape mismatch, gpu={} cpu={}".format(label, a.shape, b.shape)
    )
    if atol is None:
        peak = max(float(np.abs(a).max()), float(np.abs(b).max()))
        atol = atol_frac * peak

    diff = np.abs(a - b)
    i = int(np.argmax(diff))
    assert np.allclose(a, b, rtol=rtol, atol=atol), (
        "{}: GPU vs CPU mismatch, max abs error {:.3g} (atol={:.3g}, rtol={:.1%}) "
        "at index {} (gpu={}, cpu={})".format(label, float(diff[i]), atol, rtol, i, a[i], b[i])
    )


def load_probe_file(path):
    """Read a raw openEMS probe/port output file: header '% t/s <cols...>' then rows.

    Returns (col_names, data) where data has shape (N, ncols) including the
    time column. Reused instead of openEMS.ports.UI_data because that helper
    only keeps a single value column (fine for voltage/current, not for
    multi-component E/H field probes).
    """
    comments = []
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("%"):
                comments.append(line[1:].strip())
            else:
                s = line.strip()
                if s:
                    rows.append(s.split())
    col_names = comments[-1].split() if comments else None
    return col_names, np.array(rows, dtype=np.float64)
