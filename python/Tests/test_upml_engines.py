# -*- coding: utf-8 -*-
"""Regression tests for the UPML engine extension.

Engine_Ext_UPML has two implementations of the same four update passes: a
scalar one that addresses single cells through Engine::GetVolt()/SetVolt(),
and an AVX2 one (engine_ext_upml_avx2.cpp) that updates whole f8vectors for
the PML boxes spanning the full z range. The vector path deliberately avoids
FMA so that it stays *bit-identical* to the scalar path it replaces, and
OPENEMS_UPML_NO_AVX2=1 forces the scalar path so the two can be compared in
one binary.

Nothing else in this directory runs a PML through more than one engine, so a
layout or lane-mapping mistake in the AVX2 path would otherwise go unnoticed:
the shipped examples only check that a simulation converges, which it still
does when a fraction of the PML cells are updated with the wrong coefficients.

Grid extents here are deliberately not multiples of 8 so the padding lanes
(8 * numVectors > numLines[2]) are exercised too.
"""

import os
import sys
import tempfile
import shutil
import unittest

import numpy as np


def _bootstrap_local_openems_runtime():
    """Prepend the in-tree build to LD_LIBRARY_PATH, re-exec'ing if needed."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    python_dir = os.path.join(repo_root, "python")
    build_dir = os.path.join(repo_root, "build")

    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)

    if not os.path.exists(os.path.join(build_dir, "libopenEMS.so")):
        return

    paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    if paths and os.path.abspath(paths[0]) == os.path.abspath(build_dir):
        return

    paths = [p for p in paths if os.path.abspath(p) != os.path.abspath(build_dir)]
    os.environ["LD_LIBRARY_PATH"] = ":".join([build_dir] + paths)

    # Rebuild the original command line. Under "python -m <mod>" argv[0] is a
    # descriptive string rather than a runnable path, so recover the module
    # name from __main__'s spec and re-exec with -m instead.
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if spec is not None:
        module = spec.parent or spec.name
        argv = [sys.executable, "-m", module] + sys.argv[1:]
    else:
        argv = [sys.executable] + sys.argv
    os.execvpe(argv[0], argv, os.environ)


_bootstrap_local_openems_runtime()

from CSXCAD import ContinuousStructure  # noqa: E402
from openEMS import openEMS  # noqa: E402

import h5py  # noqa: E402


def _graded(n, d0, ratio):
    return np.concatenate([[0.0], np.cumsum(d0 * ratio ** np.arange(n - 1))])


def _run(engine, tag, graded=False, timesteps=180):
    """Run a small PML_8 box and return the dumped E-field time series."""
    sim = os.path.join(tempfile.gettempdir(), "upml_eng_%s" % tag)
    shutil.rmtree(sim, ignore_errors=True)

    FDTD = openEMS(NrTS=timesteps, EndCriteria=0)
    FDTD.SetGaussExcite(3e9, 3e9)
    FDTD.SetBoundaryCond(["PML_8"] * 6)

    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(1e-3)
    if graded:
        mesh.SetLines("x", _graded(37, 1.00, 1.03))
        mesh.SetLines("y", _graded(34, 0.90, 1.04))
        mesh.SetLines("z", _graded(41, 1.10, 1.02))
    else:
        mesh.SetLines("x", np.arange(37, dtype=float) * 1.0)
        mesh.SetLines("y", np.arange(34, dtype=float) * 0.9)
        mesh.SetLines("z", np.arange(41, dtype=float) * 1.1)
    xl, yl, zl = mesh.GetLines("x"), mesh.GetLines("y"), mesh.GetLines("z")

    sub = CSX.AddMaterial("sub", epsilon=4.2, kappa=0.02)
    sub.AddBox([xl[12], yl[12], zl[14]], [xl[24], yl[22], zl[27]], priority=5)
    exc = CSX.AddExcitation("e", exc_type=0, exc_val=[0, 0, 1])
    exc.AddBox([xl[18], yl[18], zl[16]], [xl[19], yl[19], zl[20]])
    dump = CSX.AddDump("Et", dump_type=0, dump_mode=0, file_type=1)
    dump.AddBox([xl[0], yl[0], zl[0]], [xl[-1], yl[-1], zl[-1]])

    FDTD.Run(sim, cleanup=True, engine=engine)
    with h5py.File(os.path.join(sim, "Et.h5"), "r") as f:
        td = f["FieldData/TD"]
        out = np.array([np.array(td[k]) for k in sorted(td.keys())])
    shutil.rmtree(sim, ignore_errors=True)
    return out


class TestUPMLVectorPath(unittest.TestCase):
    """The AVX2 UPML kernels must reproduce the scalar ones exactly."""

    def _compare_paths(self, tag, graded):
        prev = os.environ.get("OPENEMS_UPML_NO_AVX2")
        try:
            os.environ["OPENEMS_UPML_NO_AVX2"] = "1"
            scalar = _run("avx2", tag + "_scalar", graded=graded)
            del os.environ["OPENEMS_UPML_NO_AVX2"]
            vector = _run("avx2", tag + "_vector", graded=graded)
        finally:
            os.environ.pop("OPENEMS_UPML_NO_AVX2", None)
            if prev is not None:
                os.environ["OPENEMS_UPML_NO_AVX2"] = prev

        self.assertGreater(np.max(np.abs(scalar)), 0.0, "simulation produced no field")
        self.assertEqual(
            scalar.shape, vector.shape,
            "the two paths produced different numbers of dumps (%s vs %s) -- "
            "one of the runs went unstable" % (scalar.shape, vector.shape),
        )
        self.assertTrue(
            np.array_equal(scalar, vector),
            "AVX2 UPML path is not bit-identical to the scalar path: "
            "max|d| = %.3e" % float(np.max(np.abs(vector - scalar))),
        )

    def test_uniform_mesh(self):
        self._compare_paths("uni", graded=False)

    def test_graded_mesh(self):
        self._compare_paths("grd", graded=True)


class TestUPMLCrossEngine(unittest.TestCase):
    """Every engine must agree on a PML-bounded, graded, inhomogeneous grid."""

    @classmethod
    def setUpClass(cls):
        cls.ref = _run("sse", "ref", graded=True)
        cls.peak = float(np.max(np.abs(cls.ref)))

    def _check(self, engine, tol):
        got = _run(engine, engine.replace("-", "_"), graded=True)
        self.assertEqual(
            got.shape, self.ref.shape,
            "%s produced %s dumps, sse produced %s -- one run went unstable"
            % (engine, got.shape, self.ref.shape),
        )
        diff = float(np.max(np.abs(got - self.ref)))
        self.assertLessEqual(
            diff, self.peak * tol,
            "%s differs from sse by rel=%.3e (tol %.1e)" % (engine, diff / self.peak, tol),
        )

    def test_basic(self):
        self._check("basic", 0.0)

    def test_sse_compressed(self):
        self._check("sse-compressed", 0.0)

    def test_multithreaded(self):
        self._check("multithreaded", 0.0)

    # AVX2 uses FMA in the Yee kernel, so it rounds differently from SSE.
    def test_avx2(self):
        self._check("avx2", 2e-5)

    def test_avx2_multithreaded(self):
        self._check("avx2-multithreaded", 2e-5)


if __name__ == "__main__":
    unittest.main()
