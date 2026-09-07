# -*- coding: utf-8 -*-
"""Regression tests for Engine_Vulkan's trapezoidal temporal blocking.

The schedule in FDTD/engine_vulkan.cpp (OPENEMS_GPU_TEMPORAL_BLOCK) advances
one x-slab through k timesteps before moving on, so that the slab -- not the
whole grid -- is the working set the GPU's last-level cache has to hold. It is
a transcription of the geometry Engine_AVX2_Multithread already runs, and the
claim under test is the same one: bit-identical to the flat sweep.

The trap this file exists to avoid is the same as its AVX2 counterpart's. A
test that only checks "flat run == blocked run" passes *vacuously* whenever
blocking declines to engage -- a sweep is of course identical to itself, and
ConfigureTemporalBlocking() declines for several good reasons (an extension
without a slab path, a grid that already fits the tile target, too few
x-lines). Every case here therefore greps the blocked run's output for the
line that says blocking actually engaged, before trusting any comparison.

The grid is 64 x-lines in four 16-line tiles (k=4, W=16, satisfying the hard
W>=2k constraint). OPENEMS_GPU_TEMPORAL_BLOCK_MB=1 shrinks the tile target so
the "grid already fits cache" guard -- the right call for real runs -- does
not refuse a CI-sized grid.
"""

import os
import sys
import shutil
import subprocess
import tempfile
import unittest

import numpy as np


def _bootstrap_local_openems_runtime():
    """Prepend the in-tree build to LD_LIBRARY_PATH, re-exec'ing if needed.

    This process never imports openEMS itself -- it only launches subprocesses
    that do -- but they inherit this environment, which is what lets them find
    the freshly built library instead of the installed one.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    build_dir = os.path.join(repo_root, "build")
    paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    if paths and os.path.abspath(paths[0]) == os.path.abspath(build_dir):
        return
    paths = [p for p in paths if os.path.abspath(p) != os.path.abspath(build_dir)]
    os.environ["LD_LIBRARY_PATH"] = ":".join([build_dir] + paths)
    os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)


_bootstrap_local_openems_runtime()

import h5py  # noqa: E402

NX, NY, NZ = 64, 32, 32
TIMESTEPS = 120
BLOCK_ENV = "4:16"          # k=4, W=16 -> four tiles across the grid
EXC_X = 20                  # excitation plane, inside tile 1 ([16,32))
GPU_INDEX = os.environ.get("OPENEMS_TEST_GPU_INDEX", "0")

ACTIVE_MARK = "trapezoidal temporal blocking active"

_SCRIPT_TEMPLATE = """
import numpy as np
from CSXCAD import ContinuousStructure
from openEMS import openEMS

FDTD = openEMS(NrTS={timesteps}, EndCriteria=0)
FDTD.SetGaussExcite(5e9, 5e9)
FDTD.SetBoundaryCond({boundary!r})

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(1e-3)
mesh.SetLines("x", np.arange({nx}, dtype=float))
mesh.SetLines("y", np.arange({ny}, dtype=float))
mesh.SetLines("z", np.arange({nz}, dtype=float))
xl, yl, zl = mesh.GetLines("x"), mesh.GetLines("y"), mesh.GetLines("z")

exc = CSX.AddExcitation("exc", exc_type=0, exc_val=[0, 0, 1])
exc.AddBox([xl[{exc_x}], yl[10], zl[10]], [xl[{exc_x}] + 1, yl[22], zl[22]])

dumpE = CSX.AddDump("Et", dump_type=0, dump_mode=0, file_type=1)
dumpE.AddBox([xl[0], yl[0], zl[0]], [xl[-1], yl[-1], zl[-1]])
dumpH = CSX.AddDump("Ht", dump_type=1, dump_mode=0, file_type=1)
dumpH.AddBox([xl[0], yl[0], zl[0]], [xl[-1], yl[-1], zl[-1]])

FDTD.Run({sim_path!r}, cleanup=True, engine="gpu")
"""


def _run(tag, boundary, block_env):
    """Run one small GPU simulation in a subprocess.

    Returns (output, E_time_series, H_time_series). `block_env` is the value
    of OPENEMS_GPU_TEMPORAL_BLOCK for this run, or None for the flat sweep;
    the two runs being compared must differ in *only* that.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sim_path = os.path.join(tempfile.gettempdir(), "gputblk_%s" % tag)
    shutil.rmtree(sim_path, ignore_errors=True)

    script = _SCRIPT_TEMPLATE.format(
        timesteps=TIMESTEPS, boundary=boundary, nx=NX, ny=NY, nz=NZ,
        exc_x=EXC_X, sim_path=sim_path,
    )
    script_path = os.path.join(tempfile.gettempdir(), "gputblk_%s.py" % tag)
    with open(script_path, "w") as f:
        f.write(script)

    env = dict(os.environ)
    env["OPENEMS_GPU_INDEX"] = GPU_INDEX
    env.pop("OPENEMS_GPU_TEMPORAL_BLOCK", None)
    env.pop("OPENEMS_GPU_TEMPORAL_BLOCK_MB", None)
    if block_env:
        env["OPENEMS_GPU_TEMPORAL_BLOCK"] = block_env
        env["OPENEMS_GPU_TEMPORAL_BLOCK_MB"] = "1"

    proc = subprocess.run([sys.executable, script_path], cwd=repo_root, env=env,
                          capture_output=True, text=True, timeout=300)
    try:
        if proc.returncode != 0:
            raise RuntimeError("simulation subprocess failed (tag=%s, rc=%s):\n%s\n%s"
                               % (tag, proc.returncode, proc.stdout[-4000:], proc.stderr[-4000:]))

        def _load(name):
            with h5py.File(os.path.join(sim_path, name), "r") as f:
                td = f["FieldData/TD"]
                return np.array([np.array(td[k]) for k in sorted(td.keys())])

        E, H = _load("Et.h5"), _load("Ht.h5")
    finally:
        shutil.rmtree(sim_path, ignore_errors=True)
        os.remove(script_path)

    return proc.stdout + proc.stderr, E, H


def _relevant_lines(output):
    """The lines that say what ConfigureTemporalBlocking() decided."""
    lines = [l for l in output.splitlines() if "temporal blocking" in l]
    return "\n".join(lines) if lines else "(no 'temporal blocking' line in the output at all)"


class GPUTemporalBlockingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.flat_out, cls.flat_E, cls.flat_H = _run("flat", ["PEC"] * 6, None)

    def test_flat_reference_is_not_vacuous(self):
        """A comparison against an all-zero field would pass no matter what."""
        self.assertGreater(
            float(np.max(np.abs(self.flat_E))), 0.0,
            "flat run produced an all-zero E field -- the excitation never reached the dump")

    def test_blocking_engages(self):
        out, _, _ = _run("engage", ["PEC"] * 6, BLOCK_ENV)
        self.assertIn(ACTIVE_MARK, out,
                      "temporal blocking did not engage, so any field comparison below "
                      "would pass vacuously. The engine said:\n%s" % _relevant_lines(out))

    def test_pec_bit_identical(self):
        out, E, H = _run("pec", ["PEC"] * 6, BLOCK_ENV)
        self.assertIn(ACTIVE_MARK, out, _relevant_lines(out))
        self.assertEqual(self.flat_E.shape, E.shape,
                         "flat and blocked E dumps have different shapes -- one run went "
                         "unstable or dumped a different number of steps")
        self.assertTrue(np.array_equal(self.flat_E, E),
                        "blocked E differs from flat: max |delta| = %g"
                        % float(np.max(np.abs(self.flat_E - E))))
        self.assertTrue(np.array_equal(self.flat_H, H),
                        "blocked H differs from flat: max |delta| = %g"
                        % float(np.max(np.abs(self.flat_H - H))))

    def test_schedule_sweep(self):
        """Several (k, W) pairs, including one where W does not divide NX.

        A tile width that divides the domain evenly is the easy case; the
        wedges only have to reconcile a boundary the tiling put there. 4:13
        leaves a short tail, which is what exercises the fold in
        ConfigureTemporalBlocking().
        """
        for env in ("2:8", "4:16", "3:12", "4:13", "6:20"):
            with self.subTest(schedule=env):
                out, E, H = _run("sweep_%s" % env.replace(":", "_"), ["PEC"] * 6, env)
                self.assertIn(ACTIVE_MARK, out, _relevant_lines(out))
                self.assertTrue(np.array_equal(self.flat_E, E) and np.array_equal(self.flat_H, H),
                                "schedule %s is not bit-identical to the flat sweep" % env)

    def test_refuses_unsupported_extension_and_stays_correct(self):
        """PML has no slab path yet: blocking must decline, not silently differ."""
        flat_out, flat_E, flat_H = _run("pml_flat", ["PML_8"] * 6, None)
        out, E, H = _run("pml_blocked", ["PML_8"] * 6, BLOCK_ENV)
        self.assertNotIn(ACTIVE_MARK, out,
                         "blocking engaged with UPML active, which has no slab path")
        self.assertIn("temporal blocking disabled", out, _relevant_lines(out))
        self.assertTrue(np.array_equal(flat_E, E) and np.array_equal(flat_H, H),
                        "the fallback path changed results")


if __name__ == "__main__":
    unittest.main(verbosity=2)
