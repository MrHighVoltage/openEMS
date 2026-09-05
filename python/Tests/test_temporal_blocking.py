# -*- coding: utf-8 -*-
"""Regression tests for Engine_AVX2_Multithread's trapezoidal temporal blocking.

The schedule in FDTD/engine_avx2_multithread.cpp (OPENEMS_AVX2_TEMPORAL_BLOCK)
advances a cache-resident tile of x-lines through k timesteps before moving
on, instead of streaming the whole grid once per timestep. It is claimed to
be bit-identical to the flat sweep -- but only for extensions that implement
the slab hooks in Engine_Extension (SupportsSlabApply() / *Slab()); anything
else makes ConfigureTemporalBlocking() refuse to engage and fall back to the
flat sweep, printing:

    AVX2 temporal blocking: disabled, extension '<name>' cannot be applied
    per x-range.

instead of, when it does engage:

    AVX2 temporal blocking ACTIVE (prototype): k=..., tile width ... x-lines,
    ... MB per tile

A comparison test that only checks "flat run == blocked run" would pass
*vacuously* whenever blocking silently declines to engage -- of course a
sweep is identical to itself. Every case here therefore greps the blocked
run's stdout for the ACTIVE line before trusting the field comparison; if
blocking did not engage, the test fails with a message naming whichever
extension the engine refused (or reports some other reason, e.g. the grid
still fitting cache). That failure is itself the useful signal while slab
support is still being added extension-by-extension: it tells you which one
is still missing, rather than reporting a green, meaningless test.

Simulations run in a subprocess so the engine's stdout -- a C++-level side
effect of ConfigureTemporalBlocking(), not something the Python bindings
surface -- can be captured cleanly, and so OPENEMS_AVX2_TEMPORAL_BLOCK can be
set for one run and not the other without disturbing this process's own
environment.

The grid is 64 x-lines split into four 16-line tiles (OPENEMS_AVX2_TEMPORAL_
BLOCK=4:16, i.e. k=4, W=16, satisfying the engine's hard W>=2k constraint),
small enough to finish in well under a second. Left alone, the two
performance guards in ConfigureTemporalBlocking (grid too small relative to
cache, not enough x-lines for two tiles) would refuse to engage on a grid
this size -- that is the right call for real runs, but it would make this
test file impossible to run in CI in a reasonable time. OPENEMS_AVX2_
TEMPORAL_BLOCK_FORCE=1 lifts exactly those two guards (not the extension
-support guard, which is what is actually under test here), so the schedule
can be exercised, correctly, on a CI-sized grid. Every object placed in the
grid straddles the tile boundary at x=48 on purpose, so a wedge actually has
to reconcile something other than vacuum.
"""

import os
import sys
import shutil
import subprocess
import tempfile
import textwrap
import unittest

import numpy as np


def _bootstrap_local_openems_runtime():
    """Prepend the in-tree build to LD_LIBRARY_PATH, re-exec'ing if needed.

    Only this (parent) process needs to see libopenEMS.so directly -- it
    never imports CSXCAD/openEMS itself, it only launches subprocesses that
    do. But the subprocesses inherit this process's environment, so fixing
    LD_LIBRARY_PATH here is what lets them find the freshly built library
    without every helper below having to know about `build/`.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    build_dir = os.path.join(repo_root, "build")

    if not os.path.exists(os.path.join(build_dir, "libopenEMS.so")):
        return

    paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    if paths and os.path.abspath(paths[0]) == os.path.abspath(build_dir):
        return

    paths = [p for p in paths if os.path.abspath(p) != os.path.abspath(build_dir)]
    os.environ["LD_LIBRARY_PATH"] = ":".join([build_dir] + paths)

    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if spec is not None:
        module = spec.parent or spec.name
        argv = [sys.executable, "-m", module] + sys.argv[1:]
    else:
        argv = [sys.executable] + sys.argv
    os.execvpe(argv[0], argv, os.environ)


_bootstrap_local_openems_runtime()

import h5py  # noqa: E402


# ---------------------------------------------------------------------------
# Grid shared by every case: 64 x-lines, four 16-line tiles.
# ---------------------------------------------------------------------------
NX, NY, NZ = 64, 32, 32
TIMESTEPS = 120
BLOCK_ENV = "4:16"          # k=4, W=16 -> 4 tiles across the grid
THREADS = 2                 # fixed, so a comparison is not also a thread-count change
EXC_X = 20                  # excitation plane, inside tile 1 ([16,32))

# Every extra object straddles the tile boundary at x=48 so that a wedge has
# to reconcile it, not just vacuum either side.
OBJ_X_LO, OBJ_X_HI = 38, 54
LUMPED_X_LO, LUMPED_X_HI = 46, 50


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

{extra_setup}

exc = CSX.AddExcitation("exc", exc_type=0, exc_val=[0, 0, 1])
exc.AddBox([xl[{exc_x}], yl[10], zl[10]], [xl[{exc_x}] + 1, yl[22], zl[22]])

dumpE = CSX.AddDump("Et", dump_type=0, dump_mode=0, file_type=1)
dumpE.AddBox([xl[0], yl[0], zl[0]], [xl[-1], yl[-1], zl[-1]])
dumpH = CSX.AddDump("Ht", dump_type=1, dump_mode=0, file_type=1)
dumpH.AddBox([xl[0], yl[0], zl[0]], [xl[-1], yl[-1], zl[-1]])

FDTD.Run({sim_path!r}, cleanup=True, engine="avx2-multithreaded", numThreads={threads})
"""


def _run(tag, boundary, extra_setup, blocked):
    """Run one small avx2-multithreaded simulation in a subprocess.

    Returns (stdout, E_time_series, H_time_series). `blocked` selects whether
    OPENEMS_AVX2_TEMPORAL_BLOCK[_FORCE] are set for this particular run --
    the two runs being compared must differ in *only* that.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sim_path = os.path.join(tempfile.gettempdir(), "tblk_%s" % tag)
    shutil.rmtree(sim_path, ignore_errors=True)

    script = _SCRIPT_TEMPLATE.format(
        timesteps=TIMESTEPS,
        boundary=boundary,
        nx=NX, ny=NY, nz=NZ,
        extra_setup=textwrap.dedent(extra_setup).strip(),
        exc_x=EXC_X,
        threads=THREADS,
        sim_path=sim_path,
    )
    script_path = os.path.join(tempfile.gettempdir(), "tblk_%s.py" % tag)
    with open(script_path, "w") as f:
        f.write(script)

    env = dict(os.environ)
    env.pop("OPENEMS_AVX2_TEMPORAL_BLOCK", None)
    env.pop("OPENEMS_AVX2_TEMPORAL_BLOCK_FORCE", None)
    if blocked:
        env["OPENEMS_AVX2_TEMPORAL_BLOCK"] = BLOCK_ENV
        env["OPENEMS_AVX2_TEMPORAL_BLOCK_FORCE"] = "1"

    proc = subprocess.run(
        [sys.executable, script_path],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    try:
        if proc.returncode != 0:
            raise RuntimeError(
                "simulation subprocess failed (tag=%s, rc=%s):\n%s\n%s"
                % (tag, proc.returncode, proc.stdout[-4000:], proc.stderr[-4000:])
            )

        def _load(name):
            with h5py.File(os.path.join(sim_path, name), "r") as f:
                td = f["FieldData/TD"]
                return np.array([np.array(td[k]) for k in sorted(td.keys())])

        E = _load("Et.h5")
        H = _load("Ht.h5")
    finally:
        shutil.rmtree(sim_path, ignore_errors=True)
        os.remove(script_path)

    return proc.stdout, E, H


def _relevant_lines(stdout):
    """The one or two lines out of the whole run's stdout that actually say
    what ConfigureTemporalBlocking() decided -- for failure messages."""
    lines = [l for l in stdout.splitlines() if "temporal blocking" in l]
    return "\n".join(lines) if lines else "(no 'temporal blocking' line in stdout at all)"


class TemporalBlockingCaseMixin:
    """Shared machinery: run flat vs. blocked, assert ACTIVE, compare fields.

    Not a TestCase itself; each concrete extension's test below supplies
    `boundary` and `extra_setup` and calls `_compare()`.
    """

    def _compare(self, tag, boundary, extra_setup):
        flat_out, flat_E, flat_H = _run(tag + "_flat", boundary, extra_setup, blocked=False)
        blk_out, blk_E, blk_H = _run(tag + "_blk", boundary, extra_setup, blocked=True)

        self.assertNotIn(
            "AVX2 temporal blocking ACTIVE", flat_out,
            "the control run (no OPENEMS_AVX2_TEMPORAL_BLOCK set) engaged "
            "blocking anyway -- the env var is not gating it",
        )
        self.assertGreater(
            np.max(np.abs(flat_E)), 0.0,
            "flat run produced an all-zero field -- excitation never reached the dump",
        )

        # This is the assertion that keeps the test from passing vacuously:
        # if the extension under test cannot be slabbed, blocking silently
        # falls back to the flat sweep and the field comparison below would
        # trivially succeed without having exercised the blocked schedule at
        # all. Fail loudly instead, naming whatever the engine actually said.
        self.assertIn(
            "AVX2 temporal blocking ACTIVE", blk_out,
            "temporal blocking did not engage for '%s'; engine said:\n%s"
            % (tag, _relevant_lines(blk_out)),
        )

        self.assertEqual(
            flat_E.shape, blk_E.shape,
            "flat and blocked E dumps have different shapes (%s vs %s) -- "
            "one of the runs went unstable or dumped a different number of steps"
            % (flat_E.shape, blk_E.shape),
        )
        self.assertEqual(flat_H.shape, blk_H.shape, "flat and blocked H dumps have different shapes")

        self.assertTrue(
            np.array_equal(flat_E, blk_E),
            "AVX2 temporal blocking is not bit-identical to the flat sweep "
            "for E in '%s': max|d| = %.3e"
            % (tag, float(np.max(np.abs(blk_E.astype(np.float64) - flat_E.astype(np.float64))))),
        )
        self.assertTrue(
            np.array_equal(flat_H, blk_H),
            "AVX2 temporal blocking is not bit-identical to the flat sweep "
            "for H in '%s': max|d| = %.3e"
            % (tag, float(np.max(np.abs(blk_H.astype(np.float64) - flat_H.astype(np.float64))))),
        )


class Test1_PEC_Control(TemporalBlockingCaseMixin, unittest.TestCase):
    """Control: PEC boundaries only, no extension beyond the excitation.

    Only the excitation extension is present, and it already supports slab
    apply, so this is the one case expected to engage and match today. If
    this one fails, something is wrong with the harness or the schedule
    itself, not with a specific extension's slab support.
    """

    def test_pec_only(self):
        self._compare(
            "pec",
            [0, 0, 0, 0, 0, 0],
            """
            metal = CSX.AddMetal("block")
            metal.AddBox([xl[%d], yl[10], zl[10]], [xl[%d], yl[22], zl[22]], priority=10)
            """ % (OBJ_X_LO, OBJ_X_HI),
        )


class Test2_UPML(TemporalBlockingCaseMixin, unittest.TestCase):
    """Uniaxial PML absorbing boundary on all six sides (PML_8)."""

    def test_pml(self):
        self._compare(
            "pml",
            ["PML_8"] * 6,
            """
            sub = CSX.AddMaterial("sub", epsilon=2.5)
            sub.AddBox([xl[%d], yl[10], zl[10]], [xl[%d], yl[22], zl[22]], priority=5)
            """ % (OBJ_X_LO, OBJ_X_HI),
        )


class Test3_MurABC(TemporalBlockingCaseMixin, unittest.TestCase):
    """Simple Mur absorbing boundary condition on all six sides."""

    def test_mur(self):
        self._compare(
            "mur",
            ["MUR"] * 6,
            """
            sub = CSX.AddMaterial("sub", epsilon=2.5)
            sub.AddBox([xl[%d], yl[10], zl[10]], [xl[%d], yl[22], zl[22]], priority=5)
            """ % (OBJ_X_LO, OBJ_X_HI),
        )


class Test4_DispersiveDrude(TemporalBlockingCaseMixin, unittest.TestCase):
    """A Drude/plasma dispersive material (CSPropLorentzMaterial, order 1,
    pole frequency left at zero so only the plasma/damping terms act)."""

    def test_drude(self):
        self._compare(
            "drude",
            [0, 0, 0, 0, 0, 0],
            """
            from CSXCAD.CSProperties import CSPropLorentzMaterial
            mat = CSPropLorentzMaterial(CSX.GetParameterSet(), epsilon=1.0, order=1)
            CSX.AddProperty(mat)
            mat.SetName("drude")
            mat.SetDispersiveMaterialProperty(0, eps_plasma=2 * np.pi * 5e9, eps_relax=2e-11)
            mat.AddBox([xl[%d], yl[10], zl[10]], [xl[%d], yl[22], zl[22]], priority=5)
            """ % (OBJ_X_LO, OBJ_X_HI),
        )


class Test5_LumpedRLC(TemporalBlockingCaseMixin, unittest.TestCase):
    """A single parallel-RLC lumped element (see LumpedRLC.py for the full
    port-based validation of this extension; here it is only load-bearing
    geometry, not something whose impedance is checked)."""

    def test_lumped_rlc(self):
        self._compare(
            "lumped",
            [0, 0, 0, 0, 0, 0],
            """
            lrc = CSX.AddLumpedElement("rlc", ny="z", caps=True, R=75.0, L=5e-9, C=2e-12, LEtype=0)
            lrc.AddBox([xl[%d], yl[14], zl[10]], [xl[%d], yl[18], zl[14]], priority=10)
            """ % (LUMPED_X_LO, LUMPED_X_HI),
        )


class Test6_Combined(TemporalBlockingCaseMixin, unittest.TestCase):
    """PML boundary + a lumped element + an E-field probe together.

    ConfigureTemporalBlocking() checks every active extension and stops at
    the first one that refuses, so this also exercises that the loop over
    m_Eng_exts does not, say, stop checking after the first extension that
    *does* support slab apply.
    """

    def test_pml_plus_lumped_plus_probe(self):
        self._compare(
            "combo",
            ["PML_8"] * 6,
            """
            lrc = CSX.AddLumpedElement("rlc", ny="z", caps=True, R=75.0, L=5e-9, C=2e-12, LEtype=0)
            lrc.AddBox([xl[%d], yl[14], zl[10]], [xl[%d], yl[18], zl[14]], priority=10)
            probe = CSX.AddProbe("p1", p_type=2)
            probe.AddBox([xl[49], yl[16], zl[12]], [xl[49], yl[16], zl[12]])
            """ % (LUMPED_X_LO, LUMPED_X_HI),
        )


class Test7_AbsorbingBC(TemporalBlockingCaseMixin, unittest.TestCase):
    """A local absorbing-BC sheet (Engine_Ext_Absorbing_BC), as used by
    MSL_With_Local_Absorbers.py.

    Unlike the global MUR boundary this sits on an interior plane, so it is
    the case where a boundary-shaped extension is *not* protected by the
    trapezoid holding the domain edge. The sheet is normal to z and spans the
    full x range, so it is the pattern-C path -- the axis restriction, not
    the "is the plane in this slab" test -- that has to be right.
    """

    def test_local_absorber(self):
        self._compare(
            "abc",
            [0, 0, 0, 0, 0, 0],
            """
            from CSXCAD.CSProperties import ABCtype
            ab = CSX.AddAbsorbingBC("abs1", NormalSignPositive=False,
                                    AbsorbingBoundaryType=ABCtype.MUR_1ST)
            ab.AddBox([xl[2], yl[2], zl[26]], [xl[-3], yl[-3], zl[26]], priority=20)
            """,
        )


class Test8_TFSF(TemporalBlockingCaseMixin, unittest.TestCase):
    """A total-field/scattered-field plane-wave box (Engine_Ext_TFSF).

    The box straddles the tile boundary at x=48, so its two x-normal faces
    land in different tiles and its four x-spanning faces are cut by every
    tile. openems.cpp adds Operator_Ext_TFSF to every Cartesian run, but it
    only builds an engine extension when an exc_type=10 excitation with
    exactly one box primitive is present -- which is what this sets up.
    """

    def test_plane_wave(self):
        self._compare(
            "tfsf",
            [0, 0, 0, 0, 0, 0],
            """
            pw = CSX.AddExcitation("planewave", exc_type=10, exc_val=[0, 0, 1])
            pw.SetPropagationDir([1, 0, 0])
            pw.AddBox([xl[%d], yl[6], zl[6]], [xl[%d], yl[26], zl[26]])
            """ % (OBJ_X_LO, OBJ_X_HI),
        )


class Test9_ScheduleSweep(TemporalBlockingCaseMixin, unittest.TestCase):
    """The same simulation over several (k, W) and thread counts.

    One (k, W) proves the hooks are wired up; it does not prove the geometry
    is right. Different tile widths put the tile boundaries -- and so the
    wedges -- in different places relative to the PML, the lumped element and
    the absorber, and different thread counts change how each slab is split
    across workers. A slab hook that writes one cell outside its range, or
    partitions its work so two threads share a cell, shows up here and in
    nothing above.
    """

    SETUP = """
    from CSXCAD.CSProperties import ABCtype
    lrc = CSX.AddLumpedElement("rlc", ny="z", caps=True, R=75.0, L=5e-9, C=2e-12, LEtype=0)
    lrc.AddBox([xl[46], yl[14], zl[10]], [xl[50], yl[18], zl[14]], priority=10)
    ab = CSX.AddAbsorbingBC("abs1", NormalSignPositive=False,
                            AbsorbingBoundaryType=ABCtype.MUR_1ST)
    ab.AddBox([xl[2], yl[2], zl[26]], [xl[-3], yl[-3], zl[26]], priority=20)
    """

    def test_sweep(self):
        global BLOCK_ENV, THREADS
        flat_out, flat_E, flat_H = _run("sweep_flat", ["PML_8"] * 6, self.SETUP, blocked=False)
        self.assertGreater(np.max(np.abs(flat_E)), 0.0, "flat run produced an all-zero field")

        saved = (BLOCK_ENV, THREADS)
        try:
            for kw in ("2:8", "4:16", "3:13", "6:20"):
                for threads in (1, 3, 4):
                    BLOCK_ENV, THREADS = kw, threads
                    tag = "sweep_%s_%d" % (kw.replace(":", "_"), threads)
                    out, E, H = _run(tag, ["PML_8"] * 6, self.SETUP, blocked=True)
                    self.assertIn(
                        "AVX2 temporal blocking ACTIVE", out,
                        "blocking did not engage at k:W=%s, %d threads; engine said:\n%s"
                        % (kw, threads, _relevant_lines(out)),
                    )
                    self.assertTrue(
                        np.array_equal(flat_E, E) and np.array_equal(flat_H, H),
                        "k:W=%s with %d threads is not bit-identical to the flat sweep"
                        % (kw, threads),
                    )
        finally:
            BLOCK_ENV, THREADS = saved


if __name__ == "__main__":
    unittest.main()
