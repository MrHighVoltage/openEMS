#!/usr/bin/env python3
"""Check that the Vulkan engine's blocked schedule is bit-identical to its flat sweep.

    gpu_tblock_identity.py --bin BUNDLE/fork/openEMS [--gpu 0000:03:00.0]
                           [--nts 600] [--ref OLD/openEMS] [CASE...]

Every case runs --engine=gpu flat (twice: the flat path must agree with itself)
and with several blocked schedules (OPENEMS_GPU_TEMPORAL_BLOCK=k:W). Every
E and H dump of every run, and every probe file, must equal the first flat
run's bit for bit. A blocked run that did not engage blocking fails, unless
the case is one blocking is expected to refuse; then the refusal message must
appear and the fields must still match.

Without CASE arguments the built-in cases run. Each puts one GPU extension
across the tile edges of the schedules below (x = 12, 20, 40, 48, ... on a
100-line grid):

    mur_rlc     Mur on all faces; lumped R, series RLC and parallel RC
                elements (one a port with its excitation), voltage and current
                probes, a lossy dielectric block
    tfsf        a TF/SF plane-wave box with Mur on all faces
    pml_mur     PML_8 on y and z, Mur on x, a Lorentz material block, and the
                same with the fused UPML kernel off
    local_abc   a local absorbing sheet normal to y
    local_abc_sa  super-absorbing sheets, one normal to x on the tile edges
                (the tile edges must move off it), one normal to z

A CASE argument is a CSXCAD-only geometry (gets Mur on all faces) or a
complete openEMS XML. The excitation of every case is replaced by a sine whose
Nyquist interval puts dumps and probes every --interval timesteps (the
timestep is read from a 2-step run first), so the fields are non-zero from the
start and the blocks are not clamped short. It is a custom function, not
openEMS's sinusoidal type: a periodic signal switches on steady-state
detection, which blocking refuses. The physics is meaningless;
identity is all that is checked.

--ref adds a flat run of another build, reported but not counted: the flat
path itself may change between builds.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
	import h5py
	import numpy as np
except ImportError:  # run only, keep the dumps, compare elsewhere with --compare
	h5py = None

BLOCKED = ["4:12", "8:20", "16:40", "24:48"]
ACTIVE_MARK = "trapezoidal temporal blocking active"


# --------------------------------------------------------------------------
# Built-in cases
# --------------------------------------------------------------------------

def _box(parent, p1, p2, prio=0):
	prims = parent.find("Primitives")
	if prims is None:
		prims = ET.SubElement(parent, "Primitives")
	b = ET.SubElement(prims, "Box", Priority=str(prio))
	ET.SubElement(b, "P1", X=repr(float(p1[0])), Y=repr(float(p1[1])), Z=repr(float(p1[2])))
	ET.SubElement(b, "P2", X=repr(float(p2[0])), Y=repr(float(p2[1])), Z=repr(float(p2[2])))
	return b


def _doc(bc, nx=100, ny=40, nz=36):
	doc = ET.Element("openEMS")
	fdtd = ET.SubElement(doc, "FDTD", NumberOfTimesteps="600", endCriteria="1e-300")
	ET.SubElement(fdtd, "Excitation", Type="1", f0="1e9")
	ET.SubElement(fdtd, "BoundaryCond", **dict(zip(("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"), bc)))
	csx = ET.SubElement(doc, "ContinuousStructure", CoordSystem="0")
	grid = ET.SubElement(csx, "RectilinearGrid", DeltaUnit="0.001", CoordSystem="0")
	for ax, n in zip("XYZ", (nx, ny, nz)):
		ET.SubElement(grid, ax + "Lines").text = ",".join(str(i) for i in range(n))
	props = ET.SubElement(csx, "Properties")
	return doc, props


_ids = iter(range(1000, 100000))


def _material(props, name, eps, kappa, p1, p2, prio=10):
	m = ET.SubElement(props, "Material", ID=str(next(_ids)), Name=name, Isotropy="1")
	ET.SubElement(m, "Property", Epsilon="%g,%g,%g" % (eps, eps, eps), Kappa="%g,%g,%g" % (kappa, kappa, kappa))
	_box(m, p1, p2, prio)


def _excitation(props, name, vec, p1, p2, exc_type=0, extra=None):
	e = ET.SubElement(props, "Excitation", ID=str(next(_ids)), Name=name, Number="0", Enabled="1",
	                  Frequency="0", Delay="0", Type=str(exc_type), Excite="%g,%g,%g" % vec,
	                  **(extra or {}))
	_box(e, p1, p2)
	return e


def _lumped(props, name, ny, R, L, C, letype, p1, p2):
	e = ET.SubElement(props, "LumpedElement", ID=str(next(_ids)), Name=name, Direction=str(ny), Caps="0",
	                  R=repr(R), L=repr(L), C=repr(C), LEtype=repr(float(letype)))
	_box(e, p1, p2)


def _probe(props, name, ptype, p1, p2, normdir=-1, weight=1):
	e = ET.SubElement(props, "ProbeBox", ID=str(next(_ids)), Name=name, Number="0", Type=str(ptype),
	                  Weight=str(weight), NormDir=str(normdir), StartTime="0", StopTime="0")
	_box(e, p1, p2)


def case_mur_rlc():
	doc, props = _doc(["MUR"] * 6)
	_material(props, "block", 4.0, 0.02, (30, 5, 5), (60, 30, 25))
	# A lumped port next to the x-min face: resistor and excitation on the
	# same edges, with its voltage and current probes.
	_lumped(props, "port_R", 2, 50.0, float("nan"), float("nan"), 0, (6, 20, 14), (6, 20, 20))
	_excitation(props, "port_exc", (0, 0, -1), (6, 20, 14), (6, 20, 20))
	_probe(props, "port_ut", 0, (6, 20, 14), (6, 20, 20), weight=-1)
	_probe(props, "port_it", 1, (5.5, 19.5, 17), (6.5, 20.5, 17), normdir=2)
	# Series RLC across the tile edge at x=40, parallel RC next to the x-max face.
	_lumped(props, "series_RLC", 0, 10.0, 1e-9, 1e-12, 1, (38, 12, 10), (42, 12, 10))
	_lumped(props, "par_RC", 2, 100.0, float("nan"), 2e-12, 0, (97, 30, 8), (97, 30, 13))
	_probe(props, "far_ut", 0, (97, 30, 8), (97, 30, 13), weight=-1)
	return doc


def case_tfsf():
	doc, props = _doc(["MUR"] * 6)
	_excitation(props, "plane_wave", (0, 0, 1), (18, 8, 8), (82, 31, 27), exc_type=10,
	            extra={"PropDir": "1,0,0"})
	_material(props, "scatterer", 6.0, 0.0, (44, 15, 12), (56, 25, 22), prio=10)
	return doc


def case_pml_mur():
	doc, props = _doc(["MUR", "MUR", "PML_8", "PML_8", "PML_8", "PML_8"])
	lor = ET.SubElement(props, "LorentzMaterial", ID=str(next(_ids)), Name="lorentz", Isotropy="1")
	ET.SubElement(lor, "Property", Epsilon="2,2,2", EpsilonPlasmaFrequency_1="5e9,5e9,5e9",
	              EpsilonLorPoleFrequency_1="2e9,2e9,2e9", EpsilonRelaxTime_1="1e-9,1e-9,1e-9")
	_box(lor, (35, 12, 12), (65, 28, 24), 10)
	_excitation(props, "soft", (0, 1, 0), (20, 15, 12), (20, 25, 24))
	return doc


def _abc(props, name, sign_positive, abc_type, p1, p2):
	ab = ET.SubElement(props, "AbsorbingBC", ID=str(next(_ids)), Name=name,
	                   NormalSignPositive="1" if sign_positive else "0",
	                   AbsorbingBoundaryType=str(abc_type), PhaseVelocity=repr(299792458.0))
	_box(ab, p1, p2, 20)


def case_local_abc():
	# A first-order sheet normal to y, across every tile.
	doc, props = _doc(["PEC"] * 6)
	_abc(props, "abs", False, 1, (0, 30, 0), (99, 30, 35))
	_excitation(props, "soft", (0, 0, 1), (50, 10, 10), (50, 20, 25))
	return doc


def case_local_abc_sa():
	# Super-absorbing sheets (their current phases run too): one normal to x
	# at x=41, on or next to the tile edges of every schedule above, which
	# ConfigureTemporalBlocking() has to move; one normal to z across x.
	doc, props = _doc(["PEC"] * 6)
	_abc(props, "abs_x", True, 2, (41, 4, 4), (41, 35, 31))
	_abc(props, "abs_z", False, 2, (10, 4, 30), (90, 35, 30))
	_excitation(props, "soft", (0, 1, 0), (60, 10, 10), (60, 20, 25))
	return doc


BUILTIN = {
	"mur_rlc": (case_mur_rlc, {}),
	"tfsf": (case_tfsf, {}),
	"pml_mur": (case_pml_mur, {}),
	"pml_mur_unfused": (case_pml_mur, {"OPENEMS_GPU_DISABLE_FUSED_UPML": "1"}),
	"local_abc": (case_local_abc, {}),
	"local_abc_sa": (case_local_abc_sa, {}),
}
REFUSING = {}


def load_case(path):
	root = ET.parse(str(path)).getroot()
	if root.tag == "ContinuousStructure":
		doc = ET.Element("openEMS")
		fdtd = ET.SubElement(doc, "FDTD", NumberOfTimesteps="600", endCriteria="1e-300")
		ET.SubElement(fdtd, "Excitation", Type="1", f0="1e9")
		ET.SubElement(fdtd, "BoundaryCond", **{f: "MUR" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")})
		doc.append(root)
		return doc
	return root


# --------------------------------------------------------------------------
# Running and comparing
# --------------------------------------------------------------------------

def finalize(doc, nts, f0, subsample):
	"""The case with nts timesteps, a sine at f0, and E/H dumps over the whole domain."""
	fdtd = doc.find("FDTD")
	fdtd.set("NumberOfTimesteps", str(nts))
	fdtd.set("endCriteria", "1e-300")
	for k in ("OverSampling", "MaxTime"):
		fdtd.attrib.pop(k, None)
	# A custom sine rather than openEMS's sinusoidal type: a periodic signal
	# switches on steady-state detection, which blocking refuses by design.
	exc = fdtd.find("Excitation")
	exc.attrib.clear()
	exc.set("Type", "10")
	exc.set("f0", repr(f0))
	exc.set("Function", "sin(2*pi*%r*t)" % f0)
	csx = doc.find("ContinuousStructure")
	grid = csx.find("RectilinearGrid")
	ext = [(min(v), max(v)) for v in ([float(t) for t in grid.find(a + "Lines").text.split(",")] for a in "XYZ")]
	props = csx.find("Properties")
	for p in props.findall("DumpBox"):
		props.remove(p)
	for n, (name, dtype) in enumerate((("Et", "0"), ("Ht", "1"))):
		db = ET.SubElement(props, "DumpBox", ID=str(90000 + n), Name=name, DumpType=dtype, DumpMode="0",
		                   FileType="1", SubSampling=subsample)
		box = ET.SubElement(ET.SubElement(db, "Primitives"), "Box", Priority="0")
		ET.SubElement(box, "P1", X=repr(ext[0][0]), Y=repr(ext[1][0]), Z=repr(ext[2][0]))
		ET.SubElement(box, "P2", X=repr(ext[0][1]), Y=repr(ext[1][1]), Z=repr(ext[2][1]))
	return ET.tostring(doc, encoding="unicode")


def run(binary, xml_text, outdir, env_extra, gpu):
	if outdir.exists():
		shutil.rmtree(str(outdir))
	outdir.mkdir(parents=True)
	(outdir / "sim.xml").write_text(xml_text)
	env = {k: v for k, v in os.environ.items() if not k.startswith("OPENEMS_GPU_")}
	env.update(env_extra)
	if gpu:
		env["DRI_PRIME"] = "pci-" + re.sub(r"[:.]", "_", gpu)
		env["MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE"] = "1"
	p = subprocess.run([str(binary), "sim.xml", "--engine=gpu"], cwd=str(outdir), env=env,
	                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
	(outdir / "openEMS.log").write_text(p.stdout)
	return p.returncode, p.stdout


def datasets(path):
	out = {}
	with h5py.File(str(path), "r") as f:
		f.visititems(lambda name, obj: out.__setitem__(name, obj[()]) if isinstance(obj, h5py.Dataset) else None)
	return out


def probe_files(d):
	"""Probe outputs: every file that is not a dump, the log or the input. Comment lines dropped."""
	out = {}
	for f in sorted(d.iterdir()):
		if f.is_file() and f.suffix not in (".h5", ".xml", ".log") and f.name not in ("et", "ht"):
			out[f.name] = [l for l in f.read_text().splitlines() if not l.startswith("%")]
	return out


def compare(ref_dir, d):
	"""None if every dump and probe file is bit-identical, else the first difference."""
	for name in ("Et.h5", "Ht.h5"):
		a, b = datasets(ref_dir / name), datasets(d / name)
		if sorted(a) != sorted(b):
			return "%s: dataset sets differ (%d vs %d)" % (name, len(a), len(b))
		for k in sorted(a):
			x, y = np.asarray(a[k]), np.asarray(b[k])
			if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
				if x.shape != y.shape:
					return "%s/%s: shapes differ" % (name, k)
				diff = np.abs(x.astype(np.float64) - y.astype(np.float64))
				where = np.unravel_index(np.argmax(diff), diff.shape)
				return "%s/%s differs in %d values, max |diff| %.3g at %s" % (
					name, k, int(np.count_nonzero(diff)), diff.max(), tuple(int(i) for i in where))
	return compare_probes(ref_dir, d)


def compare_probes(ref_dir, d):
	pa, pb = probe_files(ref_dir), probe_files(d)
	if sorted(pa) != sorted(pb):
		return "probe file sets differ: %s vs %s" % (sorted(pa), sorted(pb))
	for k in pa:
		if pa[k] != pb[k]:
			return "probe %s differs" % k
	return None


def nonzero_summary(d):
	e = datasets(d / "Et.h5")
	last = e[sorted(k for k in e if k.startswith("FieldData/TD/"))[-1]]
	return "%d dumps, last %.1f %% non-zero, max |E| %.3g, %d probe files" % (
		sum(k.startswith("FieldData/TD/") for k in e), 100.0 * np.count_nonzero(last) / last.size,
		np.abs(last).max(), len(probe_files(d)))


def compare_work(work):
	"""Compare every run directory of every case in work against that case's flat run."""
	failures = 0
	for case in sorted(p for p in work.iterdir() if (p / "flat").is_dir()):
		print("=== %s" % case.name)
		ref = case / "flat"
		print("  %-16s reference  [%s]" % ("flat", nonzero_summary(ref)))
		for d in sorted(p for p in case.iterdir() if p.is_dir() and p.name not in ("flat", "dt")):
			log = (d / "openEMS.log").read_text()
			engaged = re.search(r"k=(\d+) W=(\d+) \((\d+) tiles", log) if ACTIVE_MARK in log else None
			diff = compare(ref, d)
			informational = d.name == "ref-flat"
			verdict = "identical" if diff is None else ("DIFFERS (%s)" % diff)
			if diff is not None and not informational:
				verdict = "FAIL " + verdict
				failures += 1
			if d.name.startswith("tblock") and not engaged:
				verdict = "FAIL (blocking did not engage)"
				failures += 1
			print("  %-16s %s%s" % (d.name, verdict, "  [k=%s W=%s, %s tiles]" % engaged.groups() if engaged else ""))
	print("\n%s" % ("all identical" if failures == 0 else "%d FAILURES" % failures))
	return 1 if failures else 0


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("cases", nargs="*", help="XML files; default: the built-in cases")
	ap.add_argument("--bin", type=Path, help="openEMS binary with the Vulkan engine")
	ap.add_argument("--ref", type=Path, help="flat run of another build, for information")
	ap.add_argument("--gpu", help="PCI address of the card to pin, e.g. 0000:03:00.0")
	ap.add_argument("--nts", type=int, default=600, help="timesteps per run (default: %(default)s)")
	ap.add_argument("--interval", type=int, default=96, help="timesteps between dumps (default: %(default)s)")
	ap.add_argument("--subsample", default="1,1,1", help="dump subsampling (default: %(default)s)")
	ap.add_argument("--schedules", default=",".join(BLOCKED),
	                help="k:W or k (engine-sized tile) list (default: %(default)s)")
	ap.add_argument("--work", type=Path, default=Path("gpu_tblock_identity_work"),
	                help="scratch directory (default: %(default)s)")
	ap.add_argument("--keep", action="store_true", help="keep the dumps")
	ap.add_argument("--only", help="comma-separated built-in case names")
	ap.add_argument("--compare", type=Path, metavar="WORK",
	                help="only compare the runs a previous invocation left in WORK (e.g. copied "
	                     "from a host without h5py) against each case's flat run")
	args = ap.parse_args()
	if args.compare:
		return compare_work(args.compare)
	if not args.bin:
		ap.error("--bin is required")
	args.bin = args.bin.resolve()
	if args.ref:
		args.ref = args.ref.resolve()

	builtin = {n: v for n, v in BUILTIN.items() if not args.only or n in args.only.split(",")}
	cases = [(n, f, e) for n, (f, e) in builtin.items()] if not args.cases else \
	        [(Path(c).stem, (lambda c=c: load_case(c)), {}) for c in args.cases]
	failures = 0
	for name, make, case_env in cases:
		print("=== %s" % name, flush=True)
		work = args.work / name
		# Pass 1: the timestep, from a 2-step flat run.
		rc, log = run(args.bin, finalize(make(), 2, 1e9, args.subsample), work / "dt", case_env, args.gpu)
		m = re.search(r"FDTD timestep is: *([0-9.eE+-]+) s", log)
		if rc != 0 or not m:
			print("  FAIL (could not read the timestep, rc=%d, see %s)" % (rc, work / "dt" / "openEMS.log"))
			failures += 1
			continue
		dt = float(m.group(1))
		# openEMS samples every Nyquist/OverSampling timesteps (OverSampling 4),
		# with Nyquist = floor(1/(2 f0 dt)) for a custom excitation.
		f0 = 1.0 / (2.0 * dt * 4 * args.interval + dt)
		xml_text = finalize(make(), args.nts, f0, args.subsample)
		# Blocking is on by default for large grids, so the flat runs turn it off.
		flat = {"OPENEMS_GPU_TEMPORAL_BLOCK": "0"}
		variants = [("flat", args.bin, flat), ("flat again", args.bin, flat)]
		if args.ref:
			variants.append(("ref flat", args.ref, flat))
		# k:W sets the tile; the 1 MB target only keeps the "grid already fits"
		# guard from refusing a small grid.  A bare k leaves the tile to the
		# engine's own sizing.
		variants += [("tblock " + s, args.bin, dict({"OPENEMS_GPU_TEMPORAL_BLOCK": s},
		                                            **({"OPENEMS_GPU_TEMPORAL_BLOCK_MB": "1"} if ":" in s else {})))
		             for s in args.schedules.split(",")]
		variants.append(("default", args.bin, {}))
		ref_dir = None
		for label, binary, env in variants:
			d = work / re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
			env = dict(case_env, **env)
			rc, log = run(binary, xml_text, d, env, args.gpu)
			blocked = env.get("OPENEMS_GPU_TEMPORAL_BLOCK", "0") != "0"
			engaged = ACTIVE_MARK in log
			info = ""
			if rc != 0:
				verdict = "FAIL (rc=%d, see %s)" % (rc, d / "openEMS.log")
			elif blocked and name in REFUSING and (engaged or REFUSING[name] not in log):
				verdict = "FAIL (expected the refusal '%s')" % REFUSING[name]
			elif blocked and name not in REFUSING and not engaged:
				why = re.search(r"temporal blocking disabled\s+\S+\s+(.*)", log)
				verdict = "FAIL (blocking did not engage: %s)" % (why.group(1) if why else "no message")
			elif ref_dir is None:
				ref_dir, verdict = d, "reference"
				info = "  [%s]" % nonzero_summary(d) if h5py else ""
			elif h5py is None:
				verdict = "ran (not compared: no h5py here)"
			else:
				diff = compare(ref_dir, d)
				verdict = "identical" if diff is None else ("DIFFERS (" + diff + ")")
				if diff is not None and label != "ref flat":
					verdict = "FAIL " + verdict
			if engaged:
				t = re.search(r"k=(\d+) W=(\d+) \((\d+) tiles", log)
				info += "  [k=%s W=%s, %s tiles]" % t.groups() if t else ""
			elif label == "default":
				why = re.search(r"temporal blocking disabled\s+\S+\s+(.*)", log)
				info += "  [not blocked: %s]" % (why.group(1) if why else "not a discrete GPU, or no message")
			if blocked and name in REFUSING and not engaged:
				info += "  [refused, as expected]"
			failures += verdict.startswith("FAIL")
			speed = re.search(r"^Speed: *([0-9.]+) MCells", log, re.M)
			print("  %-16s %8s MC/s  %s%s" % (label, speed.group(1) if speed else "-", verdict, info), flush=True)
		if not args.keep and h5py is not None:
			for h5 in work.rglob("*.h5"):
				h5.unlink()
	if h5py is None:
		print("\nno h5py: dumps kept in %s; compare them elsewhere with --compare" % args.work)
		return 0
	print("\n%s" % ("all identical" if failures == 0 else "%d FAILURES" % failures))
	return 1 if failures else 0


if __name__ == "__main__":
	sys.exit(main())
