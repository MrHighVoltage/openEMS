#!/usr/bin/env python3
"""Check that the AVX2 engine's blocked schedule is still bit-identical to the flat sweep.

    tblock_identity.py --ref OLD/openEMS --new NEW/openEMS [--threads 8,7]
                       [--cpus 0,2,4,6,8,10,12,14] [--nts 600] case.xml...

For every case, every variant below runs avx2-multithreaded with full-domain
E and H dumps (HDF5, subsampled), and every dump of every variant must equal
the reference binary's flat run bit for bit. --ref is the build before the
change under test, --new the one with it (the same binary is fine, too). A blocked variant that did not
engage blocking counts as a failure, not as a pass.

A case is either a complete openEMS XML (python/Tests/models/*.xml) or a
CSXCAD-only geometry (the TL cases), which gets an FDTD section with Mur on
all faces. The TL meshes have such a small timestep that dumps (and with them
the blocked schedule's probe cadence) would be ~440000 timesteps apart, so the
TL excitation is pushed up in frequency until dumps come every ~100
timesteps: the physics is meaningless, identity is all that is checked.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

BINS = {}

# (label, binary, environment). The reference is first.
VARIANTS = [
	("ref flat", "ref", {}),
	("new flat", "new", {}),
	("ref tblock 16", "ref", {"OPENEMS_AVX2_TEMPORAL_BLOCK": "16"}),
	("new tblock 16 (lines, no elision)", "new", {"OPENEMS_AVX2_TEMPORAL_BLOCK": "16",
	                                             "OPENEMS_AVX2_TBLOCK_SPLIT": "lines", "OPENEMS_AVX2_TBLOCK_ELIDE": "0"}),
	("new tblock 16 (rows, no elision)", "new", {"OPENEMS_AVX2_TEMPORAL_BLOCK": "16", "OPENEMS_AVX2_TBLOCK_ELIDE": "0"}),
	("new tblock 16 (lines, elision)", "new", {"OPENEMS_AVX2_TEMPORAL_BLOCK": "16", "OPENEMS_AVX2_TBLOCK_SPLIT": "lines"}),
	("new tblock 16 (rows, elision)", "new", {"OPENEMS_AVX2_TEMPORAL_BLOCK": "16"}),
	("new tblock 8:16 (rows, elision)", "new", {"OPENEMS_AVX2_TEMPORAL_BLOCK": "8:16"}),
	("new tblock 16 (rows, elision, spin 50 us)", "new", {"OPENEMS_AVX2_TEMPORAL_BLOCK": "16",
	                                                     "OPENEMS_AVX2_BARRIER_SPIN_US": "50"}),
	("new tblock 16 + sync stats", "new", {"OPENEMS_AVX2_TEMPORAL_BLOCK": "16", "OPENEMS_AVX2_SYNC_STATS": "1"}),
]
KNOBS = ("OPENEMS_AVX2_TEMPORAL_BLOCK", "OPENEMS_AVX2_TEMPORAL_BLOCK_FORCE", "OPENEMS_AVX2_TBLOCK_SPLIT",
         "OPENEMS_AVX2_TBLOCK_ELIDE", "OPENEMS_AVX2_BARRIER_SPIN_US", "OPENEMS_AVX2_SYNC_STATS")
SUBSAMPLE = "3,3,3"


def lines(grid, axis):
	return [float(v) for v in grid.find(axis + "Lines").text.split(",")]


def build_xml(case, nts):
	"""The case as an openEMS document with nts timesteps and E/H dumps over the whole domain."""
	root = ET.parse(case).getroot()
	if root.tag == "ContinuousStructure":
		doc = ET.Element("openEMS")
		fdtd = ET.SubElement(doc, "FDTD", NumberOfTimesteps=str(nts), endCriteria="1e-300")
		ET.SubElement(fdtd, "Excitation", Type="0", f0="6.6e14", fc="6.6e14")
		ET.SubElement(fdtd, "BoundaryCond", **{f: "MUR" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")})
		doc.append(root)
		csx = root
	else:
		doc = root
		doc.find("FDTD").set("NumberOfTimesteps", str(nts))
		csx = doc.find("ContinuousStructure")
	grid = csx.find("RectilinearGrid")
	ext = [(min(v), max(v)) for v in (lines(grid, a) for a in "XYZ")]
	props = csx.find("Properties")
	for n, (name, dtype) in enumerate((("Et", "0"), ("Ht", "1"))):
		db = ET.SubElement(props, "DumpBox", ID=str(90000 + n), Name=name, DumpType=dtype, DumpMode="0",
		                   FileType="1", SubSampling=SUBSAMPLE)
		box = ET.SubElement(ET.SubElement(db, "Primitives"), "Box", Priority="0")
		ET.SubElement(box, "P1", X=repr(ext[0][0]), Y=repr(ext[1][0]), Z=repr(ext[2][0]))
		ET.SubElement(box, "P2", X=repr(ext[0][1]), Y=repr(ext[1][1]), Z=repr(ext[2][1]))
	return ET.tostring(doc, encoding="unicode")


def run(xml_text, outdir, binary, env_extra, threads, cpus):
	if outdir.exists():
		shutil.rmtree(outdir)
	outdir.mkdir(parents=True)
	(outdir / "sim.xml").write_text(xml_text)
	env = {k: v for k, v in os.environ.items() if k not in KNOBS}
	env.update(env_extra)
	cmd = ["taskset", "-c", cpus, str(BINS[binary]), "sim.xml", "--engine=avx2-multithreaded",
	       "--numThreads=%d" % threads]
	p = subprocess.run(cmd, cwd=outdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
	                   universal_newlines=True)
	(outdir / "openEMS.log").write_text(p.stdout)
	return p.returncode, p.stdout


def datasets(path):
	out = {}
	with h5py.File(path, "r") as f:
		f.visititems(lambda name, obj: out.__setitem__(name, obj[()]) if isinstance(obj, h5py.Dataset) else None)
	return out


def compare(ref_dir, dir_):
	"""None if every dump dataset is bit-identical, else a description of the first difference."""
	for name in ("Et.h5", "Ht.h5"):
		a, b = datasets(ref_dir / name), datasets(dir_ / name)
		if sorted(a) != sorted(b):
			return "%s: dataset sets differ (%d vs %d)" % (name, len(a), len(b))
		for k in sorted(a):
			x, y = np.asarray(a[k]), np.asarray(b[k])
			if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
				d = np.abs(x.astype(np.float64) - y.astype(np.float64)).max() if x.shape == y.shape else float("nan")
				return "%s/%s differs (max |diff| %.3g)" % (name, k, d)
	return None


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("cases", nargs="+", type=Path)
	ap.add_argument("--ref", type=Path, required=True, help="openEMS binary of the reference build")
	ap.add_argument("--new", type=Path, required=True, help="openEMS binary of the build under test")
	ap.add_argument("--threads", default="8,7", help="comma-separated thread counts (default: %(default)s)")
	ap.add_argument("--cpus", default="0,2,4,6,8,10,12,14", help="CPUs to pin to (default: %(default)s)")
	ap.add_argument("--nts", type=int, default=600, help="timesteps per run (default: %(default)s)")
	ap.add_argument("--work", type=Path, default=Path("tblock_identity_work"),
	                help="scratch directory for the runs (default: %(default)s)")
	args = ap.parse_args()
	BINS.update(ref=args.ref.resolve(), new=args.new.resolve())

	failures = 0
	for case in args.cases:
		xml_text = build_xml(case, args.nts)
		ref_dir = None
		print("=== %s" % case.name, flush=True)
		for threads in (int(t) for t in args.threads.split(",")):
			for label, binary, env in VARIANTS:
				d = args.work / case.stem / ("%dt-" % threads + re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-"))
				rc, log = run(xml_text, d, binary, env, threads, args.cpus)
				blocked = "OPENEMS_AVX2_TEMPORAL_BLOCK" in env
				engaged = "AVX2 temporal blocking ACTIVE" in log
				speed = re.search(r"^Speed: *([0-9.]+) MCells", log, re.M)
				if rc != 0:
					verdict = "FAIL (rc=%d, see %s)" % (rc, d / "openEMS.log")
				elif blocked and not engaged:
					why = re.search(r"AVX2 temporal blocking: (.*)", log)
					verdict = "FAIL (blocking did not engage: %s)" % (why.group(1) if why else "no message")
				elif ref_dir is None:
					ref_dir, verdict = d, "reference"
				else:
					diff = compare(ref_dir, d)
					verdict = "identical" if diff is None else "FAIL (" + diff + ")"
				failures += verdict.startswith("FAIL")
				print("  %2dt  %-44s %8s MC/s  %s" % (threads, label, speed.group(1) if speed else "-", verdict),
				      flush=True)
				if "OPENEMS_AVX2_SYNC_STATS" in env:
					for line in re.findall(r"^(AVX2 sync stats.*(?:\n  .*)*)", log, re.M):
						print("        " + line.replace("\n", "\n        "))
	print("\n%s" % ("all identical" if failures == 0 else "%d FAILURES" % failures))
	return 1 if failures else 0


if __name__ == "__main__":
	sys.exit(main())
