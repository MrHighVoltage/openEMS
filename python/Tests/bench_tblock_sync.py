#!/usr/bin/env python3
"""Measure the AVX2 engine's blocked schedule under different synchronisation settings.

    bench_tblock_sync.py --ref OLD/openEMS --new NEW/openEMS --threads 8 \
                         --cpus 0,2,4,6,8,10,12,14 --nts 8000 case.xml VARIANT...

A VARIANT is BIN/TBLOCK[/OPTION...]:
    BIN      ref (--ref, e.g. the build before a change) or new (--new)
    TBLOCK   flat, or an OPENEMS_AVX2_TEMPORAL_BLOCK value: k or k:W
    OPTION   lines | rows        how a slab's update is split over threads (new only)
             elide | noelide     extension barriers skipped/merged or not (new only)
             spinN               spin N us before sleeping at a barrier (new only)
e.g. ref/16:44  new/16:44/lines/noelide  new/16:44  new/16:44/spin20

Variants run one at a time, round-robin over --repeat rounds. Every new-binary
run has OPENEMS_AVX2_SYNC_STATS=1 (it costs two clock reads per barrier), so
its barrier count and waiting time are reported next to its speed. DRAM traffic
is read from the memory controllers while the run steps if perf may read them
system-wide (kernel.perf_event_paranoid <= 0), else left out. Results also go
to <out>/results.jsonl, one JSON object per run.

A case is a CSXCAD-only geometry (gets tests/run_gpu.py's FDTD settings:
Gaussian DC..300 GHz, Mur on all faces) or a complete openEMS XML.
"""
import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

BINS = {"ref": None, "new": None}
KNOBS = ("OPENEMS_AVX2_TEMPORAL_BLOCK", "OPENEMS_AVX2_TEMPORAL_BLOCK_FORCE", "OPENEMS_AVX2_TBLOCK_SPLIT",
         "OPENEMS_AVX2_TBLOCK_ELIDE", "OPENEMS_AVX2_BARRIER_SPIN_US", "OPENEMS_AVX2_SYNC_STATS")
SETTLE, WINDOW = 5, 20   # DRAM counters: seconds after stepping starts, and how long


def variant_env(spec):
	parts = spec.split("/")
	if len(parts) < 2 or parts[0] not in BINS:
		raise SystemExit("bad variant %r (expected BIN/TBLOCK[/OPTION...])" % spec)
	binary, tblock, env = parts[0], parts[1], {}
	if tblock != "flat":
		if not re.match(r"^\d+(:\d+)?$", tblock):
			raise SystemExit("bad TBLOCK in %r" % spec)
		env["OPENEMS_AVX2_TEMPORAL_BLOCK"] = tblock
	for opt in parts[2:]:
		if binary != "new":
			raise SystemExit("options need the new binary: %r" % spec)
		if opt in ("lines", "rows"):
			env["OPENEMS_AVX2_TBLOCK_SPLIT"] = opt
		elif opt in ("elide", "noelide"):
			env["OPENEMS_AVX2_TBLOCK_ELIDE"] = "1" if opt == "elide" else "0"
		elif re.match(r"^spin\d+$", opt):
			env["OPENEMS_AVX2_BARRIER_SPIN_US"] = opt[4:]
		else:
			raise SystemExit("unknown option %r in %r" % (opt, spec))
	if binary == "new":
		env["OPENEMS_AVX2_SYNC_STATS"] = "1"
	return binary, env


def build_xml(case, nts):
	root = ET.parse(str(case)).getroot()
	if root.tag == "ContinuousStructure":
		doc = ET.Element("openEMS")
		fdtd = ET.SubElement(doc, "FDTD", NumberOfTimesteps=str(nts), MaxTime="6e-11", endCriteria="1e-5")
		ET.SubElement(fdtd, "Excitation", Type="0", f0="1.5e11", fc="1.5e11")
		ET.SubElement(fdtd, "BoundaryCond", **{f: "MUR" for f in ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")})
		doc.append(root)
	else:
		doc = root
		doc.find("FDTD").set("NumberOfTimesteps", str(nts))
	return ET.tostring(doc, encoding="unicode")


def dram_events():
	"""(perf event list, bytes per count) for this machine's memory controllers, or None."""
	dev = Path("/sys/bus/event_source/devices")
	# (PMU directory to look in, perf alias, read event, write event); the first
	# whose events exist wins. perf scales an event that has a .scale file to
	# its .unit (MiB here); amd_df's DRAM events have none and count 64-byte lines.
	candidates = (("amd_df", "amd_df", None, None),
	              ("uncore_imc_free_running_0", "uncore_imc_free_running", "data_read", "data_write"),
	              ("uncore_imc", "uncore_imc", "data_reads", "data_writes"),
	              ("uncore_imc_0", "uncore_imc", "cas_count_read", "cas_count_write"))
	ev = None
	for pmu, alias, rd, wr in candidates:
		if not (dev / pmu).is_dir():
			continue
		if rd is None:
			ev = ("amd_df/dram_channel_data_controller_0/,amd_df/dram_channel_data_controller_1/", 64.0)
			break
		unit = dev / pmu / "events" / (rd + ".unit")
		if (dev / pmu / "events" / rd).exists():
			mib = unit.exists() and unit.read_text().strip() == "MiB"
			ev = ("%s/%s/,%s/%s/" % (alias, rd, alias, wr), 1048576.0 if mib else 64.0)
			break
	if ev is None:
		return None
	try:
		p = subprocess.run(["perf", "stat", "-a", "-x,", "-e", ev[0], "--", "sleep", "0.1"],
		                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
	except OSError:
		return None
	return ev if p.returncode == 0 else None


def read_dram(ev, seconds):
	p = subprocess.run(["perf", "stat", "-a", "-x,", "-e", ev[0], "--", "sleep", str(seconds)],
	                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
	total = 0.0
	for line in p.stderr.splitlines():
		f = line.split(",")
		try:
			total += float(f[0]) * ev[1]
		except (ValueError, IndexError):
			pass
	return total / seconds


def parse_stats(log):
	m = re.search(r"barriers per timestep ([0-9.]+) \(update ([0-9.]+), extension ([0-9.]+)\)\n"
	              r"  share of thread time waiting for the slowest thread ([0-9.]+) % \(update ([0-9.]+), "
	              r"extension ([0-9.]+)\), waking up ([0-9.]+) %; per barrier ([0-9.]+) \+ ([0-9.]+) us", log)
	if not m:
		return {}
	v = [float(x) for x in m.groups()]
	return {"barriers_per_ts": v[0], "barriers_update": v[1], "barriers_ext": v[2], "wait_pct": v[3],
	        "wait_update_pct": v[4], "wait_ext_pct": v[5], "wake_pct": v[6], "wait_us": v[7], "wake_us": v[8]}


def run_one(xml_text, outdir, spec, threads, cpus, ev):
	binary, env_extra = variant_env(spec)
	if outdir.exists():
		shutil.rmtree(str(outdir))
	outdir.mkdir(parents=True)
	(outdir / "sim.xml").write_text(xml_text)
	env = {k: v for k, v in os.environ.items() if k not in KNOBS}
	env.update(env_extra)
	cmd = ["taskset", "-c", cpus, "stdbuf", "-oL", str(BINS[binary]), "sim.xml", "--engine=avx2-multithreaded",
	       "--numThreads=%d" % threads]
	logf = open(str(outdir / "openEMS.log"), "w")
	p = subprocess.Popen(cmd, cwd=str(outdir), env=env, stdout=logf, stderr=subprocess.STDOUT)
	bw = None
	if ev:
		log = outdir / "openEMS.log"
		while p.poll() is None and "Running FDTD engine" not in log.read_text():
			time.sleep(0.5)
		time.sleep(SETTLE)
		if p.poll() is None:
			bw = read_dram(ev, WINDOW)
			if p.poll() is not None:
				bw = None   # the run ended inside the window: the reading is partly idle
	p.wait()
	logf.close()
	log = (outdir / "openEMS.log").read_text()
	res = {"host": socket.gethostname().split(".")[0], "variant": spec, "threads": threads, "cpus": cpus,
	       "returncode": p.returncode}
	m = re.search(r"^Speed: *([0-9.]+) MCells", log, re.M)
	res["mcs"] = float(m.group(1)) if m else None
	m = re.search(r"^AVX2 temporal blocking (.*)$", log, re.M)
	res["tblock"] = m.group(1) if m else ("not requested" if "OPENEMS_AVX2_TEMPORAL_BLOCK" not in env_extra else "?")
	res.update(parse_stats(log))
	if bw is not None:
		res["dram_gbs"] = bw / 1e9
		if res["mcs"]:
			res["bytes_per_cell"] = bw / (res["mcs"] * 1e6)
	return res


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("case", type=Path)
	ap.add_argument("variants", nargs="+")
	ap.add_argument("--threads", type=int, required=True)
	ap.add_argument("--cpus", required=True, help="taskset CPU list, one thread per physical core")
	ap.add_argument("--nts", type=int, default=8000, help="timesteps per run (default: %(default)s)")
	ap.add_argument("--repeat", type=int, default=1, help="rounds over all variants (default: %(default)s)")
	ap.add_argument("--ref", type=Path, help="openEMS binary for ref/ variants")
	ap.add_argument("--new", type=Path, help="openEMS binary for new/ variants")
	ap.add_argument("--out", type=Path, default=Path("bench_tblock_sync"),
	                help="directory for the runs and results.jsonl (default: %(default)s)")
	args = ap.parse_args()
	BINS.update(ref=args.ref.resolve() if args.ref else None, new=args.new.resolve() if args.new else None)
	for spec in args.variants:
		if BINS[spec.split("/")[0]] is None:
			ap.error("variant %s needs --%s" % (spec, spec.split("/")[0]))
	for spec in args.variants:
		variant_env(spec)

	ev = dram_events()
	print("%s: %d threads on CPUs %s, %d timesteps, %s; DRAM counters %s" % (
		socket.gethostname().split(".")[0], args.threads, args.cpus, args.nts, args.case.name,
		ev[0].split("/")[0] if ev else "not readable"), flush=True)
	print("%-28s %8s %9s %8s %8s %9s %8s %8s  %s" % ("variant", "MC/s", "barr/TS", "wait %", "wake %", "us/barr",
	                                                   "GB/s", "B/cell", "blocking"), flush=True)
	xml_text = build_xml(args.case, args.nts)
	args.out.mkdir(parents=True, exist_ok=True)
	with open(str(args.out / "results.jsonl"), "a") as jf:
		for r in range(args.repeat):
			for spec in args.variants:
				d = args.out / args.case.stem / ("r%d-" % r + re.sub(r"[^a-z0-9]+", "-", spec.lower()).strip("-"))
				res = run_one(xml_text, d, spec, args.threads, args.cpus, ev)
				res.update(case=args.case.name, nts=args.nts, round=r)
				jf.write(json.dumps(res) + "\n")
				jf.flush()

				def f(key, fmt):
					return fmt % res[key] if res.get(key) is not None else "-"
				print("%-28s %8s %9s %8s %8s %9s %8s %8s  %s" % (
					spec, f("mcs", "%.1f"), f("barriers_per_ts", "%.1f"), f("wait_pct", "%.1f"), f("wake_pct", "%.1f"),
					f("wait_us", "%.2f") if res.get("wait_us") is None else "%.2f+%.2f" % (res["wait_us"], res["wake_us"]),
					f("dram_gbs", "%.1f"), f("bytes_per_cell", "%.1f"),
					res["tblock"][:60] if res["returncode"] == 0 else "FAILED rc=%d" % res["returncode"]), flush=True)
	return 0


if __name__ == "__main__":
	sys.exit(main())
