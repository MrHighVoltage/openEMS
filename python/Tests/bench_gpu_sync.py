#!/usr/bin/env python3
"""Measure where the Vulkan engine's time goes: batch gaps, timestep stages, barriers.

    bench_gpu_sync.py --bin BUNDLE/fork/openEMS [--gpu 03:00.0] --nts 20000 \
                      case.xml VARIANT...

A VARIANT is MODE[/subN][/cmaxN]:
    plain       no profiling: the reference speed
    batch       --gpu-profile, OPENEMS_GPU_PROFILE_PHASES=1: GPU busy time per
                batch, the idle gaps between batches, compute/probe split
    stages      --gpu-profile, OPENEMS_GPU_PROFILE_STAGES=1: GPU time of each
                stage of every timestep (Yee V/I and the extension stages
                between them, each with its barrier)
    nobarrier   --gpu-profile, OPENEMS_GPU_PROFILE_NO_BARRIERS=1: the timestep
                loop without its compute barriers. Timing only, the fields are
                wrong; plain minus nobarrier bounds what the barriers cost
    subN        OPENEMS_GPU_MAX_TS_PER_SUBMIT=N timesteps per batch (default 64)
    cmaxN       OPENEMS_GPU_COMPRESS_MAX_TABLE_KB=N: keep the coefficients
                deduplicated up to an N KB table (default 2048)
    tbK, tbKwW  OPENEMS_GPU_TEMPORAL_BLOCK=K or K:W: trapezoidal temporal
                blocking of depth K, tile width W x-lines (default: from the
                tile target)
    tbmbN       OPENEMS_GPU_TEMPORAL_BLOCK_MB=N: tile target in MB of fields
                and coefficients (default 64 on a discrete card)
    murfuse0    OPENEMS_GPU_MUR_FUSION=0: Mur post/apply as their own
                dispatches instead of folded into the voltage update

Variants run one at a time, round-robin over --repeat rounds. --gpu pins the
run to one card through Mesa's device-select layer (DRI_PRIME), which is the
only reliable way where several identical cards are installed; a card with
more than 1 GiB of memory in use is refused unless --force. Results also go to
<out>/results.jsonl, one JSON object per run.

--counters reads the memory controllers and the RAPL energy counters while the
run steps, for an integrated GPU, whose traffic is the CPU's DRAM traffic and
whose power is in the CPU package (perf system-wide, so
kernel.perf_event_paranoid <= 0). Each run is preceded by an idle reading of the
same counters, which is subtracted: what is left is the run's, mostly the GPU's.
Where the memory controller counts the GPU's requests apart from the cores'
(uncore_imc/gt_requests/, Skylake-era client parts) that is reported too, as
"GT rd": it is the GPU's reads only. Its writes reach DRAM as write-backs from
the shared L3, which the controller counts as the cores' (ia_requests). The
power columns are the RAPL domains the machine has (package, its GPU plane, the
DRAM where measured); nJ/cell is package plus DRAM energy per cell update.

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
from pathlib import Path

from bench_tblock_sync import PERF_ENV, build_xml, dram_events

MODES = {
	"plain": ([], {}),
	"batch": (["--gpu-profile"], {"OPENEMS_GPU_PROFILE_PHASES": "1"}),
	"stages": (["--gpu-profile"], {"OPENEMS_GPU_PROFILE_STAGES": "1"}),
	"nobarrier": (["--gpu-profile"], {"OPENEMS_GPU_PROFILE_NO_BARRIERS": "1"}),
}
NUM = r"([-0-9.eE+]+)"
SETTLE, WINDOW, IDLE = 5, 20, 10   # DRAM counters: seconds after stepping starts, how long, idle baseline
GT_EVENT = "uncore_imc/gt_requests/"
RAPL = Path("/sys/bus/event_source/devices/power/events")


def variant(spec):
	parts = spec.split("/")
	if parts[0] not in MODES:
		raise SystemExit("bad variant %r (expected MODE[/OPTION...], MODE one of %s)" % (spec, ", ".join(MODES)))
	args, env = MODES[parts[0]][0], dict(MODES[parts[0]][1])
	for opt in parts[1:]:
		if re.match(r"^sub\d+$", opt):
			env["OPENEMS_GPU_MAX_TS_PER_SUBMIT"] = opt[3:]
		elif re.match(r"^cmax\d+$", opt):
			env["OPENEMS_GPU_COMPRESS_MAX_TABLE_KB"] = opt[4:]
		elif re.match(r"^tb\d+(w\d+)?$", opt):
			env["OPENEMS_GPU_TEMPORAL_BLOCK"] = opt[2:].replace("w", ":")
		elif re.match(r"^tbmb\d+$", opt):
			env["OPENEMS_GPU_TEMPORAL_BLOCK_MB"] = opt[4:]
		elif opt == "murfuse0":
			env["OPENEMS_GPU_MUR_FUSION"] = "0"
		else:
			raise SystemExit("unknown option %r in %r" % (opt, spec))
	return args, env


def vram_used_mib(pci):
	try:
		return int(Path("/sys/bus/pci/devices", pci, "mem_info_vram_used").read_text()) // 2**20
	except (OSError, ValueError):
		return None


def read_counters(ev, seconds):
	"""Over one window: all DRAM traffic and the GPU's reads (or None) in bytes/s, and
	watts per RAPL domain ("pkg", "gpu", "cores", "ram": whichever the machine has)."""
	gt = (Path("/sys/bus/event_source/devices/uncore_imc/events/gt_requests").exists()
	      and ev[0].startswith("uncore_imc/"))
	rapl = ["power/energy-%s/" % d for d in ("pkg", "gpu", "cores", "ram") if (RAPL / ("energy-" + d)).exists()]
	events = ",".join([ev[0]] + ([GT_EVENT] if gt else []) + rapl)
	p = subprocess.run(["perf", "stat", "-a", "-x,", "-e", events, "--", "sleep", str(seconds)], env=PERF_ENV,
	                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
	total, gpu, watts = 0.0, 0.0, {}
	for line in p.stderr.splitlines():
		f = line.split(",")
		try:
			v = float(f[0])
		except (ValueError, IndexError):
			continue
		name = f[2] if len(f) > 2 else ""
		if name.startswith("power/energy-"):
			watts[name[len("power/energy-"):].strip("/")] = v / seconds   # perf scales RAPL to Joules
		elif "gt_requests" in name:
			gpu += v * ev[1]
		else:
			total += v * ev[1]
	return {"dram": total / seconds, "gt": gpu / seconds if gt else None, "watts": watts}


def parse(log):
	res = {}
	def grab(key, pattern, conv=float):
		m = re.search(pattern, log, re.M)
		if m:
			res[key] = conv(m.group(1))
	grab("mcs", r"^Speed: *" + NUM + " MCells")
	grab("wall_s", r"^Time for \d+ iterations with [0-9.]+ cells : *" + NUM + " sec")
	grab("device", r"^Engine_Vulkan: using (.*)$", str)
	grab("batches", r"Batches: (\d+),", int)
	grab("gpu_us_per_ts", r"GPU time: .* ms/batch, " + NUM + " us/TS")
	grab("fence_wait_ms", r"Fence wait: " + NUM + " ms total")
	grab("idle_pct", r"idle between batches .* ms \(" + NUM + r"%\)")
	grab("barriers_per_ts", r"Compute barriers per timestep: " + NUM)
	m = re.search(r"temporal blocking active .* k=(\d+) W=(\d+) \((\d+) tiles, (\d+) MB/tile", log)
	if m:
		res.update(tb_k=int(m.group(1)), tb_w=int(m.group(2)), tb_tiles=int(m.group(3)), tb_mb=int(m.group(4)))
	grab("tb_off", r"temporal blocking disabled\s+\S+\s+(.*)$", str)
	m = re.search(r"Phase split \(GPU\): compute=" + NUM + " ms .*probe=" + NUM + " ms .*tail=" + NUM + " ms", log)
	if m:
		res.update(compute_ms=float(m.group(1)), probe_ms=float(m.group(2)), tail_ms=float(m.group(3)))
	m = re.search(r"Stage split \(GPU, (\d+) timesteps, us/TS\): (.*), sum=" + NUM, log)
	if m:
		res["stage_ts"] = int(m.group(1))
		res["stages_us"] = {k.strip(): float(v) for k, v in re.findall(r"([^,=]+?)=" + NUM + r" \(", m.group(2))}
		res["stage_sum_us"] = float(m.group(3))
	return res


def run_one(binary, xml_text, outdir, spec, gpu, ev):
	extra_args, extra_env = variant(spec)
	if outdir.exists():
		shutil.rmtree(str(outdir))
	outdir.mkdir(parents=True)
	(outdir / "sim.xml").write_text(xml_text)
	idle = read_counters(ev, IDLE) if ev else None
	env = {k: v for k, v in os.environ.items() if not k.startswith("OPENEMS_GPU_")}
	env.update(extra_env)
	if gpu:
		env["DRI_PRIME"] = "pci-" + re.sub(r"[:.]", "_", gpu)
		env["MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE"] = "1"
	cmd = ["stdbuf", "-oL", str(binary), "sim.xml", "--engine=gpu"] + extra_args
	busy = None
	with open(str(outdir / "openEMS.log"), "w") as logf:
		p = subprocess.Popen(cmd, cwd=str(outdir), env=env, stdout=logf, stderr=subprocess.STDOUT)
		if ev:
			log = outdir / "openEMS.log"
			while p.poll() is None and "Running FDTD engine" not in log.read_text():
				time.sleep(0.5)
			time.sleep(SETTLE)
			if p.poll() is None:
				busy = read_counters(ev, WINDOW)
				if p.poll() is not None:
					busy = None   # the run ended inside the window: the reading is partly idle
		rc = p.wait()
	res = {"host": socket.gethostname().split(".")[0], "variant": spec, "gpu_pci": gpu, "returncode": rc}
	res.update(parse((outdir / "openEMS.log").read_text()))
	if idle:
		res["dram_idle_gbs"] = idle["dram"] / 1e9
		res["idle_w"] = idle["watts"]
	if busy:
		res["dram_gbs"] = (busy["dram"] - idle["dram"]) / 1e9
		res["busy_w"] = busy["watts"]
		res["net_w"] = {d: w - idle["watts"].get(d, 0.0) for d, w in busy["watts"].items()}
		if busy["gt"] is not None:
			res["gt_gbs"] = (busy["gt"] - idle["gt"]) / 1e9
		if res.get("mcs"):
			res["bytes_per_cell"] = res["dram_gbs"] * 1e3 / res["mcs"]
			if "gt_gbs" in res:
				res["gt_bytes_per_cell"] = res["gt_gbs"] * 1e3 / res["mcs"]
			if "pkg" in res["net_w"]:
				res["nj_per_cell"] = (res["net_w"]["pkg"] + res["net_w"].get("ram", 0.0)) * 1e3 / res["mcs"]
	return res


def main():
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("case", type=Path)
	ap.add_argument("variants", nargs="+")
	ap.add_argument("--bin", type=Path, required=True, help="openEMS binary with the Vulkan engine")
	ap.add_argument("--gpu", help="PCI address of the card to pin, e.g. 0000:03:00.0")
	ap.add_argument("--force", action="store_true", help="run even if the card's memory is in use")
	ap.add_argument("--nts", type=int, default=20000, help="timesteps per run (default: %(default)s)")
	ap.add_argument("--repeat", type=int, default=1, help="rounds over all variants (default: %(default)s)")
	ap.add_argument("--out", type=Path, default=Path("bench_gpu_sync"),
	                help="directory for the runs and results.jsonl (default: %(default)s)")
	ap.add_argument("--counters", action="store_true",
	                help="read the memory controllers and RAPL energy while the run steps")
	args = ap.parse_args()
	for spec in args.variants:
		variant(spec)
	if args.gpu and not re.match(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$", args.gpu):
		ap.error("--gpu wants a full PCI address such as 0000:03:00.0")

	ev = None
	if args.counters:
		ev = dram_events()
		if ev is None:
			sys.exit("--counters: the memory controllers are not readable here (perf, kernel.perf_event_paranoid <= 0)")
	print("%s: %s, %d timesteps, %s%s" % (socket.gethostname().split(".")[0], args.gpu or "default GPU", args.nts,
	                                      args.case.name, "; DRAM counters " + ev[0].split("/")[0] if ev else ""),
	      flush=True)
	print("%-18s %8s %8s %7s %7s %8s %7s %7s %7s %6s %6s %6s %7s  %s" % (
		"variant", "MC/s", "us/TS", "idle %", "barr/TS", "batches", "GB/s", "B/cell", "GT rd",
		"pkg W", "gpu W", "ram W", "nJ/cell", "stages (us/TS)"), flush=True)
	xml_text = build_xml(args.case, args.nts)
	args.out.mkdir(parents=True, exist_ok=True)
	with open(str(args.out / "results.jsonl"), "a") as jf:
		for r in range(args.repeat):
			for spec in args.variants:
				if args.gpu and not args.force:
					used = vram_used_mib(args.gpu)
					if used is not None and used > 1024:
						sys.exit("GPU %s has %d MiB of memory in use by another process (--force runs anyway)" % (args.gpu, used))
				d = args.out / args.case.stem / ("r%d-" % r + re.sub(r"[^a-z0-9]+", "-", spec.lower()).strip("-"))
				res = run_one(args.bin.resolve(), xml_text, d, spec, args.gpu, ev)
				res.update(case=args.case.name, nts=args.nts, round=r)
				jf.write(json.dumps(res) + "\n")
				jf.flush()

				def f(key, fmt):
					return fmt % res[key] if res.get(key) is not None else "-"

				def w(domain):
					return "%.1f" % res["net_w"][domain] if domain in res.get("net_w", {}) else "-"
				stages = " ".join("%s %.1f" % (k.replace(" ", ""), v) for k, v in res.get("stages_us", {}).items())
				print("%-18s %8s %8s %7s %7s %8s %7s %7s %7s %6s %6s %6s %7s  %s" % (
					spec, f("mcs", "%.1f"), f("gpu_us_per_ts", "%.1f"), f("idle_pct", "%.2f"),
					f("barriers_per_ts", "%.1f"), f("batches", "%d"),
					f("dram_gbs", "%.1f"), f("bytes_per_cell", "%.1f"), f("gt_bytes_per_cell", "%.1f"),
					w("pkg"), w("gpu"), w("ram"), f("nj_per_cell", "%.1f"),
					"FAILED rc=%d" % res["returncode"] if res["returncode"] else
		"k=%d W=%d, %d tiles, %d MB field state/tile" % (res["tb_k"], res["tb_w"], res["tb_tiles"], res["tb_mb"])
		if "tb_k" in res else "blocking off: " + res["tb_off"] if "tb_off" in res else stages), flush=True)
	return 0


if __name__ == "__main__":
	sys.exit(main())
