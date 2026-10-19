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

Variants run one at a time, round-robin over --repeat rounds. --gpu pins the
run to one card through Mesa's device-select layer (DRI_PRIME), which is the
only reliable way where several identical cards are installed; a card with
more than 1 GiB of memory in use is refused unless --force. Results also go to
<out>/results.jsonl, one JSON object per run.

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
from pathlib import Path

from bench_tblock_sync import build_xml

MODES = {
	"plain": ([], {}),
	"batch": (["--gpu-profile"], {"OPENEMS_GPU_PROFILE_PHASES": "1"}),
	"stages": (["--gpu-profile"], {"OPENEMS_GPU_PROFILE_STAGES": "1"}),
	"nobarrier": (["--gpu-profile"], {"OPENEMS_GPU_PROFILE_NO_BARRIERS": "1"}),
}
NUM = r"([-0-9.eE+]+)"


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
		else:
			raise SystemExit("unknown option %r in %r" % (opt, spec))
	return args, env


def vram_used_mib(pci):
	try:
		return int(Path("/sys/bus/pci/devices", pci, "mem_info_vram_used").read_text()) // 2**20
	except (OSError, ValueError):
		return None


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


def run_one(binary, xml_text, outdir, spec, gpu):
	extra_args, extra_env = variant(spec)
	if outdir.exists():
		shutil.rmtree(str(outdir))
	outdir.mkdir(parents=True)
	(outdir / "sim.xml").write_text(xml_text)
	env = {k: v for k, v in os.environ.items() if not k.startswith("OPENEMS_GPU_")}
	env.update(extra_env)
	if gpu:
		env["DRI_PRIME"] = "pci-" + re.sub(r"[:.]", "_", gpu)
		env["MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE"] = "1"
	cmd = ["stdbuf", "-oL", str(binary), "sim.xml", "--engine=gpu"] + extra_args
	with open(str(outdir / "openEMS.log"), "w") as logf:
		rc = subprocess.call(cmd, cwd=str(outdir), env=env, stdout=logf, stderr=subprocess.STDOUT)
	res = {"host": socket.gethostname().split(".")[0], "variant": spec, "gpu_pci": gpu, "returncode": rc}
	res.update(parse((outdir / "openEMS.log").read_text()))
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
	args = ap.parse_args()
	for spec in args.variants:
		variant(spec)
	if args.gpu and not re.match(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$", args.gpu):
		ap.error("--gpu wants a full PCI address such as 0000:03:00.0")

	print("%s: %s, %d timesteps, %s" % (socket.gethostname().split(".")[0], args.gpu or "default GPU", args.nts,
	                                    args.case.name), flush=True)
	print("%-18s %8s %8s %7s %7s %8s  %s" % ("variant", "MC/s", "us/TS", "idle %", "barr/TS", "batches", "stages (us/TS)"),
	      flush=True)
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
				res = run_one(args.bin.resolve(), xml_text, d, spec, args.gpu)
				res.update(case=args.case.name, nts=args.nts, round=r)
				jf.write(json.dumps(res) + "\n")
				jf.flush()

				def f(key, fmt):
					return fmt % res[key] if res.get(key) is not None else "-"
				stages = " ".join("%s %.1f" % (k.replace(" ", ""), v) for k, v in res.get("stages_us", {}).items())
				print("%-18s %8s %8s %7s %7s %8s  %s" % (
					spec, f("mcs", "%.1f"), f("gpu_us_per_ts", "%.1f"), f("idle_pct", "%.2f"),
					f("barriers_per_ts", "%.1f"), f("batches", "%d"),
					"FAILED rc=%d" % res["returncode"] if res["returncode"] else
		"k=%d W=%d, %d tiles, %d MB field state/tile" % (res["tb_k"], res["tb_w"], res["tb_tiles"], res["tb_mb"])
		if "tb_k" in res else "blocking off: " + res["tb_off"] if "tb_off" in res else stages), flush=True)
	return 0


if __name__ == "__main__":
	sys.exit(main())
