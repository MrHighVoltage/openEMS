#!/bin/bash
#
# Drive the GPU half of the fork-vs-upstream matrix from a bundle built by
# build_portable_bench_gpu.sh.
#
#   run_portable_bench_gpu.sh probe                          # what Vulkan sees
#   run_portable_bench_gpu.sh ksweep <label:index> [k...]    # block-depth sweep
#   run_portable_bench_gpu.sh matrix <label:index:k> <threads> <pin_tuned> [pin_single]
#
# Deliberately separate from run_portable_bench.sh: that script produced the
# CPU numbers for seven machines and stays frozen, and it hardcodes --gpu none.
# This one is additive -- it never writes where the CPU runner writes (results
# land in results-gpu-$HOST.json), so a GPU run cannot overwrite a CPU record.
#
# Unlike the CPU bundle, this one needs things from the target: a Vulkan ICD
# (/usr/share/vulkan/icd.d) and a system libvulkan.so.1. The loader is the
# host's on purpose -- it has to match the host's driver.
#
# The GPU rows are unpinned by design (benchmark_fork_vs_upstream.py:
# "the CPU is not the resource under test"), but the CPU rows in the same
# matrix still are, so the run needs the same tuned thread count the CPU
# calibration picked for this host.
set -u

BUNDLE=$(cd "$(dirname "$0")" && pwd)
FORK=$BUNDLE/fork
UP=$BUNDLE/upstream
LIB=$BUNDLE/lib
XML=$BUNDLE/xml
HOST=$(hostname -s)
BLOCK_K=${BLOCK_K:-16}

speed() {  # speed <build> <engine> <threads> <cpus> <xml> [env...]
    local build=$1 engine=$2 threads=$3 cpus=$4 xml=$5; shift 5
    local pin=""
    [ -n "$cpus" ] && pin="taskset -c $cpus"
    local nt=""
    [ "$threads" != "0" ] && nt="--numThreads=$threads"
    local out
    out=$(cd "$(mktemp -d)" && env "$@" LD_LIBRARY_PATH="$build:$LIB" \
          $pin "$build/openEMS" "$xml" "--engine=$engine" $nt 2>&1)
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "FAIL(rc=$rc)"
        return
    fi
    echo "$out" | sed -n 's/^Speed:[ \t]*\([0-9.eE+-]*\).*/\1/p' | tail -1
}

case "${1:-}" in
probe)
    echo "=== loader and ICDs on $HOST"
    ls /usr/share/vulkan/icd.d/ 2>/dev/null || echo "  no ICD directory"
    found=$(ls -d /usr/lib*/libvulkan.so.1 /usr/lib/*/libvulkan.so.1 2>/dev/null)
    [ -n "$found" ] && echo "$found" || echo "  no libvulkan.so.1"
    echo "=== devices as the fork's engine enumerates them"
    # The engine clamps an out-of-range OPENEMS_GPU_INDEX back to device 0
    # rather than failing, so enumeration stops as soon as a higher index
    # reports the same device as index 0 -- otherwise this loops forever on a
    # single-GPU host.
    xml=$XML/64x64x64_pec.xml
    first=""
    for i in 0 1 2 3; do
        out=$(cd "$(mktemp -d)" && env OPENEMS_GPU_INDEX=$i \
              LD_LIBRARY_PATH="$FORK:$LIB" \
              "$FORK/openEMS" "$xml" --engine=gpu 2>&1)
        dev=$(echo "$out" | sed -n 's/.*Engine_Vulkan:[ ]*\(using\|selected\|device\)[ ]*/\1 /p' | head -1)
        [ -z "$dev" ] && dev=$(echo "$out" | grep -i -m1 "Engine_Vulkan" )
        [ -z "$dev" ] && break
        if [ -z "$first" ]; then
            first=$dev
        elif [ "$dev" = "$first" ]; then
            break
        fi
        printf "  index %d: %s\n" "$i" "$dev"
    done
    ;;
ksweep)
    # Pick the GPU temporal-block depth before the matrix fixes one, the way
    # the CPU side picks its thread count first. k=0 is the flat control.
    spec=${2:?need label:index}
    label=${spec%%:*}; index=${spec#*:}
    shift 2
    ks=${*:-0 4 8 16 32}
    xml=$XML/224x224x224_pec.xml
    printf "%-14s %-10s %6s %10s\n" host gpu k mcells
    for k in $ks; do
        if [ "$k" = "0" ]; then
            v=$(speed "$FORK" gpu 0 "" "$xml" OPENEMS_GPU_INDEX="$index")
        else
            v=$(speed "$FORK" gpu 0 "" "$xml" OPENEMS_GPU_INDEX="$index" \
                      OPENEMS_GPU_TEMPORAL_BLOCK="$k")
        fi
        printf "%-14s %-10s %6s %10s\n" "$HOST" "$label" "$k" "$v"
    done
    ;;
matrix)
    spec=${2:?need label:index:k}
    threads=${3:?need tuned thread count}
    pin_tuned=${4:?need tuned pin list}
    pin_single=${5:-0}
    out=$BUNDLE/results-gpu-$HOST.json
    exec python3 "$BUNDLE/benchmark_fork_vs_upstream.py" \
        --fork-build "$FORK" --upstream-build "$UP" \
        --csxcad-lib "$LIB" --xml-dir "$XML" \
        --gpu "$spec" --block-k "$BLOCK_K" \
        --tuned-threads "$threads" --pin-tuned "$pin_tuned" \
        --pin-single "$pin_single" --out "$out"
    ;;
*)
    sed -n '2,22p' "$0"; exit 1;;
esac
