#!/bin/bash
#
# Drive the fork-vs-upstream benchmark matrix from a portable bundle built by
# build_portable_bench.sh. Meant to be copied to a machine that has none of
# openEMS's dependencies installed and must not have any installed on it: the
# bundle is self-contained, this script only reads it.
#
#   run_portable_bench.sh calib  <spec> [<spec>...]   # thread-count sweep
#   run_portable_bench.sh matrix <threads> <pin_tuned> [pin_single]
#
# A calibration <spec> is "label:threads:cpulist", e.g. "s0:6:0-5". The sweep
# reports upstream's multithreaded engine, the fork's AVX2 one and the fork's
# temporally blocked one at each point, which is what picks the pinned thread
# count the matrix then uses for both sides (PERFORMANCE.md 2.1).
#
# Everything runs out of the bundle directory and writes only into it.
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
calib)
    shift
    xml=$XML/calib_224_pec.xml
    [ -e "$xml" ] || xml=$XML/224x224x224_pec.xml
    printf "%-14s %-8s %-12s %10s %10s %10s\n" host point cpus upstream-mt fork-avx2 fork-tblock
    for spec in "$@"; do
        label=${spec%%:*}; rest=${spec#*:}
        threads=${rest%%:*}; cpus=${rest#*:}
        a=$(speed "$UP"   multithreaded      "$threads" "$cpus" "$xml")
        b=$(speed "$FORK" avx2-multithreaded "$threads" "$cpus" "$xml")
        c=$(speed "$FORK" avx2-multithreaded "$threads" "$cpus" "$xml" \
                  OPENEMS_AVX2_TEMPORAL_BLOCK=$BLOCK_K)
        printf "%-14s %-8s %-12s %10s %10s %10s\n" "$HOST" "$label" "$cpus" "$a" "$b" "$c"
    done
    ;;
matrix)
    threads=${2:?need tuned thread count}
    pin_tuned=${3:?need tuned pin list}
    pin_single=${4:-0}
    out=$BUNDLE/results-$HOST.json
    exec python3 "$BUNDLE/benchmark_fork_vs_upstream.py" \
        --fork-build "$FORK" --upstream-build "$UP" \
        --csxcad-lib "$LIB" --xml-dir "$XML" \
        --gpu none --block-k "$BLOCK_K" \
        --tuned-threads "$threads" --pin-tuned "$pin_tuned" \
        --pin-single "$pin_single" --out "$out"
    ;;
*)
    sed -n '2,20p' "$0"; exit 1;;
esac
