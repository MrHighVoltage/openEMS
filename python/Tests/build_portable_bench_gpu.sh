#!/bin/bash
#
# Build a portable openEMS benchmark bundle *with* the Vulkan GPU engine, for
# the GPU half of the fork-vs-upstream matrix.
#
# Why this is a second script rather than a flag on build_portable_bench.sh:
# the CPU bundle is what produced every CPU number in PERFORMANCE.md, on seven
# machines, and those numbers have to stay reproducible from an unchanged
# script. A GPU-enabled fork binary is a *different* binary -- WITH_GPU adds
# -DWITH_GPU, links the Vulkan loader and embeds SPIR-V -- so it gets its own
# builder and its own bundle, and the CPU record stays untouched.
#
# What is shared and what is not:
#   - shared: every dependency stage (gmp, mpfr, boost, CGAL, tinyxml, hdf5,
#     VTK, fparser, CSXCAD) and the *upstream* openEMS build, all reused from
#     $ROOT by stamp. Run build_portable_bench.sh first; this script refuses to
#     guess. Upstream has no GPU engine, so the GPU rows are fork-only and the
#     upstream side must be bit-identical to the CPU bundle's for the
#     comparison to mean anything.
#   - added here: Vulkan-Headers, the Vulkan loader (link target only) and
#     shaderc's glslc (compiles the .comp shaders to SPIR-V, which CMake then
#     embeds into the binary -- nothing ships loose).
#
# Runtime requirement on the target, unlike the CPU bundle: a Vulkan ICD and a
# system libvulkan.so.1. The bundle deliberately does NOT carry a loader --
# the loader must match the target's driver, so it uses the host's.
#
# Usage:
#   python/Tests/build_portable_bench_gpu.sh
#   ROOT=/somewhere python/Tests/build_portable_bench_gpu.sh
set -euo pipefail

IMAGE=${IMAGE:-quay.io/pypa/manylinux2014_x86_64}
ROOT=${ROOT:-$HOME/openems-portable}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
UPSTREAM=${UPSTREAM:-$(cd "$REPO/.." && pwd)/openEMS-upstream-bench}
CSXCAD_SRC=${CSXCAD_SRC:-$(cd "$REPO/.." && pwd)/CSXCAD}

if [ "${IN_CONTAINER:-0}" != "1" ]; then
    for d in "$REPO" "$UPSTREAM" "$CSXCAD_SRC"; do
        [ -d "$d" ] || { echo "missing source tree: $d" >&2; exit 1; }
    done
    for s in csxcad openems-upstream; do
        [ -e "$ROOT/stamps/$s" ] || {
            echo "missing stage '$s' in $ROOT -- run build_portable_bench.sh first" >&2
            exit 1; }
    done
    echo "=== building in $IMAGE ==="
    exec docker run --rm -i \
        -v "$ROOT:$ROOT" -v "$REPO:$REPO" \
        -v "$UPSTREAM:$UPSTREAM" -v "$CSXCAD_SRC:$CSXCAD_SRC" \
        -e IN_CONTAINER=1 -e ROOT="$ROOT" -e REPO="$REPO" \
        -e UPSTREAM="$UPSTREAM" -e CSXCAD_SRC="$CSXCAD_SRC" \
        -e HOME="$ROOT" \
        -w "$ROOT" "$IMAGE" bash "$REPO/python/Tests/build_portable_bench_gpu.sh"
fi

# ---------------------------------------------------------------- in container
SRC=$ROOT/src
DEPS=$ROOT/deps
STAMPS=$ROOT/stamps
BUNDLE=$ROOT/bundle-gpu
J=$(nproc)
mkdir -p "$SRC" "$DEPS" "$STAMPS" "$BUNDLE"

export CFLAGS="-O2 -fPIC"
export CXXFLAGS="-O2 -fPIC"
export LDFLAGS="-static-libstdc++ -static-libgcc"
export CMAKE_PREFIX_PATH=$DEPS
PORTABLE_LINK="-static-libstdc++ -static-libgcc -Wl,--enable-new-dtags"

# Same toolchain the CPU build left in $ROOT/tools, plus a python3 for the
# Vulkan loader's and shaderc's generators (the image's own is under /opt).
export PATH=$ROOT/tools/bin:$PATH
command -v python3 >/dev/null || export PATH=$PATH:/opt/python/cp39-cp39/bin
for t in cmake ninja python3 git; do
    command -v $t >/dev/null || { echo "missing tool in container: $t" >&2; exit 1; }
done

stage() {
    if [ -e "$STAMPS/$1" ]; then echo "=== $1: already built"; return 1; fi
    echo "=== $1"; return 0
}
done_stage() { touch "$STAMPS/$1"; }

VULKAN_TAG=${VULKAN_TAG:-v1.3.280}
SHADERC_TAG=${SHADERC_TAG:-v2024.1}

# --- Vulkan-Headers ---------------------------------------------------------
if stage vulkan-headers; then
    [ -d "$SRC/Vulkan-Headers" ] || git clone -q --depth 1 -b "$VULKAN_TAG" \
        https://github.com/KhronosGroup/Vulkan-Headers "$SRC/Vulkan-Headers"
    cmake -S "$SRC/Vulkan-Headers" -B "$SRC/Vulkan-Headers/build" -G Ninja \
        -DCMAKE_INSTALL_PREFIX="$DEPS" >/dev/null
    cmake --install "$SRC/Vulkan-Headers/build" >/dev/null
    done_stage vulkan-headers
fi

# --- Vulkan loader ----------------------------------------------------------
# Built only so the fork has something to link against. At runtime the target's
# own loader is used: the bundle does not carry libvulkan (see header).
# WSI off: the image has no X11/Wayland headers, and the engine is headless
# compute -- it never presents a surface. The target's own loader, which does
# have WSI, is what actually runs.
if stage vulkan-loader; then
    [ -d "$SRC/Vulkan-Loader" ] || git clone -q --depth 1 -b "$VULKAN_TAG" \
        https://github.com/KhronosGroup/Vulkan-Loader "$SRC/Vulkan-Loader"
    cmake -S "$SRC/Vulkan-Loader" -B "$SRC/Vulkan-Loader/build" -G Ninja \
        -DCMAKE_INSTALL_PREFIX="$DEPS" -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_BUILD_TYPE=Release \
        -DVULKAN_HEADERS_INSTALL_DIR="$DEPS" \
        -DBUILD_TESTS=OFF \
        -DBUILD_WSI_XLIB_SUPPORT=OFF -DBUILD_WSI_XCB_SUPPORT=OFF \
        -DBUILD_WSI_WAYLAND_SUPPORT=OFF >/dev/null
    cmake --build "$SRC/Vulkan-Loader/build" -j"$J" >/dev/null
    cmake --install "$SRC/Vulkan-Loader/build" >/dev/null
    done_stage vulkan-loader
fi

# --- shaderc (glslc) --------------------------------------------------------
# CMake's WITH_GPU path wants glslc specifically. Google's prebuilt glslc is
# linked against a newer glibc than this image has, so build it from source;
# git-sync-deps fetches glslang and SPIRV-Tools at the pinned revisions.
if stage shaderc; then
    [ -d "$SRC/shaderc" ] || git clone -q --depth 1 -b "$SHADERC_TAG" \
        https://github.com/google/shaderc "$SRC/shaderc"
    (cd "$SRC/shaderc" && python3 ./utils/git-sync-deps)
    cmake -S "$SRC/shaderc" -B "$SRC/shaderc/build" -G Ninja \
        -DCMAKE_INSTALL_PREFIX="$DEPS" -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_BUILD_TYPE=Release \
        -DSHADERC_SKIP_TESTS=ON -DSHADERC_SKIP_EXAMPLES=ON \
        -DSHADERC_SKIP_COPYRIGHT_CHECK=ON >/dev/null
    cmake --build "$SRC/shaderc/build" -j"$J" >/dev/null
    cmake --install "$SRC/shaderc/build" >/dev/null
    done_stage shaderc
fi
"$DEPS/bin/glslc" --version | head -2

# --- openEMS: this fork, with the Vulkan engine -----------------------------
# Everything except WITH_GPU matches build_portable_bench.sh's build_openems,
# so the CPU engines in this binary are built exactly as the CPU bundle's are.
if stage openems-fork-gpu; then
    cmake -S "$REPO" -B "$ROOT/build-fork-gpu" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$DEPS" \
        -DCSXCAD_ROOT_DIR="$DEPS" -DFPARSER_ROOT_DIR="$DEPS" \
        -DTinyXML_ROOT_DIR="$DEPS" \
        -DHDF5_ROOT="$DEPS" -DHDF5_USE_STATIC_LIBRARIES=ON \
        -DBOOST_ROOT="$DEPS" -DBoost_USE_STATIC_LIBS=ON \
        -DWITH_GPU=ON \
        -DVulkan_INCLUDE_DIR="$DEPS/include" \
        -DVulkan_LIBRARY="$DEPS/lib/libvulkan.so" \
        -DGLSLC="$DEPS/bin/glslc" \
        -DCMAKE_EXE_LINKER_FLAGS="$PORTABLE_LINK" \
        -DCMAKE_SHARED_LINKER_FLAGS="$PORTABLE_LINK" \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DCMAKE_INSTALL_RPATH='$ORIGIN'
    cmake --build "$ROOT/build-fork-gpu" -j"$J"
    done_stage openems-fork-gpu
fi

# --- bundle -----------------------------------------------------------------
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/fork" "$BUNDLE/upstream" "$BUNDLE/lib" "$BUNDLE/xml"
cp -a "$ROOT/build-fork-gpu/openEMS" "$BUNDLE/fork/"
cp -a "$ROOT"/build-fork-gpu/libopenEMS.so* "$BUNDLE/fork/"
# The same upstream binary the CPU bundle shipped -- not a rebuild.
cp -a "$ROOT/build-upstream/openEMS" "$BUNDLE/upstream/"
cp -a "$ROOT"/build-upstream/libopenEMS.so* "$BUNDLE/upstream/"
cp -a "$DEPS"/lib/libCSXCAD.so* "$DEPS"/lib/libfparser.so* "$BUNDLE/lib/"
cp -a "$REPO/python/Tests/benchmark_fork_vs_upstream.py" "$BUNDLE/"
cp -a "$REPO/python/Tests/run_portable_bench_gpu.sh" "$BUNDLE/"
cp -a "$REPO"/python/Tests/models/*.xml "$BUNDLE/xml/"

echo
echo "=== bundle: $BUNDLE"
echo "--- fork/openEMS links (libvulkan.so.1 must be UNRESOLVED here and"
echo "    supplied by the target; libCSXCAD/libfparser come from lib/)"
ldd "$BUNDLE/fork/openEMS" || true
