#!/bin/bash
#
# Build portable openEMS benchmark binaries (fork + upstream) for the
# fork-vs-upstream matrix (python/Tests/benchmark_fork_vs_upstream.py).
#
# Why this exists: the matrix is run on machines that are not the build host
# and must not be modified -- old distributions (down to CentOS 7 / glibc 2.17,
# kernel 3.10) with no openEMS dependencies installed. So everything is built
# from source inside a glibc-2.17 container (quay.io/pypa/manylinux2014_x86_64,
# GCC 10.2.1) and linked statically except for libc: the resulting bundle needs
# nothing on the target but a glibc >= 2.17 kernel >= 2.6.32 x86-64 system.
#
# Both builds -- this fork and the upstream merge base -- come out of the same
# container with the same compiler and the same dependency set, which is what
# the comparison requires.
#
# Usage:
#   python/Tests/build_portable_bench.sh            # build everything
#   ROOT=/somewhere python/Tests/build_portable_bench.sh
#
# Stages are stamped: rerunning skips what is already built. Remove a stamp in
# $ROOT/stamps to force a rebuild of that stage.
#
# Baseline ISA: nothing is compiled with -march=native. The only vector flags
# in the tree are the per-file "-mavx2 -mfma" the AVX2 engine sources carry,
# and GMP's CPU-specific assembly is disabled outright, so the bundle also runs
# on pre-Haswell hosts -- everything except the fork's AVX2 engine itself,
# which is an unconditional compile-time feature (see PERFORMANCE.md).
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
    mkdir -p "$ROOT"
    echo "=== building in $IMAGE ==="
    exec docker run --rm -i \
        -v "$ROOT:$ROOT" \
        -v "$REPO:$REPO" \
        -v "$UPSTREAM:$UPSTREAM" \
        -v "$CSXCAD_SRC:$CSXCAD_SRC" \
        -e IN_CONTAINER=1 -e ROOT="$ROOT" -e REPO="$REPO" \
        -e UPSTREAM="$UPSTREAM" -e CSXCAD_SRC="$CSXCAD_SRC" \
        -e HOME="$ROOT" \
        -w "$ROOT" "$IMAGE" bash "$REPO/python/Tests/build_portable_bench.sh"
fi

# ---------------------------------------------------------------- in container
SRC=$ROOT/src
DEPS=$ROOT/deps
STAMPS=$ROOT/stamps
BUNDLE=$ROOT/bundle
J=$(nproc)
mkdir -p "$SRC" "$DEPS" "$STAMPS" "$BUNDLE"

# Static everywhere, PIC everywhere (the static deps end up inside two shared
# libraries), and the GCC runtimes linked in so the target needs no libstdc++.
export CFLAGS="-O2 -fPIC"
export CXXFLAGS="-O2 -fPIC"
export LDFLAGS="-static-libstdc++ -static-libgcc"
export CMAKE_PREFIX_PATH=$DEPS
PORTABLE_LINK="-static-libstdc++ -static-libgcc -Wl,--enable-new-dtags"

stage() {  # stage <name> ; returns 1 if already done
    if [ -e "$STAMPS/$1" ]; then echo "=== $1: already built"; return 1; fi
    echo "=== $1"; return 0
}
done_stage() { touch "$STAMPS/$1"; }

fetch() {  # fetch <url> <file>
    [ -e "$SRC/$2" ] || curl -fsSL -o "$SRC/$2" "$1"
}

# --- toolchain --------------------------------------------------------------
# The image ships CMake 4, which refuses the "cmake_minimum_required(3.1...)"
# that openEMS, CSXCAD and fparser all declare. Fetch a 3.x CMake and a Ninja
# into $ROOT, which is the one directory that survives the container: anything
# installed into the image itself is gone on the next invocation, so every
# resumable stage has to find its tools here.
export PATH=$ROOT/tools/bin:$PATH
if [ ! -x "$ROOT/tools/bin/cmake" ]; then
    echo "=== toolchain"
    fetch https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz cmake-3.31.6.tar.gz
    mkdir -p "$ROOT/tools"
    tar -C "$ROOT/tools" --strip-components=1 -xf "$SRC/cmake-3.31.6.tar.gz"
fi
if [ ! -x "$ROOT/tools/bin/ninja" ]; then
    fetch https://github.com/ninja-build/ninja/releases/download/v1.12.1/ninja-linux.zip ninja-linux.zip
    (cd "$ROOT/tools/bin" && unzip -oq "$SRC/ninja-linux.zip")
fi
cmake --version | head -1
ninja --version
gcc --version | head -1

# --- gmp + mpfr (CGAL's arithmetic backends) --------------------------------
# --disable-assembly: GMP's configure otherwise selects assembly for the *build*
# host's microarchitecture, which would make the bundle crash on the older
# targets. GMP is not on any hot path here.
if stage gmp; then
    fetch https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz gmp-6.3.0.tar.xz
    rm -rf "$SRC/gmp-6.3.0" && tar -C "$SRC" -xf "$SRC/gmp-6.3.0.tar.xz"
    cd "$SRC/gmp-6.3.0"
    ./configure --prefix="$DEPS" --disable-shared --enable-static \
                --enable-cxx --disable-assembly --with-pic >/dev/null
    make -j"$J" >/dev/null && make install >/dev/null
    done_stage gmp
fi

if stage mpfr; then
    fetch https://www.mpfr.org/mpfr-4.2.1/mpfr-4.2.1.tar.xz mpfr-4.2.1.tar.xz
    rm -rf "$SRC/mpfr-4.2.1" && tar -C "$SRC" -xf "$SRC/mpfr-4.2.1.tar.xz"
    cd "$SRC/mpfr-4.2.1"
    ./configure --prefix="$DEPS" --with-gmp="$DEPS" --disable-shared \
                --enable-static --with-pic >/dev/null
    make -j"$J" >/dev/null && make install >/dev/null
    done_stage mpfr
fi

# --- boost (upstream openEMS needs the compiled libs; CGAL needs the headers) -
if stage boost; then
    fetch https://archives.boost.io/release/1.74.0/source/boost_1_74_0.tar.bz2 boost_1_74_0.tar.bz2
    rm -rf "$SRC/boost_1_74_0" && tar -C "$SRC" -xf "$SRC/boost_1_74_0.tar.bz2"
    cd "$SRC/boost_1_74_0"
    ./bootstrap.sh --prefix="$DEPS" \
        --with-libraries=thread,system,date_time,serialization,chrono,program_options >/dev/null
    ./b2 -j"$J" -d0 link=static runtime-link=shared threading=multi \
        variant=release cxxflags=-fPIC cflags=-fPIC install >/dev/null
    done_stage boost
fi

# --- CGAL (header-only; CSXCAD needs it) ------------------------------------
if stage cgal; then
    fetch https://github.com/CGAL/cgal/releases/download/v5.6.1/CGAL-5.6.1-library.tar.xz CGAL-5.6.1.tar.xz
    rm -rf "$SRC/CGAL-5.6.1" && tar -C "$SRC" -xf "$SRC/CGAL-5.6.1.tar.xz"
    cmake -S "$SRC/CGAL-5.6.1" -B "$SRC/CGAL-5.6.1/build" \
        -DCMAKE_INSTALL_PREFIX="$DEPS" -DCMAKE_BUILD_TYPE=Release \
        -DCGAL_HEADER_ONLY=ON >/dev/null
    cmake --install "$SRC/CGAL-5.6.1/build" >/dev/null
    done_stage cgal
fi

# --- tinyxml ----------------------------------------------------------------
# Built with TIXML_USE_STL, which is what every distribution ships and what
# CSXCAD/openEMS are written against; without it the API is a different one.
if stage tinyxml; then
    fetch https://downloads.sourceforge.net/project/tinyxml/tinyxml/2.6.2/tinyxml_2_6_2.tar.gz tinyxml_2_6_2.tar.gz
    rm -rf "$SRC/tinyxml" && tar -C "$SRC" -xf "$SRC/tinyxml_2_6_2.tar.gz"
    cd "$SRC/tinyxml"
    # Upstream leaves TIXML_USE_STL to the build system; distributions patch it
    # into the header so consumers cannot disagree with the library about it.
    # Do the same, or CSXCAD compiles against a different API than it links.
    sed -i 's|^#define TINYXML_INCLUDED|#define TINYXML_INCLUDED\n#define TIXML_USE_STL 1|' tinyxml.h
    grep -q '^#define TIXML_USE_STL 1' tinyxml.h
    for f in tinyxml tinyxmlparser tinyxmlerror tinystr; do
        g++ $CXXFLAGS -c $f.cpp -o $f.o
    done
    ar rcs libtinyxml.a tinyxml.o tinyxmlparser.o tinyxmlerror.o tinystr.o
    install -Dm644 libtinyxml.a "$DEPS/lib/libtinyxml.a"
    install -Dm644 tinyxml.h "$DEPS/include/tinyxml.h"
    install -Dm644 tinystr.h "$DEPS/include/tinystr.h"
    done_stage tinyxml
fi

# --- hdf5 -------------------------------------------------------------------
if stage hdf5; then
    fetch https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5_1.14.6.tar.gz hdf5-1.14.6.tar.gz
    rm -rf "$SRC/hdf5-hdf5_1.14.6" && tar -C "$SRC" -xf "$SRC/hdf5-1.14.6.tar.gz"
    cmake -S "$SRC/hdf5-hdf5_1.14.6" -B "$SRC/hdf5-hdf5_1.14.6/build" -G Ninja \
        -DCMAKE_INSTALL_PREFIX="$DEPS" -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DHDF5_BUILD_HL_LIB=ON -DHDF5_BUILD_TOOLS=OFF -DHDF5_BUILD_UTILS=OFF \
        -DHDF5_BUILD_EXAMPLES=OFF -DHDF5_BUILD_CPP_LIB=OFF \
        -DBUILD_TESTING=OFF -DHDF5_ENABLE_Z_LIB_SUPPORT=OFF \
        -DHDF5_ENABLE_SZIP_SUPPORT=OFF >/dev/null
    cmake --build "$SRC/hdf5-hdf5_1.14.6/build" -j"$J" >/dev/null
    cmake --install "$SRC/hdf5-hdf5_1.14.6/build" >/dev/null
    done_stage hdf5
fi

# --- VTK (only the four IO modules openEMS and CSXCAD ask for) --------------
if stage vtk; then
    fetch https://www.vtk.org/files/release/9.3/VTK-9.3.1.tar.gz VTK-9.3.1.tar.gz
    rm -rf "$SRC/VTK-9.3.1" && tar -C "$SRC" -xf "$SRC/VTK-9.3.1.tar.gz"
    cmake -S "$SRC/VTK-9.3.1" -B "$SRC/VTK-9.3.1/build" -G Ninja \
        -DCMAKE_INSTALL_PREFIX="$DEPS" -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DVTK_BUILD_TESTING=OFF -DVTK_BUILD_EXAMPLES=OFF \
        -DVTK_ENABLE_WRAPPING=OFF -DVTK_LEGACY_REMOVE=ON \
        -DVTK_GROUP_ENABLE_Rendering=DONT_WANT -DVTK_GROUP_ENABLE_Views=DONT_WANT \
        -DVTK_GROUP_ENABLE_Web=DONT_WANT -DVTK_GROUP_ENABLE_Imaging=DONT_WANT \
        -DVTK_GROUP_ENABLE_Qt=DONT_WANT -DVTK_GROUP_ENABLE_MPI=DONT_WANT \
        -DVTK_GROUP_ENABLE_StandAlone=DONT_WANT \
        -DVTK_MODULE_ENABLE_VTK_IOXML=YES \
        -DVTK_MODULE_ENABLE_VTK_IOGeometry=YES \
        -DVTK_MODULE_ENABLE_VTK_IOLegacy=YES \
        -DVTK_MODULE_ENABLE_VTK_IOPLY=YES >/dev/null
    cmake --build "$SRC/VTK-9.3.1/build" -j"$J" >/dev/null
    cmake --install "$SRC/VTK-9.3.1/build" >/dev/null
    done_stage vtk
fi

# --- fparser ----------------------------------------------------------------
if stage fparser; then
    [ -d "$SRC/fparser" ] || git clone -q --depth 1 https://github.com/thliebig/fparser "$SRC/fparser"
    cmake -S "$SRC/fparser" -B "$SRC/fparser/build" -G Ninja \
        -DCMAKE_INSTALL_PREFIX="$DEPS" -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON >/dev/null
    cmake --build "$SRC/fparser/build" -j"$J" >/dev/null
    cmake --install "$SRC/fparser/build" >/dev/null
    done_stage fparser
fi

# --- CSXCAD (shared; both openEMS builds link the same one) -----------------
if stage csxcad; then
    cmake -S "$CSXCAD_SRC" -B "$ROOT/build-csxcad" \
        -DCMAKE_INSTALL_PREFIX="$DEPS" -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$DEPS" \
        -DFPARSER_ROOT_DIR="$DEPS" -DTinyXML_ROOT_DIR="$DEPS" \
        -DHDF5_ROOT="$DEPS" -DHDF5_USE_STATIC_LIBRARIES=ON \
        -DCGAL_DIR="$DEPS/lib/cmake/CGAL" \
        -DGMP_INCLUDE_DIR="$DEPS/include" -DGMP_LIBRARIES="$DEPS/lib/libgmp.a" \
        -DMPFR_INCLUDE_DIR="$DEPS/include" -DMPFR_LIBRARIES="$DEPS/lib/libmpfr.a" \
        -DBOOST_ROOT="$DEPS" -DBoost_USE_STATIC_LIBS=ON \
        -DCMAKE_EXE_LINKER_FLAGS="$PORTABLE_LINK" \
        -DCMAKE_SHARED_LINKER_FLAGS="$PORTABLE_LINK" \
        -DCMAKE_INSTALL_RPATH='$ORIGIN' >/dev/null
    cmake --build "$ROOT/build-csxcad" -j"$J" >/dev/null
    cmake --install "$ROOT/build-csxcad" >/dev/null
    done_stage csxcad
fi

# --- openEMS: this fork, and the upstream merge base ------------------------
build_openems() {  # build_openems <srcdir> <builddir> <extra cmake args...>
    local src=$1 bld=$2; shift 2
    cmake -S "$src" -B "$bld" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$DEPS" \
        -DCSXCAD_ROOT_DIR="$DEPS" -DFPARSER_ROOT_DIR="$DEPS" \
        -DTinyXML_ROOT_DIR="$DEPS" \
        -DHDF5_ROOT="$DEPS" -DHDF5_USE_STATIC_LIBRARIES=ON \
        -DBOOST_ROOT="$DEPS" -DBoost_USE_STATIC_LIBS=ON \
        -DWITH_GPU=OFF \
        -DCMAKE_EXE_LINKER_FLAGS="$PORTABLE_LINK" \
        -DCMAKE_SHARED_LINKER_FLAGS="$PORTABLE_LINK" \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
        -DCMAKE_INSTALL_RPATH='$ORIGIN' "$@"
    cmake --build "$bld" -j"$J"
}

if stage openems-fork; then
    build_openems "$REPO" "$ROOT/build-fork"
    done_stage openems-fork
fi

if stage openems-upstream; then
    build_openems "$UPSTREAM" "$ROOT/build-upstream"
    done_stage openems-upstream
fi

# --- bundle -----------------------------------------------------------------
# Layout matches what benchmark_fork_vs_upstream.py expects: a "build directory"
# per side holding the openEMS binary, plus one directory holding libCSXCAD.
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/fork" "$BUNDLE/upstream" "$BUNDLE/lib" "$BUNDLE/xml"
cp -a "$ROOT/build-fork/openEMS" "$BUNDLE/fork/"
cp -a "$ROOT"/build-fork/libopenEMS.so* "$BUNDLE/fork/"
cp -a "$ROOT/build-upstream/openEMS" "$BUNDLE/upstream/"
cp -a "$ROOT"/build-upstream/libopenEMS.so* "$BUNDLE/upstream/"
cp -a "$DEPS"/lib/libCSXCAD.so* "$DEPS"/lib/libfparser.so* "$BUNDLE/lib/"
cp -a "$REPO/python/Tests/benchmark_fork_vs_upstream.py" "$BUNDLE/"
cp -a "$REPO/python/Tests/run_portable_bench.sh" "$BUNDLE/"
# The models travel with the bundle rather than being regenerated per host:
# every machine in PERFORMANCE.md has to run the same bytes, and a host with no
# CSXCAD/openEMS Python bindings installed could not regenerate them anyway.
cp -a "$REPO"/python/Tests/models/*.xml "$BUNDLE/xml/"

echo
echo "=== bundle: $BUNDLE"
for b in "$BUNDLE/fork/openEMS" "$BUNDLE/upstream/openEMS"; do
    echo "--- $b"
    ldd "$b" || true
done
