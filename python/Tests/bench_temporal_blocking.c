/*
 * Temporal-blocking feasibility benchmark (candidate C2 in OPTIMIZATIONS.md).
 *
 * Build:   gcc -O3 -march=native -fopenmp bench_temporal_blocking.c -o bench_tb
 * Run:     ./bench_tb flat|blocked <timesteps> <slab_width> <k> <threads>
 *          GX=224 GY=224 GZ=224 ./bench_tb blocked 40 16 8 8
 *
 * DRAM traffic (difference two run lengths to drop the init phase):
 *   taskset -c 0-15 perf stat -e cpu_core/longest_lat_cache.miss/ ./bench_tb ...
 *
 * Question: on this host, does temporally blocking an FDTD-shaped sweep over
 * x-slabs actually cut DRAM traffic by ~k, or do halo re-reads eat it?
 *
 * This reproduces the *memory access pattern* of Engine_AVX2_Multithread --
 * two arrays (E,H) of 3 components in N-I-J-K layout, z contiguous, the E pass
 * reading H at (x,y,z),(x-1,y,z),(x,y-1,z),(x,y,z-1) and the H pass the mirror
 * image -- but not its physics. Values here are meaningless; only the loads,
 * stores and their order matter, and those are what set cache behaviour. A
 * correct trapezoidal implementation would issue the same accesses.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static int NX, NY, NZ;
static long N;
static float *E, *H;

#define IDX(c,x,y,z) ((long)(c)*N + (long)(x)*NY*NZ + (long)(y)*NZ + (z))

static inline void updateE_plane(int x)
{
    int xm = x > 0 ? x - 1 : 0;
    for (int c = 0; c < 3; c++) {
        int c2 = (c + 1) % 3;
        for (int y = 0; y < NY; y++) {
            int ym = y > 0 ? y - 1 : 0;
            const float *h  = &H[IDX(c,  x,  y,  0)];
            const float *hx = &H[IDX(c,  xm, y,  0)];
            const float *hy = &H[IDX(c2, x,  ym, 0)];
            const float *hz = &H[IDX(c2, x,  y,  0)];
            float *e = &E[IDX(c, x, y, 0)];
            for (int z = 1; z < NZ; z++)
                e[z] = 0.999f * e[z] + 0.001f * (h[z] - hx[z] + hz[z] - hy[z-1]);
        }
    }
}

static inline void updateH_plane(int x)
{
    int xp = x < NX - 1 ? x + 1 : NX - 1;
    for (int c = 0; c < 3; c++) {
        int c2 = (c + 1) % 3;
        for (int y = 0; y < NY; y++) {
            int yp = y < NY - 1 ? y + 1 : NY - 1;
            const float *e  = &E[IDX(c,  x,  y,  0)];
            const float *ex = &E[IDX(c,  xp, y,  0)];
            const float *ey = &E[IDX(c2, x,  yp, 0)];
            const float *ez = &E[IDX(c2, x,  y,  0)];
            float *h = &H[IDX(c, x, y, 0)];
            for (int z = 0; z < NZ - 1; z++)
                h[z] = 0.999f * h[z] + 0.001f * (e[z] - ex[z] + ez[z] - ey[z+1]);
        }
    }
}

/* Flat sweep: whole grid, one timestep at a time. What the engine does today. */
static void run_flat(int T)
{
    for (int t = 0; t < T; t++) {
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int x = 0; x < NX; x++) updateE_plane(x);
            #pragma omp for schedule(static)
            for (int x = 0; x < NX; x++) updateH_plane(x);
        }
    }
}

/* Temporally blocked: advance a slab of W x-planes through k timesteps before
 * moving on, so the slab's working set is touched k times per DRAM round trip.
 * Halo planes on each side are re-swept every inner timestep, which is the
 * redundant work the candidate is being tested for. */
static void run_blocked(int T, int W, int k)
{
    for (int t0 = 0; t0 < T; t0 += k) {
        int kk = (T - t0 < k) ? (T - t0) : k;
        for (int x0 = 0; x0 < NX; x0 += W) {
            int x1 = x0 + W; if (x1 > NX) x1 = NX;
            for (int t = 0; t < kk; t++) {
                int lo = x0 - (kk - t); if (lo < 0) lo = 0;
                int hi = x1 + (kk - t); if (hi > NX) hi = NX;
                #pragma omp parallel
                {
                    #pragma omp for schedule(static)
                    for (int x = lo; x < hi; x++) updateE_plane(x);
                    #pragma omp for schedule(static)
                    for (int x = lo; x < hi; x++) updateH_plane(x);
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "flat";
    int T  = argc > 2 ? atoi(argv[2]) : 20;
    int W  = argc > 3 ? atoi(argv[3]) : 48;
    int k  = argc > 4 ? atoi(argv[4]) : 4;
    int nt = argc > 5 ? atoi(argv[5]) : 8;
    NX = getenv("GX")?atoi(getenv("GX")):160; NY = getenv("GY")?atoi(getenv("GY")):128; NZ = getenv("GZ")?atoi(getenv("GZ")):192; N = (long)NX*NY*NZ;
    omp_set_num_threads(nt);

    size_t bytes = 3ul * N * sizeof(float);
    E = aligned_alloc(64, bytes); H = aligned_alloc(64, bytes);
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < 3*N; i++) { E[i] = 1e-3f; H[i] = 2e-3f; }

    /* warm-up, excluded from the differencing by using two T values */
    run_flat(2);

    double t0 = omp_get_wtime();
    if (strcmp(mode, "flat") == 0) run_flat(T); else run_blocked(T, W, k);
    double el = omp_get_wtime() - t0;

    double sink = 0; for (long i = 0; i < 3*N; i += 1000003) sink += E[i] + H[i];
    double mcs = (double)N * T / el / 1e6;
    printf("mode=%-7s T=%-4d W=%-3d k=%d nt=%d  %.3f s  %.1f MC/s  (sink %.3e)\n",
           mode, T, W, k, nt, el, mcs, sink);
    return 0;
}
