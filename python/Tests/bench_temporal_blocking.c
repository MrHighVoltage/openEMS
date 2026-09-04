/*
 * Temporal-blocking feasibility benchmark and reference schedule
 * (candidate C2 in OPTIMIZATIONS.md).
 *
 * Build:  gcc -O3 -march=native -fopenmp bench_temporal_blocking.c -o bench_tb
 * Verify: ./bench_tb verify <timesteps> <W> <k> <threads>
 * Time:   ./bench_tb flat|trap <timesteps> <W> <k> <threads>
 *         GX=224 GY=224 GZ=224 ./bench_tb trap 40 32 8 8
 * Traffic (difference two run lengths so the init phase drops out):
 *         taskset -c 0-15 perf stat -e cpu_core/longest_lat_cache.miss/ ./bench_tb ...
 *
 * This reproduces the *access pattern* of Engine_AVX2_Multithread -- two
 * arrays of three components in N-I-J-K layout with z contiguous, the E pass
 * reading H at (x,y,z),(x-1,y,z),(x,y-1,z),(x,y,z-1) and the H pass the mirror
 * image -- but not its physics. Values are meaningless; only the loads, stores
 * and their order matter, and those are what set cache behaviour. The flat
 * schedule reproduces the engine's measured 31 B/cell of DRAM traffic and its
 * ~900 MC/s to within ~2%, which is the evidence that the proxy is faithful.
 *
 * `trap` is the trapezoidal schedule intended for the engine, and `verify`
 * checks it produces bit-identical results to `flat` -- the property the port
 * has to preserve.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

static int NX, NY, NZ;
static long N;
static float *E, *H;

#define IDX(c,x,y,z) ((long)(c)*N + (long)(x)*NY*NZ + (long)(y)*NZ + (z))

static void updateE_plane(int x)
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

static void updateH_plane(int x)
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

/* Flat sweep: the whole grid, one timestep at a time. What the engine does. */
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

/*
 * One trapezoid, k steps. [A0,B0) is the range on which H is valid at the
 * block's start time. dir=+1 narrows (the cores of pass 1); dir=-1 widens
 * (the wedges of pass 2, which fill what pass 1 left behind).
 *
 * With RH_t = [a,b) the range where H is valid at time t:
 *     E[t+1] is computable on [a+1, b)      (it reads H at x and x-1)
 *     H[t+1] is computable on [a+1, b-1)    (it reads E[t+1] at x and x+1)
 * so H narrows by one per side per step.
 *
 * The E *sweep* range differs between the two directions, and getting this
 * wrong is the easy mistake: the passes must partition the domain so that
 * every cell is updated exactly once per timestep. A narrowing tile leaves
 * gaps of width 2t+1 in E and 2t+2 in H at each tile boundary, so the widening
 * wedge sweeps E on [An+1, Bn) -- one cell narrower on each side than the
 * symmetric-looking [An, Bn+1), which would update those cells twice.
 *
 * Requires W >= 2k so that a wedge's dependencies still reach into the
 * neighbouring cores.
 */
static void trapezoid(int A0, int B0, int k, int dir)
{
    int A = A0, B = B0;
    if (A < 0) A = 0;
    if (B > NX) B = NX;

    for (int t = 0; t < k; t++) {
        /* The slope exists only because data outside [A,B) is at the wrong
         * time. At a domain edge there is no outside -- updateE_plane clamps
         * x-1 to 0 and updateH_plane clamps x+1 to NX-1 -- so the edge is
         * held, not sloped, or the cells there never advance. */
        int An = (A <= 0)  ? 0  : A + dir;
        int Bn = (B >= NX) ? NX : B - dir;
        if (An < 0) An = 0;
        if (Bn > NX) Bn = NX;

        int eLo, eHi;
        if (dir > 0) { eLo = An;     eHi = (Bn >= NX) ? NX : Bn + 1; }
        else         { eLo = (An <= 0) ? 0 : An + 1;
                       eHi = (Bn >= NX) ? NX : Bn; }
        if (eHi > NX) eHi = NX;

        #pragma omp for schedule(static)
        for (int x = eLo; x < eHi; x++) updateE_plane(x);
        #pragma omp for schedule(static)
        for (int x = An; x < Bn; x++) updateH_plane(x);

        A = An; B = Bn;
    }
}

/* Trapezoidal temporal blocking: advance the grid k timesteps per outer pass,
 * so a tile's working set is touched k times per DRAM round trip. */
static void run_trap(int T, int W, int k)
{
    if (W < 2 * k) { fprintf(stderr, "need W >= 2k\n"); exit(1); }
    for (int t0 = 0; t0 < T; t0 += k) {
        int kk = (T - t0 < k) ? (T - t0) : k;
        #pragma omp parallel
        {
            for (int x0 = 0; x0 < NX; x0 += W) trapezoid(x0, x0 + W, kk, +1);
            for (int c  = W; c  < NX; c  += W) trapezoid(c,  c,      kk, -1);
        }
    }
}

static void init_fields(int pattern)
{
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < 3*N; i++) {
        if (pattern) { E[i] = (float)((i * 2654435761u) % 1000) * 1e-4f;
                       H[i] = (float)((i * 40503u) % 997) * 1e-4f; }
        else         { E[i] = 1e-3f; H[i] = 2e-3f; }
    }
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "verify";
    int T  = argc > 2 ? atoi(argv[2]) : 16;
    int W  = argc > 3 ? atoi(argv[3]) : 32;
    int k  = argc > 4 ? atoi(argv[4]) : 8;
    int nt = argc > 5 ? atoi(argv[5]) : 8;

    NX = getenv("GX") ? atoi(getenv("GX")) : 160;
    NY = getenv("GY") ? atoi(getenv("GY")) : 128;
    NZ = getenv("GZ") ? atoi(getenv("GZ")) : 192;
    N  = (long)NX * NY * NZ;
    omp_set_num_threads(nt);

    size_t bytes = 3ul * N * sizeof(float);
    E = aligned_alloc(64, bytes);
    H = aligned_alloc(64, bytes);

    if (strcmp(mode, "verify") == 0) {
        float *Ef = aligned_alloc(64, bytes), *Hf = aligned_alloc(64, bytes);
        init_fields(1);
        run_flat(T);
        memcpy(Ef, E, bytes); memcpy(Hf, H, bytes);
        init_fields(1);
        run_trap(T, W, k);
        long bad = 0; double worst = 0;
        for (long i = 0; i < 3*N; i++) {
            if (E[i] != Ef[i]) { bad++; double d = fabs((double)E[i] - Ef[i]); if (d > worst) worst = d; }
            if (H[i] != Hf[i]) { bad++; double d = fabs((double)H[i] - Hf[i]); if (d > worst) worst = d; }
        }
        printf("VERIFY T=%d W=%d k=%d nt=%d : %s (mismatches=%ld worst=%.3e)\n",
               T, W, k, nt, bad ? "DIFFERS" : "BIT-IDENTICAL", bad, worst);
        return bad != 0;
    }

    init_fields(0);
    run_flat(2);                        /* warm-up, excluded by differencing */
    double t0 = omp_get_wtime();
    if (strcmp(mode, "flat") == 0) run_flat(T); else run_trap(T, W, k);
    double el = omp_get_wtime() - t0;

    double sink = 0;
    for (long i = 0; i < 3*N; i += 1000003) sink += E[i] + H[i];
    printf("mode=%-5s T=%-4d W=%-3d k=%-3d nt=%-2d  %.3f s  %.1f MC/s  (sink %.4e)\n",
           mode, T, W, k, nt, el, (double)N * T / el / 1e6, sink);
    return 0;
}
