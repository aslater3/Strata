// gemm_probe: oneMKL FP16 GEMM throughput on the A770 at a large shape and at the prompt path's per-expert shapes.
// C[M x N] = A[M x K] * B[K x N]^T style, fp16 in, fp32 accumulate/out (gemm_f16f16f32 is what the prefill uses).
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <chrono>
#include <cstdio>
#include <vector>

using clk = std::chrono::steady_clock;
namespace blas = oneapi::mkl::blas::column_major;

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    const size_t maxe = (size_t) 8192 * 8192;
    sycl::half* A = sycl::malloc_device<sycl::half>(maxe, q);
    sycl::half* B = sycl::malloc_device<sycl::half>(maxe, q);
    float* C = sycl::malloc_device<float>(maxe, q);
    q.memset(A, 0, maxe * 2); q.memset(B, 0, maxe * 2); q.wait();
    struct Shape { int M, N, K, reps; const char* what; int ta = 0, tb = 1; };
    std::vector<Shape> sh = {
        {8192, 8192, 8192, 5, "large square"},
        {1280, 312, 2560, 200, "ENGINE gate+up: W^T(1280x2560) X(2560x312)", 1, 0},
        {2560, 312, 640, 200, "ENGINE down: W^T(2560x640) X(640x312)", 1, 0},
        {1280, 1024, 2560, 100, "ENGINE gate+up, 1024 tokens", 1, 0},
        {1280, 64, 2560, 200, "ENGINE gate+up, 64 tokens", 1, 0},
        {1280, 312, 2560, 200, "W(1280x2560) row-major as N,N", 0, 0},
        {1280, 312, 2560, 200, "N,T", 0, 1},
        {312, 1280, 2560, 200, "swapped: X^T W (tokens as M), T,N", 1, 0},
        {312, 2560, 640, 200, "swapped down, T,N", 1, 0},
        {1280, 8192, 2560, 20, "ENGINE layout, 8192 tokens gate+up", 1, 0},
    };

    for (auto& s : sh) {
        if ((size_t) s.M * s.K > maxe || (size_t) s.N * s.K > maxe || (size_t) s.M * s.N > maxe) { std::printf("skip %s\n", s.what); continue; }
        auto run = [&] {
            // C(M,N) = A(M,K) * B(N,K)^T  (column major: A is M x K, B is N x K)
            using oneapi::mkl::transpose;
            const auto ta = s.ta ? transpose::trans : transpose::nontrans, tb = s.tb ? transpose::trans : transpose::nontrans;
            blas::gemm(q, ta, tb, s.M, s.N, s.K, 1.0f, A, s.ta ? s.K : s.M, B, s.tb ? s.N : s.K, 0.0f, C, s.M);
        };
        run(); q.wait();
        auto t0 = clk::now();
        for (int r = 0; r < s.reps; ++r) run();
        const double host_us = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / s.reps;   // submission only
        q.wait();
        const double sec = std::chrono::duration<double>(clk::now() - t0).count() / s.reps;
        std::printf("   host submit %.1f us/call, ", host_us);
        const double tf = 2.0 * s.M * s.N * s.K / sec / 1e12;
        std::printf("%-46s M=%6d N=%5d K=%5d: %8.1f us  %6.2f TFLOPS\n", s.what, s.M, s.N, s.K, sec * 1e6, tf);
    }
    return 0;
}
