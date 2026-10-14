// moe_layer_probe: one prompt layer's routed experts as the A770 prompt path runs them (sycl/src/prefill/prefill.cpp),
// against lighter dispatch structures. 256 experts, 79,872 routed rows (8K tokens x top-10), uneven token counts.
//   A  engine: per expert dequant kernel (writes the fp16 gate/up and down), staging copies of W and X (the oneMKL
//      >4 GiB workaround), gemm gate/up, swiglu, staging copy of W_down, gemm down
//   B  as A without the staging copies
//   C  as B without the dequant kernel (weights already fp16)
//   D  one gemm gate/up + swiglu + one gemm down over all 79,872 rows with one expert's weights (the dense lower bound)
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using clk = std::chrono::steady_clock;
namespace blas = oneapi::mkl::blas::column_major;
using oneapi::mkl::transpose;

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    const int E = 256, H = 2560, F = 640, ROWS = 7987 * 10;
    // uneven routing: log-normal counts scaled to ROWS
    std::mt19937 rng(7);
    std::lognormal_distribution<double> ln(0.0, 1.0);
    std::vector<double> w(E); double sw = 0; for (auto& x : w) { x = ln(rng); sw += x; }
    std::vector<int> cnt(E); int tot = 0; for (int e = 0; e < E; ++e) { cnt[e] = std::max(1, (int) std::lround(w[e] / sw * ROWS)); tot += cnt[e]; }
    std::vector<int> off(E + 1, 0); for (int e = 0; e < E; ++e) off[e + 1] = off[e] + cnt[e];
    std::printf("rows %d, per expert min %d max %d mean %.0f\n", tot, *std::min_element(cnt.begin(), cnt.end()),
                *std::max_element(cnt.begin(), cnt.end()), tot / (double) E);
    sycl::half* X = sycl::malloc_device<sycl::half>((size_t) tot * H, q);        // gathered rows
    float* GU = sycl::malloc_device<float>((size_t) tot * 2 * F, q);
    sycl::half* Hh = sycl::malloc_device<sycl::half>((size_t) tot * F, q);
    float* Dm = sycl::malloc_device<float>((size_t) tot * H, q);
    sycl::half* Wgu = sycl::malloc_device<sycl::half>((size_t) 2 * F * H, q);    // one expert's dequantized weights
    sycl::half* Wd = sycl::malloc_device<sycl::half>((size_t) H * F, q);
    uint8_t* blob = sycl::malloc_device<uint8_t>((size_t) 2 << 20, q);           // a ~2 MB quantized blob
    uint8_t* stage = sycl::malloc_device<uint8_t>((size_t) 2 * F * H * 2 + (size_t) 4096 * H * 2, q);
    q.memset(X, 0, (size_t) tot * H * 2); q.memset(Wgu, 0, (size_t) 2 * F * H * 2); q.memset(Wd, 0, (size_t) H * F * 2);
    q.memset(blob, 1, 2 << 20); q.wait();
    auto dequant = [&] {   // reads a 2 MB blob, writes 9.8 MB of fp16 (the work of iq_dequant_expert_f16)
        const size_t n = (size_t) 3 * F * H;
        q.parallel_for(sycl::range<1>(n / 8), [=](sycl::id<1> i) {
            const uint8_t b = blob[(i[0] * 8) % (2 << 20)];
            sycl::half* dst = i[0] * 8 < (size_t) 2 * F * H ? Wgu + i[0] * 8 : Wd + (i[0] * 8 - (size_t) 2 * F * H);
            for (int j = 0; j < 8; ++j) dst[j] = (sycl::half) (float) (b + j);
        });
    };
    auto swiglu = [&](int o, int n) {
        q.parallel_for(sycl::range<1>((size_t) n * F), [=](sycl::id<1> i) {
            const size_t r = i[0] / F, c = i[0] % F;
            const float g = GU[(size_t) (o + r) * 2 * F + c], u = GU[(size_t) (o + r) * 2 * F + F + c];
            Hh[(size_t) (o + r) * F + c] = (sycl::half) (g / (1.0f + sycl::exp(-g)) * u);
        });
    };
    auto layer = [&](int mode) {
        for (int e = 0; e < E; ++e) {
            const int o = off[e], n = cnt[e];
            if (mode <= 1) dequant();
            const sycl::half* wg = Wgu; const sycl::half* x = X + (size_t) o * H;
            if (mode == 0) { q.memcpy(stage, Wgu, (size_t) 2 * F * H * 2); q.memcpy(stage + (size_t) 2 * F * H * 2, x, (size_t) std::min(n, 4096) * H * 2);
                             wg = (const sycl::half*) stage; x = (const sycl::half*) (stage + (size_t) 2 * F * H * 2); }
            blas::gemm(q, transpose::trans, transpose::nontrans, 2 * F, n, H, 1.0f, wg, H, x, H, 0.0f, GU + (size_t) o * 2 * F, 2 * F);
            swiglu(o, n);
            const sycl::half* wd = Wd;
            if (mode == 0) { q.memcpy(stage, Wd, (size_t) H * F * 2); wd = (const sycl::half*) stage; }
            blas::gemm(q, transpose::trans, transpose::nontrans, H, n, F, 1.0f, wd, F, Hh + (size_t) o * F, F, 0.0f, Dm + (size_t) o * H, H);
        }
    };
    auto dense = [&] {
        blas::gemm(q, transpose::trans, transpose::nontrans, 2 * F, tot, H, 1.0f, Wgu, H, X, H, 0.0f, GU, 2 * F);
        swiglu(0, tot);
        blas::gemm(q, transpose::trans, transpose::nontrans, H, tot, F, 1.0f, Wd, F, Hh, F, 0.0f, Dm, H);
    };
    const char* names[3] = {"A engine (dequant + staging + per-expert gemms)", "B no staging copies", "C no staging, no dequant"};
    for (int mode = 0; mode < 3; ++mode) {
        layer(mode); q.wait();
        double best = 1e9;
        for (int r = 0; r < 5; ++r) { auto t0 = clk::now(); layer(mode); q.wait(); best = std::min(best, std::chrono::duration<double, std::milli>(clk::now() - t0).count()); }
        std::printf("%-52s %7.2f ms per layer  (x48 layers = %.2f s)\n", names[mode], best, best * 48 / 1000);
    }
    // E: oneMKL grouped batch (one call per projection for all 256 experts, each its own group of size 1)
    {
        std::vector<transpose> ta(E, transpose::trans), tb(E, transpose::nontrans);
        std::vector<int64_t> m1(E, 2 * F), n1(E), k1(E, H), lda1(E, H), ldb1(E, H), ldc1(E, 2 * F), gs(E, 1);
        std::vector<int64_t> m2(E, H), k2(E, F), lda2(E, F), ldb2(E, F), ldc2(E, H);
        std::vector<float> al(E, 1.0f), be(E, 0.0f);
        std::vector<const sycl::half*> a1(E), b1(E), a2(E), b2(E); std::vector<float*> c1(E), c2(E);
        for (int e = 0; e < E; ++e) {
            n1[e] = cnt[e];
            a1[e] = Wgu; b1[e] = X + (size_t) off[e] * H; c1[e] = GU + (size_t) off[e] * 2 * F;
            a2[e] = Wd; b2[e] = Hh + (size_t) off[e] * F; c2[e] = Dm + (size_t) off[e] * H;
        }
        auto grouped = [&] {
            blas::gemm_batch(q, ta.data(), tb.data(), m1.data(), n1.data(), k1.data(), al.data(), a1.data(), lda1.data(),
                             b1.data(), ldb1.data(), be.data(), c1.data(), ldc1.data(), (int64_t) E, gs.data());
            swiglu(0, tot);
            blas::gemm_batch(q, ta.data(), tb.data(), m2.data(), n1.data(), k2.data(), al.data(), a2.data(), lda2.data(),
                             b2.data(), ldb2.data(), be.data(), c2.data(), ldc2.data(), (int64_t) E, gs.data());
        };
        try {
            grouped(); q.wait();
            double gb = 1e9;
            for (int r = 0; r < 5; ++r) { auto t0 = clk::now(); grouped(); q.wait(); gb = std::min(gb, std::chrono::duration<double, std::milli>(clk::now() - t0).count()); }
            std::printf("%-52s %7.2f ms per layer  (x48 layers = %.2f s)\n", "E oneMKL grouped gemm_batch, no dequant", gb, gb * 48 / 1000);
        } catch (const std::exception& ex) { std::printf("E gemm_batch failed: %s\n", ex.what()); }
    }
    dense(); q.wait();
    double best = 1e9;
    for (int r = 0; r < 5; ++r) { auto t0 = clk::now(); dense(); q.wait(); best = std::min(best, std::chrono::duration<double, std::milli>(clk::now() - t0).count()); }
    const double fl = 2.0 * tot * (2.0 * F * H + (double) H * F);
    std::printf("%-52s %7.2f ms per layer  (x48 layers = %.2f s, %.1f TFLOPS)\n", "D one gemm pair over all rows (lower bound)", best, best * 48 / 1000, fl / best / 1e9);
    return 0;
}
