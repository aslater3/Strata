// contend_probe: does the GPU's zero-copy read of pinned RAM and the CPU pool's RAM read add up, or share one budget?
//   contend_probe <cpu_threads>   CPU threads stream a 3 GiB pageable buffer; the GPU streams a 2 GiB pinned buffer
#include <sycl/sycl.hpp>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
#include <immintrin.h>

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }
constexpr size_t GiB = 1ull << 30;

static double cpu_read(const uint8_t* p, size_t n, int threads, double seconds) {
    std::atomic<bool> stop{false};
    std::vector<size_t> bytes(threads, 0);
    std::vector<std::thread> th;
    for (int t = 0; t < threads; ++t)
        th.emplace_back([&, t] {
            const size_t per = n / threads;
            const __m256i* s = (const __m256i*) (p + t * per);
            __m256i acc = _mm256_setzero_si256();
            size_t done = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                for (size_t i = 0; i < per / 32; i += 4) {
                    acc = _mm256_xor_si256(acc, _mm256_load_si256(s + i));
                    acc = _mm256_xor_si256(acc, _mm256_load_si256(s + i + 1));
                    acc = _mm256_xor_si256(acc, _mm256_load_si256(s + i + 2));
                    acc = _mm256_xor_si256(acc, _mm256_load_si256(s + i + 3));
                }
                done += per;
            }
            if (_mm256_testz_si256(acc, acc) == 2) std::printf("x");
            bytes[t] = done;
        });
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    stop = true;
    for (auto& t : th) t.join();
    size_t tot = 0;
    for (auto b : bytes) tot += b;
    return tot / seconds / 1e9;
}

int main(int argc, char** argv) {
    const int threads = argc > 1 ? std::atoi(argv[1]) : 7;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    const size_t hn = 2 * GiB, cn = 3 * GiB;
    uint8_t* h = sycl::malloc_host<uint8_t>(hn, q);
    std::memset(h, 1, hn);
    uint8_t* c = (uint8_t*) std::aligned_alloc(64, cn);
    std::memset(c, 2, cn);
    uint32_t* out = sycl::malloc_device<uint32_t>(1, q);
    const size_t words = hn / 16;
    auto gpu_pass = [&] {
        const sycl::uint4* src = (const sycl::uint4*) h;
        q.parallel_for(sycl::nd_range<1>(sycl::range<1>(65536 * 4), sycl::range<1>(256)), [=](sycl::nd_item<1> it) {
            uint32_t acc = 0;
            for (size_t i = it.get_global_id(0); i < words; i += it.get_global_range(0)) { auto v = src[i]; acc ^= v.x() ^ v.y() ^ v.z() ^ v.w(); }
            if (acc == 0x12345678u) out[0] = acc;
        });
    };
    gpu_pass(); q.wait();
    // alone
    auto t0 = clk::now(); for (int r = 0; r < 4; ++r) gpu_pass(); q.wait();
    const double gpu_alone = 4.0 * hn / secs(t0, clk::now()) / 1e9;
    const double cpu_alone = cpu_read(c, cn, threads, 2.0);
    // together: the GPU loops for ~2 s while the CPU threads read
    std::atomic<bool> go{true};
    double gpu_both = 0;
    std::thread g([&] {
        auto a = clk::now(); int passes = 0;
        while (secs(a, clk::now()) < 2.0) { gpu_pass(); q.wait(); ++passes; }
        gpu_both = passes * (double) hn / secs(a, clk::now()) / 1e9;
    });
    const double cpu_both = cpu_read(c, cn, threads, 2.0);
    g.join();
    std::printf("%d CPU threads: alone GPU %.1f GB/s, CPU %.1f GB/s | together GPU %.1f + CPU %.1f = %.1f GB/s\n", threads,
                gpu_alone, cpu_alone, gpu_both, cpu_both, gpu_both + cpu_both);
    return 0;
}
