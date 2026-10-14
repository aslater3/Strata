// pinned_probe: DG2 pinned host memory limits and GPU<->RAM bandwidth.
//   pinned_probe            device info, single malloc_host sizes, chunked totals, bandwidths
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }
constexpr size_t GiB = 1ull << 30;

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    auto d = q.get_device();
    std::printf("device: %s\n", d.get_info<sycl::info::device::name>().c_str());
    std::printf("global_mem %.2f GiB  max_mem_alloc %.2f GiB\n", d.get_info<sycl::info::device::global_mem_size>() / (double) GiB,
                d.get_info<sycl::info::device::max_mem_alloc_size>() / (double) GiB);
    std::printf("aspects: usm_host_allocations %d  usm_atomic_host_allocations %d  usm_shared_allocations %d  "
                "usm_atomic_shared_allocations %d  usm_system_allocations %d\n",
                d.has(sycl::aspect::usm_host_allocations), d.has(sycl::aspect::usm_atomic_host_allocations),
                d.has(sycl::aspect::usm_shared_allocations), d.has(sycl::aspect::usm_atomic_shared_allocations),
                d.has(sycl::aspect::usm_system_allocations));

    // 1. one malloc_host of growing size
    for (double g : {1.0, 2.0, 3.0, 3.5, 3.99, 4.0, 5.0, 6.0, 8.0}) {
        const size_t n = (size_t) (g * GiB);
        void* p = nullptr;
        try { p = sycl::malloc_host(n, q); } catch (...) { p = nullptr; }
        std::printf("single malloc_host %.2f GiB: %s\n", g, p ? "ok" : "FAILED");
        if (p) sycl::free(p, q);
    }
    // 2. chunked: as many 2 GiB chunks as fit up to 14 GiB
    std::vector<void*> ch;
    for (int i = 0; i < 7; ++i) {
        void* p = nullptr;
        try { p = sycl::malloc_host(2 * GiB, q); } catch (...) { p = nullptr; }
        if (!p) { std::printf("chunk %d of 2 GiB FAILED\n", i); break; }
        std::memset(p, 1, 2 * GiB);   // touch: really pinned/resident
        ch.push_back(p);
    }
    std::printf("chunked: %zu x 2 GiB = %zu GiB pinned\n", ch.size(), ch.size() * 2);

    // 3. bandwidths on the first chunk (2 GiB) and a device buffer
    const size_t n = 1 * GiB;
    uint8_t* dev = sycl::malloc_device<uint8_t>(n, q);
    std::vector<uint8_t> pageable(n, 2);
    auto bw = [&](const char* what, auto f) {
        f(); q.wait();   // warm
        double best = 1e9;
        for (int r = 0; r < 5; ++r) { auto t0 = clk::now(); f(); q.wait(); best = std::min(best, secs(t0, clk::now())); }
        std::printf("%-44s %6.2f GB/s\n", what, n / best / 1e9);
    };
    if (!ch.empty()) {
        uint8_t* h = (uint8_t*) ch[0];
        bw("memcpy pinned host -> device", [&] { q.memcpy(dev, h, n); });
        bw("memcpy device -> pinned host", [&] { q.memcpy(h, dev, n); });
        bw("memcpy pageable host -> device", [&] { q.memcpy(dev, pageable.data(), n); });
        // zero-copy: a kernel reads the pinned host buffer directly (what --pcie-frac relies on)
        uint64_t* out = sycl::malloc_device<uint64_t>(1, q);
        const size_t words = n / 16;
        auto rd = [&](const sycl::uint4* src) {
            q.parallel_for(sycl::nd_range<1>(sycl::range<1>(65536 * 4), sycl::range<1>(256)), [=](sycl::nd_item<1> it) {
                uint32_t acc = 0;
                for (size_t i = it.get_global_id(0); i < words; i += it.get_global_range(0)) { auto v = src[i]; acc ^= v.x() ^ v.y() ^ v.z() ^ v.w(); }
                if (acc == 0x12345678u) out[0] = acc;   // keep the loads
            });
        };
        bw("kernel reads pinned host (zero-copy)", [&] { rd((const sycl::uint4*) h); });
        bw("kernel reads device memory", [&] { rd((const sycl::uint4*) dev); });
        if (ch.size() >= 4) {   // zero-copy across 4 chunks in one kernel via a pointer table
            const sycl::uint4** tab = sycl::malloc_shared<const sycl::uint4*>(4, q);
            for (int i = 0; i < 4; ++i) tab[i] = (const sycl::uint4*) ch[i];
            const size_t per = (n / 4) / 16;
            bw("kernel reads 4 pinned chunks (pointer table)", [&] {
                q.parallel_for(sycl::nd_range<1>(sycl::range<1>(65536 * 4), sycl::range<1>(256)), [=](sycl::nd_item<1> it) {
                    uint32_t acc = 0;
                    for (size_t i = it.get_global_id(0); i < 4 * per; i += it.get_global_range(0)) {
                        auto v = tab[i / per][i % per]; acc ^= v.x() ^ v.y() ^ v.z() ^ v.w(); }
                    if (acc == 0x12345678u) out[0] = acc;
                });
            });
            sycl::free(tab, q);
        }
        sycl::free(out, q);
    }
    for (void* p : ch) sycl::free(p, q);
    sycl::free(dev, q);
    return 0;
}
