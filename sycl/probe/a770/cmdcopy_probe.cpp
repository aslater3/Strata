// cmdcopy_probe: a single-graph data plane built from commands instead of loads inside a running kernel.
// For every round, queued up front on ONE in-order queue (as a graph would hold them):
//   spin kernel (waits for the host's flag >= i, uncached-hint load)  ->  memcpy host block -> device block
//   ->  check kernel (reads the device block, counts stale words, acks with an uncached-hint store)
// The host writes block i, raises the flag, waits for the ack.
//   cmdcopy_probe <rounds> <KiB> [graph]   graph: record the 3*rounds commands into a SYCL graph and run that instead
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

namespace exi = sycl::ext::intel::experimental; namespace exo = sycl::ext::oneapi::experimental;
using uc_read = decltype(exo::properties(exi::read_hint<exi::cache_control<exi::cache_mode::uncached, exo::cache_level::L1, exo::cache_level::L3>>));
using uc_write = decltype(exo::properties(exi::write_hint<exi::cache_control<exi::cache_mode::uncached, exo::cache_level::L1, exo::cache_level::L3>>));
static inline uint32_t ucload(const uint32_t* p) {
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    exo::annotated_ptr<uint32_t, uc_read> u(const_cast<uint32_t*>(p)); return u[0]; }
static inline void ucstore(uint32_t* p, uint32_t v) { exo::annotated_ptr<uint32_t, uc_write> u(p); u[0] = v; }
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
    const int R = argc > 1 ? std::atoi(argv[1]) : 500;
    const size_t kib = argc > 2 ? (size_t) std::atoi(argv[2]) : 256;
    const bool use_graph = argc > 3 && std::string(argv[3]) == "graph";
    const size_t words = kib * 1024 / 8;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    uint64_t* hblk = sycl::malloc_host<uint64_t>(words, q);
    uint64_t* dblk = sycl::malloc_device<uint64_t>(words, q);
    uint32_t* flag = sycl::malloc_host<uint32_t>(16, q);
    uint32_t* ack = flag + 8;
    uint32_t* stale = sycl::malloc_host<uint32_t>(2, q);
    std::memset(hblk, 0, words * 8); std::memset(flag, 0, 64); stale[0] = stale[1] = 0;
    q.memset(dblk, 0, words * 8).wait();
    constexpr uint32_t BOUND = 20u * 1000u * 1000u;
    auto record = [&](sycl::queue& qq) {
        for (uint32_t i = 1; i <= (uint32_t) R; ++i) {
            qq.single_task([=] { uint32_t n = 0; while (ucload(flag) < i && n < BOUND) ++n; if (n >= BOUND) stale[1] += 1; });
            qq.memcpy(dblk, hblk, words * 8);
            qq.parallel_for(sycl::nd_range<1>(65536, 256), [=](sycl::nd_item<1> it) {
                uint32_t mine = 0;
                for (size_t w = it.get_global_id(0); w < words; w += it.get_global_range(0))
                    if (dblk[w] != (uint64_t) i * 0x9E3779B97F4A7C15ull + w) ++mine;
                if (mine) sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(stale[0]).fetch_add(mine);
            });
            qq.single_task([=] { sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system); ucstore(ack, i); });
        }
    };
    namespace sx = sycl::ext::oneapi::experimental;
    auto t0 = clk::now();
    if (use_graph) {
        sx::command_graph g{q.get_context(), q.get_device()};
        g.begin_recording(q); record(q); g.end_recording(q);
        auto ex = g.finalize();
        t0 = clk::now();
        q.ext_oneapi_graph(ex);
    } else {
        record(q);
    }
    double host_us = 0; int done = 0;
    for (uint32_t i = 1; i <= (uint32_t) R; ++i) {
        auto a = clk::now();
        for (size_t w = 0; w < words; ++w) hblk[w] = (uint64_t) i * 0x9E3779B97F4A7C15ull + w;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        std::atomic_ref<uint32_t>(*flag).store(i, std::memory_order_release);
        host_us += std::chrono::duration<double, std::micro>(clk::now() - a).count();
        auto w0 = clk::now();
        while (std::atomic_ref<uint32_t>(*ack).load(std::memory_order_acquire) < i)
            if (std::chrono::duration<double>(clk::now() - w0).count() > 3.0) { std::printf("ack %u never came\n", i); goto out; }
        ++done;
    }
out:
    q.wait();
    const double tot = std::chrono::duration<double, std::micro>(clk::now() - t0).count();
    std::printf("%s %zu KiB x %d rounds (%d done): stale words %u, spin timeouts %u, %.1f us per round (host %.1f us)\n",
                use_graph ? "graph" : "queue", kib, R, done, stale[0], stale[1], tot / std::max(1, done), host_us / std::max(1, done));
    return 0;
}
