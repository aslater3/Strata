// rows_probe: the data plane of a single-graph window on DG2. Each round the host writes a block of "CPU expert rows"
// into host USM and raises a flag; a running kernel (one work-group) waits for the flag with an uncached load, reads the
// block, checks every word, and acks with an uncached store. Measures stale reads and the time per round.
//   rows_probe plain|uc <rounds> <KiB per round> [rotate]
//     plain: the block is read with ordinary (cached) loads; uc: with uncached-hint loads
//     rotate: each round uses a different block of a ring of 8 (as a per-layer row slot would)
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
static inline uint32_t ucload(const uint32_t* p) { sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system); exo::annotated_ptr<uint32_t, uc_read> u(const_cast<uint32_t*>(p)); return u[0]; }
static inline uint64_t ucload64(const uint64_t* p) { exo::annotated_ptr<uint64_t, uc_read> u(const_cast<uint64_t*>(p)); return u[0]; }
static inline void ucstore(uint32_t* p, uint32_t v) { exo::annotated_ptr<uint32_t, uc_write> u(p); u[0] = v; }
using clk = std::chrono::steady_clock;

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "plain";
    const int R = argc > 2 ? std::atoi(argv[2]) : 1000;
    const size_t kib = argc > 3 ? (size_t) std::atoi(argv[3]) : 256;
    const bool rotate = argc > 4 && std::string(argv[4]) == "rotate";
    const bool uc = mode == "uc";
    const size_t words = kib * 1024 / 8;          // 64-bit words per block
    const int NB = rotate ? 8 : 1;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    uint64_t* blk = sycl::malloc_host<uint64_t>(words * NB, q);
    uint32_t* flag = sycl::malloc_host<uint32_t>(16, q);
    uint32_t* ack = flag + 8;
    uint32_t* res = sycl::malloc_host<uint32_t>(4, q);
    uint64_t* dbg = sycl::malloc_host<uint64_t>(4, q); dbg[0] = dbg[1] = dbg[2] = dbg[3] = 0;
    std::memset(blk, 0, words * NB * 8); std::memset(flag, 0, 64); std::memset(res, 0, 16);
    constexpr uint32_t BOUND = 20u * 1000u * 1000u;
    auto t0 = clk::now();
    auto ev = q.parallel_for(sycl::nd_range<1>(1024, 1024), [=](sycl::nd_item<1> it) {
        auto& bad = *sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t>(it.get_group());
        auto& quit = *sycl::ext::oneapi::group_local_memory_for_overwrite<uint32_t>(it.get_group());
        const int lid = (int) it.get_local_id(0);
        uint32_t stale = 0, timeouts = 0;
        for (uint32_t i = 1; i <= (uint32_t) R; ++i) {
            if (lid == 0) {
                uint32_t n = 0;
                while (ucload(flag) < i && n < BOUND) ++n;
                quit = n >= BOUND;
                bad = 0;
            }
            sycl::group_barrier(it.get_group());
            if (quit) { if (lid == 0) ++timeouts; break; }
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);   // every lane, after the flag
            const uint64_t* b = blk + (size_t) (rotate ? (i % NB) : 0) * words;
            uint32_t mine = 0;
            for (size_t w = lid; w < words; w += 1024) {
                const uint64_t v = uc ? ucload64(b + w) : b[w];
                const uint64_t want = (uint64_t) i * 0x9E3779B97F4A7C15ull + w;
                if (v != want) { if (mine == 0 && i == 2 && w < 4096 && dbg[3] == 0) { dbg[0] = w; dbg[1] = v; dbg[2] = want; dbg[3] = 1; } ++mine; }
            }
            if (mine) sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                        sycl::access::address_space::local_space>(bad).fetch_add(mine);
            sycl::group_barrier(it.get_group());
            if (lid == 0) { stale += bad; sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system); ucstore(ack, i); }
        }
        if (lid == 0) { res[0] = stale; res[1] = timeouts; }
    });
    double host_us = 0;
    for (uint32_t i = 1; i <= (uint32_t) R; ++i) {
        uint64_t* b = blk + (size_t) (rotate ? (i % NB) : 0) * words;
        auto a = clk::now();
        for (size_t w = 0; w < words; ++w) b[w] = (uint64_t) i * 0x9E3779B97F4A7C15ull + w;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        std::atomic_ref<uint32_t>(*flag).store(i, std::memory_order_release);
        host_us += std::chrono::duration<double, std::micro>(clk::now() - a).count();
        auto w0 = clk::now();
        while (std::atomic_ref<uint32_t>(*ack).load(std::memory_order_acquire) < i)
            if (std::chrono::duration<double>(clk::now() - w0).count() > 3.0) { std::printf("ack %u never came\n", i); goto done; }
    }
done:
    ev.wait();
    const double tot = std::chrono::duration<double, std::micro>(clk::now() - t0).count();
    std::printf("first mismatch (round 2): word %llu got %llx want %llx (host now %llx)\n", (unsigned long long) dbg[0], (unsigned long long) dbg[1], (unsigned long long) dbg[2], (unsigned long long) blk[dbg[0]]);
    std::printf("%-5s %s %zu KiB x %d rounds: stale words %u, timeouts %u, %.1f us per round (host writes %.1f us), GPU read %.2f GB/s\n",
                mode.c_str(), rotate ? "ring8" : "same ", kib, R, res[0], res[1], tot / R, host_us / R,
                (double) kib * 1024 / ((tot - host_us) / R * 1e-6) / 1e9);
    return 0;
}
