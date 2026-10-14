// flag_probe: can a running DG2 kernel be signalled mid-flight, and how fast is a GPU<->host handshake?
//   flag_probe dev      GPU spins on a DEVICE flag; host writes it with a memcpy on a second queue
//   flag_probe host     GPU spins on a HOST-USM flag; host writes it with a plain CPU store
//   flag_probe shared   GPU spins on a SHARED-USM flag; host writes it with a plain CPU store
//   flag_probe ping N   N round trips: GPU publishes seq to host USM, host answers into a device flag (2nd queue)
//   flag_probe pingh N  N round trips, host answers with a CPU store into a host-USM flag
// Every GPU wait is bounded (BOUND loads), so a failure is a reported timeout, not a hang.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

using clk = std::chrono::steady_clock;
static double us(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); }
constexpr uint32_t BOUND = 20u * 1000u * 1000u;   // loads per wait before giving up

namespace exi = sycl::ext::intel::experimental; namespace exo = sycl::ext::oneapi::experimental;
using uc_read = decltype(exo::properties(exi::read_hint<exi::cache_control<exi::cache_mode::uncached, exo::cache_level::L1, exo::cache_level::L3>>));
using uc_write = decltype(exo::properties(exi::write_hint<exi::cache_control<exi::cache_mode::uncached, exo::cache_level::L1, exo::cache_level::L3>>));
static inline uint32_t ucload(const uint32_t* p) { exo::annotated_ptr<uint32_t, uc_read> u(const_cast<uint32_t*>(p)); return u[0]; }
static inline void ucstore(uint32_t* p, uint32_t v) { exo::annotated_ptr<uint32_t, uc_write> u(p); u[0] = v; }
template <class T> static inline T vload(const T* p) { return *(const volatile T*) p; }
template <class T> static inline void vstore(T* p, T v) { *(volatile T*) p = v; }

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "dev";
    const int N = argc > 2 ? std::atoi(argv[2]) : 1000;
    sycl::queue qk{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    sycl::queue qc{qk.get_context(), qk.get_device(), sycl::property::queue::in_order{}};   // the "answer" queue
    uint32_t* dflag = sycl::malloc_device<uint32_t>(1, qk);
    uint32_t* hflag = sycl::malloc_host<uint32_t>(1, qk);     // host -> GPU (host-USM variant) or GPU -> host seq
    uint32_t* hseq = sycl::malloc_host<uint32_t>(1, qk);      // GPU -> host published seq
    uint32_t* sflag = sycl::malloc_shared<uint32_t>(1, qk);
    uint32_t* hval = sycl::malloc_host<uint32_t>(64, qk);     // pinned source values for the memcpy answers
    uint32_t* res = sycl::malloc_host<uint32_t>(4, qk);       // kernel results: [0] loads used / timeouts, [1] last seen
    qk.memset(dflag, 0, 4).wait();
    *hflag = 0; *hseq = 0; *sflag = 0; std::memset(res, 0, 16);
    for (int i = 0; i < 64; ++i) hval[i] = 0;

    if (mode == "dev" || mode == "host" || mode == "shared" || mode == "hostuc") {
        const int which = mode == "dev" ? 0 : mode == "host" ? 1 : mode == "shared" ? 2 : 3;
        uint32_t* f = which == 0 ? dflag : which == 2 ? sflag : hflag;
        auto t0 = clk::now();
        auto ev = qk.single_task([=] {
            uint32_t n = 0, v = 0;
            while (n < BOUND) {
                v = which == 0 ? sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(*f).load()
                  : which == 3 ? ucload(f) : vload(f);
                if (v == 1u) break;
                ++n;
            }
            res[0] = n; res[1] = v;
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));   // the kernel is spinning by now
        auto tw = clk::now();
        if (which == 0) { hval[0] = 1; qc.memcpy(f, hval, 4); }
        else std::atomic_ref<uint32_t>(*f).store(1u, std::memory_order_release);
        ev.wait();
        auto te = clk::now();
        std::printf("%-6s: kernel %s after %.1f us from the host's write (loads %u, saw %u; %.1f ms since launch)\n", mode.c_str(),
                    res[1] == 1u ? "SAW the flag" : "TIMED OUT", us(tw, te), res[0], res[1], us(t0, te) / 1000.0);
        return 0;
    }

    const bool dev_answer = mode == "ping";
    const bool uc = mode == "pinguc";
    // GPU: for i in 1..N publish seq=i to host USM, then wait for answer==i (device flag or host flag)
    auto t0 = clk::now();
    auto ev = qk.single_task([=] {
        uint32_t timeouts = 0;
        for (uint32_t i = 1; i <= (uint32_t) N; ++i) {
            if (uc) ucstore(hseq, i); else vstore(hseq, i);
            sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
            uint32_t n = 0;
            while (n < BOUND) {
                const uint32_t v = dev_answer
                    ? sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device>(*dflag).load()
                    : uc ? ucload(hflag) : vload(hflag);
                if (v >= i) break;
                ++n;
            }
            if (n >= BOUND) { ++timeouts; break; }
        }
        res[0] = timeouts;
    });
    int seen = 0;
    double host_wait_max = 0;
    for (uint32_t i = 1; i <= (uint32_t) N; ++i) {
        auto tw = clk::now();
        while (std::atomic_ref<uint32_t>(*hseq).load(std::memory_order_acquire) < i) {
            if (us(tw, clk::now()) > 2e6) { std::printf("host: GPU's seq %u never arrived (stuck at %u)\n", i, *hseq); goto done; }
        }
        host_wait_max = std::max(host_wait_max, us(tw, clk::now()));
        ++seen;
        if (dev_answer) { hval[i % 64] = i; qc.memcpy(dflag, &hval[i % 64], 4); }
        else std::atomic_ref<uint32_t>(*hflag).store(i, std::memory_order_release);
    }
done:
    ev.wait();
    const double tot = us(t0, clk::now());
    std::printf("%-6s: %d of %d round trips, GPU timeouts %u, %.2f us per round trip (incl. launch), slowest GPU->host %.1f us\n",
                mode.c_str(), seen, N, res[0], tot / std::max(1, seen), host_wait_max);
    return 0;
}
