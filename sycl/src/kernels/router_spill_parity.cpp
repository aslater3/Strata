// The routers' spill-free variants (spill_dispatch.hpp) against the previous kernels in one process, bytewise:
// STRATA_SPILL_FREE=0 vs 1 for the decode routers (native_router_top10, native_router_top10_multi_ne at 256 and 512
// experts) and the prompt router (strata::prefill::route), with ties (all-zero logits) and NaNs, from sentinel-filled
// outputs.  Synthetic, no model; runs on any device (on the CPU device it checks the algorithm, not large-GRF mode).
#include <sycl/sycl.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/prefill/kernels.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace k = strata::kernels;
static void spill_free(bool on) {
#ifdef _WIN32
    _putenv_s("STRATA_SPILL_FREE", on ? "1" : "0");
#else
    setenv("STRATA_SPILL_FREE", on ? "1" : "0", 1);
#endif
}
template<class T> struct Buffer {
    sycl::queue& q;
    T* p;
    size_t n;
    Buffer(sycl::queue& queue, size_t count) : q(queue), p(sycl::malloc_device<T>(count, q)), n(count) {
        if (!p) throw std::bad_alloc();
    }
    ~Buffer() { sycl::free(p, q); }
    Buffer(const Buffer&) = delete;
    void put(const std::vector<T>& v) { q.memcpy(p, v.data(), n * sizeof(T)).wait_and_throw(); }
    std::vector<T> get() { std::vector<T> v(n); q.memcpy(v.data(), p, n * sizeof(T)).wait_and_throw(); return v; }
};
int main() {
    auto& q = *strata::q_of(nullptr);
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::mt19937 rng(2117);
    auto random = [&](size_t n, float scale) {
        std::vector<float> v(n);
        for (auto& x : v) x = float(int(rng() % 2001) - 1000) * scale / 1000;
        return v;
    };
    int bad = 0, cases = 0;
    auto equal = [&](const auto& a, const auto& b) {
        ++cases;
        const bool ok = std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0;
        bad += !ok;
        return ok;
    };
    for (int ne : {256, 512}) for (int T : {1, 7, 8, 9, 19}) for (int mode : {0, 1, 2}) {
        Buffer<float> logits(q, size_t(T) * ne), weights(q, size_t(T) * 10);
        Buffer<int32_t> ids(q, size_t(T) * 10);
        auto values = random(logits.n, 2.0f);
        if (mode == 1) for (auto& v : values) v = 0.0f;
        if (mode == 2) for (size_t i = 3; i < values.size(); i += 71) values[i] = std::numeric_limits<float>::quiet_NaN();
        logits.put(values);
        bool ok = true;
        // 0: decode multi-token, 1: prompt router, 2: single-token decode router (512 experts, one token)
        for (int entry : {0, 1, 2}) {
            if (entry == 2 && (ne != 512 || T != 1)) continue;
            auto run = [&](bool on) {
                ids.put(std::vector<int32_t>(ids.n, -777));
                weights.put(std::vector<float>(weights.n, -777.0f));
                spill_free(on);
                if (entry == 1) strata::prefill::route(logits.p, ids.p, weights.p, T, ne, &q);
                else if (entry == 2) k::native_router_top10(logits.p, ids.p, weights.p, &q);
                else k::native_router_top10_multi_ne(logits.p, ids.p, weights.p, T, ne, &q);
                q.wait_and_throw();
            };
            run(false); auto oldi = ids.get(); auto oldw = weights.get();
            run(true); ok &= equal(oldi, ids.get()); ok &= equal(oldw, weights.get());
        }
        std::printf("routers NE=%d T=%d mode=%d: %s\n", ne, T, mode, ok ? "OK" : "FAIL");
    }
    std::printf("router_spill_parity: %d bytewise comparisons, %d failures\n", cases, bad);
    return bad ? 1 : 0;
}
