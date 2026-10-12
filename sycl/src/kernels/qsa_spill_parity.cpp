// QSA block selection three ways, bitwise: the reference kernel, the previous kernels (STRATA_SPILL_FREE=0) and the
// spill-free streaming top-k (STRATA_SPILL_FREE=1, spill_dispatch.hpp), with ties, NaN, +-inf, -0 and the tail
// block, from sentinel-filled outputs.  Synthetic, no model; runs on any device.
#include <sycl/sycl.hpp>
#include "strata/sycl_queue.hpp"
#include "strata/kernels/qsa_select.hpp"
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

static void spill_free(bool on) {
#ifdef _WIN32
    _putenv_s("STRATA_SPILL_FREE", on ? "1" : "0");
#else
    setenv("STRATA_SPILL_FREE", on ? "1" : "0", 1);
#endif
}
int main() {
    // Use the default selected queue, so the same binary can validate the GPU
    // when the parent sets ONEAPI_DEVICE_SELECTOR=level_zero:gpu.
    auto& selected = *strata::q_of(nullptr);
    std::printf("device: %s\n", selected.get_device().get_info<sycl::info::device::name>().c_str());
    strata::kernels::QsaShapes shape{};
    shape.idx_block = 4;
    constexpr int nq = 3;
    constexpr int cap = 4096;
    std::mt19937 rng(2031);
    int failures = 0;
    for (int n : {1, 2048, 2051, 2052, 2053, 8192, 131071, 131072, 131073}) {
        const int blocks = n / 4 + 1;
        std::vector<float> scores(size_t(nq) * blocks);
        for (size_t i = 0; i < scores.size(); ++i) {
            // Deliberately coarse scores exercise many threshold ties.
            scores[i] = float(int(rng() % 101) - 50);
            if (i % 97 == 0) scores[i] = std::numeric_limits<float>::quiet_NaN();
            if (i % 89 == 0) scores[i] = std::numeric_limits<float>::infinity();
            if (i % 83 == 0) scores[i] = -std::numeric_limits<float>::infinity();
            if (i % 79 == 0) scores[i] = -0.0f;
        }
        std::vector<int32_t> steps(size_t(nq) * strata::kernels::kStepCount, 0);
        for (int t = 0; t < nq; ++t) {
            auto* st = steps.data() + t * strata::kernels::kStepCount;
            st[strata::kernels::kStepNKv] = n;
            st[strata::kernels::kStepNBid] = n / 4;
            st[strata::kernels::kStepWidth] = int(strata::kernels::qsa_selection_width(n, shape));
        }
        auto* ds = sycl::malloc_device<float>(scores.size(), selected);
        auto* dt = sycl::malloc_device<int32_t>(steps.size(), selected);
        auto* di = sycl::malloc_device<int32_t>(nq * cap, selected);
        if (!ds || !dt || !di) throw std::bad_alloc();
        selected.memcpy(ds, scores.data(), scores.size() * sizeof(float)).wait_and_throw();
        selected.memcpy(dt, steps.data(), steps.size() * sizeof(int32_t)).wait_and_throw();
        std::vector<int32_t> ref(nq * cap), old(ref.size()), got(ref.size());
        auto run = [&](int arm, std::vector<int32_t>& dst) {
            selected.fill(di, int32_t(-777), dst.size()).wait_and_throw();
            spill_free(arm == 0);
            if (arm == 2) strata::kernels::qsa_block_topk_ref(ds, dt, nq, blocks, cap, shape, di, &selected);
            else strata::kernels::qsa_block_topk(ds, dt, nq, blocks, cap, shape, di, &selected);
            selected.memcpy(dst.data(), di, dst.size() * sizeof(int32_t)).wait_and_throw();
        };
        run(2, ref); run(1, old); run(0, got);
        const bool ok = ref == old && old == got;
        failures += !ok;
        std::printf("QSA n=%d queries=%d ref/old/new + sentinels: %s\n", n, nq, ok ? "OK" : "FAIL");
        sycl::free(ds, selected); sycl::free(dt, selected); sycl::free(di, selected);
    }
    std::printf("qsa_spill_parity: %d failures\n", failures);
    return failures ? 1 : 0;
}
