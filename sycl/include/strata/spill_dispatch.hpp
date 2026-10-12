// include/strata/spill_dispatch.hpp - the A770 port's spill-free kernel variants and when they run.
//
// On an Arc A770 (Alchemist, DG2) the QSA top-k (11.9 KB/thread), the routers and the grouped gate/up kernel spilled
// registers to scratch.  Their spill-free variants give the same bits: router_spill_parity, qsa_spill_parity and
// `s2_expert_grouped_parity --spill-only` compare the two byte for byte on any card, the CPU device included.
// Measured on an A770 16 GB (Coder IQ1_M, 128K context, int8 KV): decode 24.5 -> 26.2 tok/s (+7%), prefill 0 to +1%
// (QSA select 220 -> 171 ms on an 8K prompt), the same tokens.  Not measured on any other card, so the spill-free
// kernels are the default on Alchemist only.
//
// STRATA_SPILL_FREE: unset = the spill-free kernels on Alchemist, the previous ones everywhere else; 1 = the
// spill-free kernels on every device (large-GRF mode only on Intel GPUs); 0 = the previous kernels everywhere.
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "strata/sycl_queue.hpp"
#include <cstdlib>
#include <mutex>
#include <vector>

namespace strata {
template <class Original> class SpillLargeGrf;
template <class Original> class SpillPortable;

struct SpillDevice {
    bool alchemist = false;   // the default for STRATA_SPILL_FREE
    bool intel_gpu = false;   // large-GRF mode is an Intel GPU property
};
// Asked once per device: the launchers below run per layer and token.
inline SpillDevice spill_device(const sycl::queue& q) {
    struct Seen { sycl::device d; SpillDevice s; };
    static std::mutex mu;
    static std::vector<Seen> seen;
    const sycl::device d = q.get_device();
    std::lock_guard<std::mutex> lk(mu);
    for (const Seen& x : seen) if (x.d == d) return x.s;
    SpillDevice s;
    s.alchemist = intel_gpu_gen(d) == IntelGpuGen::Alchemist;
    s.intel_gpu = d.is_gpu() && d.get_info<sycl::info::device::vendor_id>() == 0x8086;
    seen.push_back({d, s});
    return s;
}

// The switch is read on every launch, not cached, so a parity test can run both arms in one process.
inline bool spill_free(const sycl::queue& q) {
    const char* v = std::getenv("STRATA_SPILL_FREE");
    if (v && v[0] == '0') return false;
    if (v && v[0] == '1') return true;
    return spill_device(q).alchemist;
}

// A kernel with a spill-free body.  Unless spill_free(q): the previous body, kernel name and properties, unchanged.
// The new body also runs on the CPU device (the parity tests); only the register-size property is Intel-GPU-only.
template <class Kernel, int D, class Properties, class OldBody, class NewBody>
inline sycl::event spill_parallel_for(sycl::queue& q, sycl::nd_range<D> range,
                                      Properties original, OldBody old_body, NewBody new_body) {
    if (!spill_free(q)) return q.parallel_for<Kernel>(range, original, old_body);
    if (spill_device(q).intel_gpu) {
        const auto props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync,
            sycl::ext::intel::experimental::grf_size<256>};
        return q.parallel_for<SpillLargeGrf<Kernel>>(range, props, new_body);
    }
    return q.parallel_for<SpillPortable<Kernel>>(range, original, new_body);
}

// A kernel whose body is unchanged and needs only large-GRF mode (256 registers) on that one kernel; a whole-build
// large-GRF mode was ~2.5x slower on the A770.  Never requests Intel register properties on the CPU or another vendor.
template <class Kernel, int D, class Properties, class Body>
inline sycl::event spill_parallel_for(sycl::queue& q, sycl::nd_range<D> range,
                                      Properties original, Body body) {
    if (spill_free(q) && spill_device(q).intel_gpu) {
        const auto props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync,
            sycl::ext::intel::experimental::grf_size<256>};
        return q.parallel_for<SpillLargeGrf<Kernel>>(range, props, body);
    }
    return q.parallel_for<Kernel>(range, original, body);
}
}  // namespace strata
