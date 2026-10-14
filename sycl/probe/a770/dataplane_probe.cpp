// dataplane_probe: how can the host hand a running DG2 kernel a fresh block of data (the CPU experts' rows) each round?
//   dataplane_probe bu <rounds> <KiB>   block in BIAS_UNCACHED host USM, GPU reads it with plain loads after the flag
//   dataplane_probe ce <rounds> <KiB>   block in VRAM, written by the COPY ENGINE (staging -> block, then the flag word);
//                                       GPU spins on the device flag, then reads the block with plain loads
// Both: flag wait with an uncached-hint load (+acquire fence), every word checked, ack to the host with an uncached store.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <level_zero/ze_api.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace exi = sycl::ext::intel::experimental; namespace exo = sycl::ext::oneapi::experimental;
using uc_read = decltype(exo::properties(exi::read_hint<exi::cache_control<exi::cache_mode::uncached, exo::cache_level::L1, exo::cache_level::L3>>));
using uc_write = decltype(exo::properties(exi::write_hint<exi::cache_control<exi::cache_mode::uncached, exo::cache_level::L1, exo::cache_level::L3>>));
static inline uint32_t ucload(const uint32_t* p) {
    sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
    exo::annotated_ptr<uint32_t, uc_read> u(const_cast<uint32_t*>(p)); return u[0]; }
static inline uint64_t ucload64(const uint64_t* p) { exo::annotated_ptr<uint32_t, uc_read> u(reinterpret_cast<uint32_t*>(const_cast<uint64_t*>(p))); const uint32_t lo = u[0], hi = u[1]; return ((uint64_t) hi << 32) | lo; }
static inline void ucstore(uint32_t* p, uint32_t v) { exo::annotated_ptr<uint32_t, uc_write> u(p); u[0] = v; }
using clk = std::chrono::steady_clock;

static sycl::queue* copy_engine_queue(sycl::queue& q) {
    auto zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
    auto zdev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    uint32_t ng = 0;
    zeDeviceGetCommandQueueGroupProperties(zdev, &ng, nullptr);
    std::vector<ze_command_queue_group_properties_t> g(ng);
    for (auto& x : g) { std::memset(&x, 0, sizeof(x)); x.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES; }
    zeDeviceGetCommandQueueGroupProperties(zdev, &ng, g.data());
    int ord = -1;
    for (uint32_t i = 0; i < ng; ++i) {
        std::printf("queue group %u: flags 0x%x, %u queues\n", i, g[i].flags, g[i].numQueues);
        if ((g[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) && !(g[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) &&
            g[i].numQueues > 0 && ord < 0) ord = (int) i;
    }
    if (ord < 0) { std::printf("no copy-only queue group\n"); return nullptr; }
    ze_command_queue_desc_t d{};
    d.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC; d.ordinal = (uint32_t) ord; d.index = 0;
    d.flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER; d.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS; d.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ze_command_list_handle_t cl = nullptr;
    if (zeCommandListCreateImmediate(zctx, zdev, &d, &cl) != ZE_RESULT_SUCCESS) { std::printf("immediate list failed\n"); return nullptr; }
    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::queue> in(
        cl, q.get_device(), sycl::ext::oneapi::level_zero::ownership::keep, sycl::property_list{sycl::property::queue::in_order{}});
    std::printf("copy engine: queue group %d\n", ord);
    return new sycl::queue(sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(in, q.get_context()));
}

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "bu";
    const int R = argc > 2 ? std::atoi(argv[2]) : 1000;
    const size_t kib = argc > 3 ? (size_t) std::atoi(argv[3]) : 256;
    const size_t words = kib * 1024 / 8;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    const bool ce = mode == "ce" || mode == "ceuc";
    const bool rduc = mode == "ceuc";
    uint64_t* blk = nullptr;      // what the kernel reads
    uint64_t* stage = nullptr;    // ce: pinned host staging the copy engine reads
    uint32_t* dflag = nullptr;    // ce: device flag
    uint32_t* hsrc = sycl::malloc_host<uint32_t>(4096, q);   // ce: flag values to copy from (one slot per round, mod 4096)
    sycl::queue* cq = nullptr;
    if (ce) {
        cq = copy_engine_queue(q);
        if (!cq) return 1;
        blk = sycl::malloc_device<uint64_t>(words, q);
        stage = sycl::malloc_host<uint64_t>(words, q);
        dflag = sycl::malloc_device<uint32_t>(16, q);
        q.memset(blk, 0, words * 8); q.memset(dflag, 0, 64); q.wait();
    } else {
        auto zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
        ze_host_mem_alloc_desc_t d{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED};
        void* p = nullptr;
        if (zeMemAllocHost(zctx, &d, words * 8, 64, &p) != ZE_RESULT_SUCCESS) { std::printf("BIAS_UNCACHED alloc failed\n"); return 1; }
        blk = (uint64_t*) p;
        std::memset(blk, 0, words * 8);
    }
    uint32_t* hflag = sycl::malloc_host<uint32_t>(16, q);   // bu: host flag; both: ack at [8]
    uint32_t* ack = hflag + 8;
    uint32_t* res = sycl::malloc_host<uint32_t>(4, q);
    std::memset(hflag, 0, 64); std::memset(res, 0, 16);
    const uint32_t* wflag = ce ? dflag : hflag;
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
                while (ucload(wflag) < i && n < BOUND) ++n;
                quit = n >= BOUND; bad = 0;
            }
            sycl::group_barrier(it.get_group());
            if (quit) { if (lid == 0) ++timeouts; break; }
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
            uint32_t mine = 0;
            for (size_t w = lid; w < words; w += 1024)
                if ((rduc ? ucload64(blk + w) : blk[w]) != (uint64_t) i * 0x9E3779B97F4A7C15ull + w) ++mine;
            if (mine) sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                        sycl::access::address_space::local_space>(bad).fetch_add(mine);
            sycl::group_barrier(it.get_group());
            if (lid == 0) { stale += bad; sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system); ucstore(ack, i); }
        }
        if (lid == 0) { res[0] = stale; res[1] = timeouts; }
    });
    double host_us = 0;
    int done_rounds = 0;
    for (uint32_t i = 1; i <= (uint32_t) R; ++i) {
        auto a = clk::now();
        uint64_t* dst = ce ? stage : blk;
        for (size_t w = 0; w < words; ++w) dst[w] = (uint64_t) i * 0x9E3779B97F4A7C15ull + w;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (ce) {
            hsrc[i % 4096] = i;
            cq->memcpy(blk, stage, words * 8);     // the rows, then (in order on the same copy list) the flag
            cq->memcpy(dflag, &hsrc[i % 4096], 4);
        } else {
            std::atomic_ref<uint32_t>(*hflag).store(i, std::memory_order_release);
        }
        host_us += std::chrono::duration<double, std::micro>(clk::now() - a).count();
        auto w0 = clk::now();
        while (std::atomic_ref<uint32_t>(*ack).load(std::memory_order_acquire) < i)
            if (std::chrono::duration<double>(clk::now() - w0).count() > 3.0) { std::printf("ack %u never came\n", i); goto done; }
        ++done_rounds;
        if (ce) cq->wait();   // the staging buffer is reused next round
    }
done:
    ev.wait();
    const double tot = std::chrono::duration<double, std::micro>(clk::now() - t0).count();
    std::printf("%-3s %zu KiB x %d rounds (%d done): stale words %u, timeouts %u, %.1f us per round (host %.1f us)\n", mode.c_str(),
                kib, R, done_rounds, res[0], res[1], tot / std::max(1, done_rounds), host_us / std::max(1, done_rounds));
    return 0;
}
