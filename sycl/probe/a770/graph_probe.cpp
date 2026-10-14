// graph_probe: per-node cost of one big SYCL graph vs many small ones on DG2 (the stepped window's 97 segments vs the
// single-graph window). N tiny kernels (each a few us of work), recorded as 1 graph or S graphs of N/S nodes.
//   graph_probe <nodes> <segments> [hostmem]   hostmem: every 20th node also reads/writes a word of host USM
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace sx = sycl::ext::oneapi::experimental;
using clk = std::chrono::steady_clock;
struct Big { float* p[150]; int n; };   // ~1.2 KB by value, like GrMulti

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 1920;
    const int S = argc > 2 ? std::atoi(argv[2]) : 1;
    const bool hostmem = argc > 3 && std::string(argv[3]) == "hostmem";
    const bool bigarg = argc > 3 && std::string(argv[3]) == "bigarg";
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    float* d = sycl::malloc_device<float>(1 << 20, q);
    uint32_t* h = sycl::malloc_host<uint32_t>(64, q);
    q.memset(d, 0, 4 << 20).wait();
    std::vector<sx::command_graph<sx::graph_state::executable>> ex;
    const int per = N / S;
    for (int s = 0; s < S; ++s) {
        sx::command_graph g{q.get_context(), q.get_device()};
        g.begin_recording(q);
        for (int i = 0; i < per; ++i) {
            const int k = s * per + i;
            if (hostmem && k % 20 == 0)
                q.single_task([=] { sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::system>(h[0]).fetch_add(1); });
            else if (bigarg) {
                Big b; for (int z = 0; z < 150; ++z) b.p[z] = d + z * 4096; b.n = k;
                q.parallel_for(sycl::nd_range<1>(65536, 256), [=](sycl::nd_item<1> it) {
                    const size_t j = it.get_global_id(0);
                    float* t = b.p[(j >> 12) % 150];
                    t[j & 4095] = t[j & 4095] * 0.999f + (float) b.n;
                });
            } else
                q.parallel_for(sycl::nd_range<1>(65536, 256), [=](sycl::nd_item<1> it) {   // ~64K floats: a small decode kernel
                    const size_t j = it.get_global_id(0);
                    d[j] = d[j] * 0.999f + (float) k;
                });
        }
        g.end_recording(q);
        ex.push_back(g.finalize());
    }
    auto run = [&] { for (auto& e : ex) { q.ext_oneapi_graph(e); if (S > 1) q.wait(); } q.wait(); };   // S>1: drain per segment, as stepped
    run(); run();
    double best = 1e9;
    for (int r = 0; r < 5; ++r) { auto t0 = clk::now(); run(); best = std::min(best, std::chrono::duration<double, std::milli>(clk::now() - t0).count()); }
    // eager baseline: the same kernels submitted directly
    auto t0 = clk::now();
    for (int k = 0; k < N; ++k) q.parallel_for(sycl::nd_range<1>(65536, 256), [=](sycl::nd_item<1> it) { const size_t j = it.get_global_id(0); d[j] = d[j] * 0.999f + (float) k; });
    q.wait();
    const double eager = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    std::printf("%d nodes as %d graph(s)%s: %.2f ms (%.2f us/node); eager %.2f ms (%.2f us/node)\n", N, S, hostmem ? " +hostmem" : bigarg ? " +1.2KB args" : "",
                best, best * 1000 / N, eager, eager * 1000 / N);
    return 0;
}
