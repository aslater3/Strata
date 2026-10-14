// import_probe: can an existing malloc'd + mlock'd host buffer (the engine's resident expert set) be made a fast copy
// source on DG2 with sycl_ext_oneapi_copy_optimize (prepare_for_device_copy = Level Zero's import of an external
// pointer), and does it work above the 4 GiB single pinned-allocation limit?
//   import_probe <GiB> [chunk_GiB]   chunk_GiB 0 = register the whole buffer in one call
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>

using clk = std::chrono::steady_clock;
#include <dirent.h>
#include <string>
static double vram_gib() {   // this process's local memory per the DRM fdinfo (xe: drm-total-vram0)
    double tot = 0; DIR* d = opendir("/proc/self/fdinfo"); if (!d) return -1;
    while (dirent* e = readdir(d)) {
        std::string p = std::string("/proc/self/fdinfo/") + e->d_name; FILE* f = fopen(p.c_str(), "r"); if (!f) continue;
        char line[256]; while (fgets(line, sizeof line, f)) { unsigned long long v; char u[8] = {0};
            if (sscanf(line, "drm-total-vram0: %llu %7s", &v, u) >= 1) tot += v * (u[0]=='K' ? 1024.0 : u[0]=='M' ? 1048576.0 : u[0]=='G' ? 1073741824.0 : 1.0); }
        fclose(f);
    }
    closedir(d); return tot / 1073741824.0;
}
constexpr size_t GiB = 1ull << 30;
namespace sx = sycl::ext::oneapi::experimental;

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const double g = argc > 1 ? std::atof(argv[1]) : 6.0;
    const double cg = argc > 2 ? std::atof(argv[2]) : 0.0;
    const size_t n = (size_t) (g * GiB), chunk = cg > 0 ? (size_t) (cg * GiB) : n;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    uint8_t* h = (uint8_t*) std::aligned_alloc(4096, n);
    std::memset(h, 1, n);
    std::printf("mlock %.1f GiB: %s\n", g, mlock(h, n) == 0 ? "ok" : "failed");
    const size_t cn = 2 * 1024 * 1024;   // one expert blob
    uint8_t* d = sycl::malloc_device<uint8_t>(cn * 64, q);
    auto rate = [&](const char* what) {
        // 64 blob copies spread over the whole buffer, the last 1 GiB past 4 GiB included
        auto t0 = clk::now();
        for (int i = 0; i < 64; ++i) q.memcpy(d + (size_t) i * cn, h + (n - cn) / 63 * i, cn);
        q.wait();
        const double s = std::chrono::duration<double>(clk::now() - t0).count();
        std::printf("%-40s %6.2f GB/s (%.1f us per 2 MiB blob)\n", what, 64.0 * cn / s / 1e9, s / 64 * 1e6);
    };
    rate("before registration (pageable)"); rate("before registration (again)");
    std::vector<void*> vfill;   // VRAM_FILL_GIB: take most of the card first, as the engine's expert cache does
    if (const char* vf = std::getenv("VRAM_FILL_GIB")) {
        const int pieces = (int) (std::atof(vf) * 4);
        for (int i = 0; i < pieces; ++i) { void* p = sycl::malloc_device((size_t) 256 << 20, q); if (!p) { std::printf("VRAM fill stopped at %d x 256 MiB\n", i); break; } vfill.push_back(p); }
    }
    std::printf("VRAM held before registration: %.3f GiB\n", vram_gib());
    auto t0 = clk::now();
    int ok = 0, bad = 0;
    for (size_t off = 0; off < n; off += chunk) {
        try { sx::prepare_for_device_copy(h + off, std::min(chunk, n - off), q); ++ok; }
        catch (const std::exception& e) { ++bad; std::printf("prepare_for_device_copy at %.1f GiB failed: %s\n", off / (double) GiB, e.what()); }
    }
    std::printf("registered %d piece(s) of %.1f GiB (%d failed) in %.1f ms\n", ok, chunk / (double) GiB, bad,
                std::chrono::duration<double, std::milli>(clk::now() - t0).count());
    std::printf("VRAM held after registration: %.3f GiB\n", vram_gib());
    rate("after registration"); rate("after registration (again)");
    for (size_t mb : {1, 64, 1024}) {
        void* dd = sycl::malloc_device(mb << 20, q); void* hh = sycl::malloc_host(mb << 20, q);
        std::printf("after registration: malloc_device %zu MiB %s, malloc_host %zu MiB %s\n", mb, dd ? "ok" : "NULL", mb, hh ? "ok" : "NULL");
        if (dd) sycl::free(dd, q); if (hh) sycl::free(hh, q);
    }
    {   // how much more pinned host memory can be allocated now?
        std::vector<void*> got; size_t tot = 0;
        for (int i = 0; i < 128; ++i) { void* p = sycl::malloc_host((size_t) 256 << 20, q); if (!p) break; got.push_back(p); tot += 256; }
        std::printf("after registration: %zu MiB more pinned host memory in 256 MiB pieces before NULL\n", tot);
        for (void* p : got) sycl::free(p, q);
    }
    for (size_t off = 0; off < n; off += chunk) sx::release_from_device_copy(h + off, q);
    return 0;
}
