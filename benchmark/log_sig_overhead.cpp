// Standalone native harness for the workloads in pySigLib benchmarks/bench_cpsig.cpp.
// Load the same library as Julia rather than rebuilding a different cpSIG version.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <random>
#include <vector>

extern "C" void fill_path(double* path, uint64_t count, double scale) {
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (uint64_t i = 0; i < count; ++i) path[i] = scale * dist(rng);
}

template <typename T>
double repeat(void* direct_ptr, void* signature_ptr, void* convert_ptr,
              const void* path_ptr, void* out_ptr, void* sig_ptr,
              uint64_t batch, uint64_t length, uint64_t dim, uint64_t degree,
              int method, uint64_t iterations) {
    using Direct = int (*)(const T*, T*, uint64_t, uint64_t, uint64_t, uint64_t, int);
    using Signature = int (*)(const T*, T*, uint64_t, uint64_t, uint64_t, uint64_t,
                             bool, bool, T, bool, bool, int, const T*, uint64_t, uint64_t, uint64_t);
    using Convert = int (*)(const T*, T*, uint64_t, uint64_t, uint64_t, bool, bool, int, bool, int);
    auto direct = reinterpret_cast<Direct>(direct_ptr);
    auto signature = reinterpret_cast<Signature>(signature_ptr);
    auto convert = reinterpret_cast<Convert>(convert_ptr);
    auto path = static_cast<const T*>(path_ptr);
    auto out = static_cast<T*>(out_ptr);
    auto sig = static_cast<T*>(sig_ptr);
    auto start = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < iterations; ++i) {
        int rc;
        if (method == 3) {
            rc = direct(path, out, batch, length, dim, degree, 1);
        } else {
            rc = signature(path, sig, batch, dim, length, degree, false, false, T(1),
                           true, false, 1, nullptr, 0, 0, 0);
            if (!rc) rc = convert(sig, out, batch, dim, degree, false, false, method, false, 1);
        }
        if (rc) return -rc;
    }
    return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() / iterations;
}

extern "C" double native_repeat(void* direct, void* signature, void* convert,
                                 const void* path, void* out, void* sig,
                                 uint64_t batch, uint64_t length, uint64_t dim, uint64_t degree,
                                 int method, int is_float, uint64_t iterations) {
    if (is_float) return repeat<float>(direct, signature, convert, path, out, sig, batch, length, dim, degree, method, iterations);
    return repeat<double>(direct, signature, convert, path, out, sig, batch, length, dim, degree, method, iterations);
}

#ifndef HARNESS_SHARED
int main(int argc, char** argv) {
    if (argc != 10) {
        std::fprintf(stderr, "Usage: %s LIBRARY CACHE_DIR float|double BATCH LENGTH DIM DEGREE METHOD ITERATIONS\n", argv[0]);
        return 1;
    }
    void* lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) { std::fprintf(stderr, "%s\n", dlerror()); return 1; }
    auto set_cache = reinterpret_cast<int (*)(const char*)>(dlsym(lib, "set_cache_dir"));
    auto prepare = reinterpret_cast<int (*)(uint64_t, uint64_t, int, bool)>(dlsym(lib, "prepare_log_sig"));
    auto log_length = reinterpret_cast<uint64_t (*)(uint64_t, uint64_t)>(dlsym(lib, "log_sig_length"));
    auto sig_length = reinterpret_cast<uint64_t (*)(uint64_t, uint64_t)>(dlsym(lib, "sig_length"));
    bool is_float = argv[3][0] == 'f';
    uint64_t batch = std::strtoull(argv[4], nullptr, 10), length = std::strtoull(argv[5], nullptr, 10);
    uint64_t dim = std::strtoull(argv[6], nullptr, 10), degree = std::strtoull(argv[7], nullptr, 10);
    int method = std::atoi(argv[8]);
    uint64_t iterations = std::strtoull(argv[9], nullptr, 10);
    // cpSIG 3.1 needs method 2 preparation for direct BCH; 4.x uses method 3.
    bool modern = dlsym(lib, "branched_log_sig_length") != nullptr;
    if (set_cache(argv[2]) || prepare(dim, degree, !modern && method == 3 ? 2 : method, false)) return 2;
    std::vector<double> data(batch * length * dim), out(batch * log_length(dim, degree)), sig(batch * (sig_length(dim, degree) - 1));
    fill_path(data.data(), data.size(), dim == 2 && length == 129 ? 0.1 : 1.0);
    std::vector<float> data_f(data.begin(), data.end()), out_f(out.size()), sig_f(sig.size());
    auto direct = dlsym(lib, is_float ? "log_sig_from_path_f" : "log_sig_from_path_d");
    auto signature = dlsym(lib, is_float ? "signature_f" : "signature_d");
    auto convert = dlsym(lib, is_float ? "sig_to_log_sig_f" : "sig_to_log_sig_d");
    if (!direct || !signature || !convert) return 3;
    const void* path_ptr = is_float ? static_cast<void*>(data_f.data()) : static_cast<void*>(data.data());
    void* out_ptr = is_float ? static_cast<void*>(out_f.data()) : static_cast<void*>(out.data());
    void* sig_ptr = is_float ? static_cast<void*>(sig_f.data()) : static_cast<void*>(sig.data());
    if (native_repeat(direct, signature, convert, path_ptr, out_ptr, sig_ptr, batch, length, dim, degree, method, is_float, 10) < 0) return 4;
    double ns = native_repeat(direct, signature, convert, path_ptr, out_ptr, sig_ptr, batch, length, dim, degree, method, is_float, iterations);
    if (ns < 0) return 5;
    std::printf("%.9f\n", ns);
}
#endif
