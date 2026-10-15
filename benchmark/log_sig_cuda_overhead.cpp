// Native CUDA C-API loop. Inputs and outputs are allocated by Julia on the GPU.
#include <chrono>
#include <cstdint>
#include <cuda_runtime_api.h>

static int synchronize() {
    return static_cast<int>(cudaDeviceSynchronize());
}

template <typename T>
double repeat(void* direct_ptr, void* signature_ptr, void* convert_ptr, void* prepare_ptr,
              const void* path_ptr, void* out_ptr, void* sig_ptr,
              uint64_t batch, uint64_t length, uint64_t dim, uint64_t degree,
              int method, bool matched_checks, uint64_t iterations) {
    using Direct = int (*)(const T*, T*, uint64_t, uint64_t, uint64_t, uint64_t);
    using Signature = int (*)(const T*, T*, uint64_t, uint64_t, uint64_t, uint64_t,
                             bool, bool, T, bool, bool, const T*, uint64_t, uint64_t, uint64_t);
    using Convert = int (*)(const T*, T*, uint64_t, uint64_t, uint64_t, int, bool);
    using Prepare = int (*)(uint64_t, uint64_t, int, bool);
    auto direct = reinterpret_cast<Direct>(direct_ptr);
    auto signature = reinterpret_cast<Signature>(signature_ptr);
    auto convert = reinterpret_cast<Convert>(convert_ptr);
    auto prepare = reinterpret_cast<Prepare>(prepare_ptr);
    auto path = static_cast<const T*>(path_ptr);
    auto out = static_cast<T*>(out_ptr);
    auto sig = static_cast<T*>(sig_ptr);
    int rc = synchronize();
    if (rc) return -(100000 + rc);
    auto start = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < iterations; ++i) {
        if (method != 3) {
            if (matched_checks && (rc = synchronize())) return -(100000 + rc);
            rc = signature(path, sig, batch, dim, length, degree, false, false, T(1), true, false, nullptr, 0, 0, 0);
            if (rc) return -rc;
            if (matched_checks && (rc = synchronize())) return -(100000 + rc);
        }
        if (matched_checks) {
            if ((rc = synchronize())) return -(100000 + rc);
            if ((rc = prepare(dim, degree, method, false))) return -rc;
            if ((rc = synchronize())) return -(100000 + rc);
            if ((rc = synchronize())) return -(100000 + rc);
        }
        rc = method == 3 ? direct(path, out, batch, length, dim, degree)
                         : convert(sig, out, batch, dim, degree, method, false);
        if (rc) return -rc;
        if (matched_checks && (rc = synchronize())) return -(100000 + rc);
        // Include completion of GPU work in every call, including the final one.
        if ((rc = synchronize())) return -(100000 + rc);
    }
    return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() / iterations;
}

extern "C" double native_cuda_repeat(void* direct, void* signature, void* convert, void* prepare,
                                      const void* path, void* out, void* sig,
                                      uint64_t batch, uint64_t length, uint64_t dim, uint64_t degree,
                                      int method, int is_float, int matched_checks, uint64_t iterations) {
    if (is_float) return repeat<float>(direct, signature, convert, prepare, path, out, sig, batch, length, dim, degree, method, matched_checks, iterations);
    return repeat<double>(direct, signature, convert, prepare, path, out, sig, batch, length, dim, degree, method, matched_checks, iterations);
}
