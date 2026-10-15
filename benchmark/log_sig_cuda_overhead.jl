# Requires CUDA.jl 5, a functional GPU, and matching CPU/CUDA artifacts or local overrides.
# Run: julia --project=<cuda-environment> benchmark/log_sig_cuda_overhead.jl
using SigLib, CUDA, Libdl, Statistics, Printf, Random

CUDA.functional() || error("CUDA benchmark requires a functional NVIDIA GPU")
CUDA.allowscalar(false)
const BUILD = mktempdir()
const CUDA_ROOT = get(ENV, "SIGLIB_BENCH_CUDA_ROOT", "/usr/local/cuda")
const SOURCE = joinpath(@__DIR__, "log_sig_cuda_overhead.cpp")
run(`g++ -O3 -std=c++17 -shared -fPIC -I$(joinpath(CUDA_ROOT, "include")) $SOURCE -L$(joinpath(CUDA_ROOT, "lib64")) -Wl,-rpath,$(joinpath(CUDA_ROOT, "lib64")) -lcudart -o $(joinpath(BUILD, "harness.so"))`)
const HARNESS = Libdl.dlopen(joinpath(BUILD, "harness.so"))
const REPEAT = Libdl.dlsym(HARNESS, :native_cuda_repeat)
const EXT = Base.get_extension(SigLib, :SigLibCUDAExt)

function julia_repeat(f, iterations)
    start = time_ns()
    for _ in 1:iterations
        f()
    end
    return Float64(time_ns() - start) / iterations
end

for T in (Float32, Float64)
    @eval function native_repeat_julia(direct, signature, convert, prepare, packed::CuArray{$T,3}, out, sig_buffer, batch, len, dim, degree, method, matched_checks, n)
        ns = ccall(REPEAT, Float64,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, UInt64, Cint, Cint, Cint, UInt64),
            direct, signature, convert, prepare, packed, out, sig_buffer, batch, len, dim, degree, method, $T == Float32, matched_checks, n)
        if ns < 0
            detail = unsafe_string(ccall(EXT._last_error[], Cstring, ()))
            error("native CUDA benchmark failed with status $(-ns): $detail")
        end
        return ns
    end
    @eval function raw_call!(direct, signature, convert, packed::CuArray{$T,3}, out, sig_buffer, batch, len, dim, degree, method)
        if method == 3
            rc = ccall(direct, Cint, (CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, UInt64), packed, out, batch, len, dim, degree)
        else
            rc = ccall(signature, Cint,
                (CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, UInt64, Bool, Bool, $T, Bool, Bool, CuPtr{$T}, UInt64, UInt64, UInt64),
                packed, sig_buffer, batch, dim, len, degree, false, false, $T(1), true, false, CuPtr{$T}(0), 0, 0, 0)
            rc == 0 || error("native signature failed: $rc")
            rc = ccall(convert, Cint, (CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, Cint, Bool),
                sig_buffer, out, batch, dim, degree, method, false)
        end
        rc == 0 || error("native log signature failed: $rc")
        CUDA.device_synchronize()
        return out
    end
end

function compare(::Type{T}, batch, len, dim, degree, method; samples=15) where T
    host = permutedims(rand(MersenneTwister(1), T, batch, len, dim) .* T(0.2) .- T(0.1), (3, 2, 1))
    path = CuArray(host)
    packed = path
    out = CuArray{T}(undef, log_sig_length(dim, degree), batch)
    sig_buffer = CuArray{T}(undef, method == 3 ? 0 : sig_length(dim, degree), batch)
    prepare_log_sig(dim, degree, method; device=:cuda)
    library = EXT._library()
    suffix = T == Float32 ? "f" : "d"
    direct = Libdl.dlsym(library, Symbol("log_sig_from_path_cuda_", suffix))
    signature = Libdl.dlsym(library, Symbol("signature_cuda_", suffix))
    convert = Libdl.dlsym(library, Symbol("sig_to_log_sig_cuda_", suffix))
    prepare = Libdl.dlsym(library, :prepare_log_sig_cuda)
    native = (n, matched) -> native_repeat_julia(direct, signature, convert, prepare, packed, out, sig_buffer, batch, len, dim, degree, method, matched, n)
    public_call = () -> begin
        result = log_sig(path, degree; method)
        CUDA.device_synchronize()
        result
    end
    wrapped_call = () -> begin
        if method == 3
            SigLib._prepare_log_sig_for(packed, dim, degree, 3)
            SigLib._log_sig_from_path_c!(out, packed, batch, len, dim, degree, 1)
        else
            SigLib._signature_c!(sig_buffer, packed, batch, dim, len, degree, false, false, true, false, 1, one(T))
            SigLib._prepare_log_sig_if_needed(packed, dim, degree, method)
            SigLib._sig_to_log_sig_c!(out, sig_buffer, batch, dim, degree, false, false, method, false, 1)
        end
        CUDA.device_synchronize()
        out
    end
    raw_call = () -> raw_call!(direct, signature, convert, packed, out, sig_buffer, batch, len, dim, degree, method)
    native(1, false)
    expected = Array(out)
    @assert isapprox(Array(public_call()), expected; rtol=T == Float32 ? 2e-4 : 2e-10, atol=T == Float32 ? 2e-5 : 2e-11)
    @assert isapprox(Array(wrapped_call()), expected)
    @assert isapprox(Array(raw_call()), expected)
    @assert isapprox(Array(public_call()), log_sig(host, degree; method); rtol=T == Float32 ? 2e-4 : 2e-10, atol=T == Float32 ? 2e-5 : 2e-11)
    ws = LogSigWorkspace(path, degree; method)
    workspace_call = () -> begin
        result = log_sig!(ws, path)
        CUDA.device_synchronize()
        result
    end
    @assert Array(workspace_call()) ≈ expected
    variants = ("bare_ccall" => raw_call, "preallocated_wrappers" => wrapped_call, "log_sig!" => workspace_call, "log_sig" => public_call)
    for (_, f) in variants
        for _ in 1:30
            f()
        end
        julia_repeat(f, 10)
    end
    native(10, false)
    native(10, true)
    estimate = max(native(10, false), julia_repeat(public_call, 10))
    iterations = clamp(ceil(Int, 50_000_000 / estimate), 5, 20_000)
    native_times, matched_times = Float64[], Float64[]
    measurements = [Float64[] for _ in variants]
    for sample in 1:samples
        GC.gc()
        CUDA.device_synchronize()
        # Rotate order so changing GPU clocks and host load do not favor a variant.
        for i in circshift(collect(0:length(variants)+1), sample)
            if i == 0
                push!(native_times, native(iterations, false))
            elseif i == length(variants)+1
                push!(matched_times, native(iterations, true))
            else
                push!(measurements[i], julia_repeat(variants[i].second, iterations))
            end
        end
    end
    shape = "$(dim)x$(len)x$(batch)"
    c = median(native_times)
    @printf("%s,%s,%d,%d,native_c,%.6f,0,0,0\n", T, shape, degree, method, c / 1000)
    @printf("%s,%s,%d,%d,native_c_matched_checks,%.6f,%.6f,%.3f,0\n", T, shape, degree, method, median(matched_times) / 1000,
        median(matched_times .- native_times) / 1000, 100 * median(matched_times .- native_times) / c)
    for (i, (name, f)) in enumerate(variants)
        delta = median(measurements[i] .- native_times)
        @printf("%s,%s,%d,%d,%s,%.6f,%.6f,%.3f,%d\n", T, shape, degree, method, name,
            median(measurements[i]) / 1000, delta / 1000, 100 * delta / c, @allocated(f()))
    end
    flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("# Julia ", VERSION, "; CUDA.jl ", pkgversion(CUDA), "; GPU: ", CUDA.name(CUDA.device()))
    println("# CPU library: ", Libdl.dlpath(SigLib._libcpsig[]))
    println("# CUDA library: ", Libdl.dlpath(EXT._library()))
    println("# Inputs stay on GPU; warmed caches and kernels; completed wall-clock work per call; no host/device copies in timed loops.")
    println("# Native C uses preallocated device buffers and synchronizes each completed log-signature call. Matched checks adds Julia's per-native-call barriers and repeated cache preparation.")
    println("# preallocated_wrappers uses existing internal CUDA wrappers; log_sig! uses the public CPU/CUDA workspace.")
    println("dtype,public_shape,degree,method,variant,median_us,paired_overhead_us,paired_overhead_pct,julia_host_alloc_bytes")
    for degree in (4, 5), method in (1, 3)
        compare(Float64, 4, 32, 3, degree, method)
    end
    for method in (1, 3)
        compare(Float32, 32, 129, 2, 6, method)
        compare(Float32, 256, 129, 2, 6, method)
    end
end
