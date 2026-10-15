using SigLib, Libdl, Statistics, Printf

# No extra Julia packages or native benchmark framework required.
# Run: julia --startup-file=no --project=. benchmark/log_sig_overhead.jl
const BUILD = mktempdir()
const SOURCE = joinpath(@__DIR__, "log_sig_overhead.cpp")
run(`g++ -O3 -std=c++17 $SOURCE -ldl -o $(joinpath(BUILD, "native_bench"))`)
run(`g++ -O3 -std=c++17 -shared -fPIC -DHARNESS_SHARED $SOURCE -o $(joinpath(BUILD, "harness.so"))`)
const HARNESS = Libdl.dlopen(joinpath(BUILD, "harness.so"))
const REPEAT = Libdl.dlsym(HARNESS, :native_repeat)
const FILL = Libdl.dlsym(HARNESS, :fill_path)
const NATIVE_LIBRARY = Libdl.dlpath(SigLib._libcpsig[])

function julia_repeat(f, iterations)
    start = time_ns()
    for _ in 1:iterations
        f()
    end
    return Float64(time_ns() - start) / iterations
end

for T in (Float32, Float64)
    @eval function native_repeat_julia(direct, signature, convert, packed::Array{$T,3}, out, sig_buffer, batch, len, dim, degree, method, n)
        ccall(REPEAT, Float64,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{$T}, Ptr{$T}, Ptr{$T}, UInt64, UInt64, UInt64, UInt64, Cint, Cint, UInt64),
            direct, signature, convert, packed, out, sig_buffer, batch, len, dim, degree, method, $T == Float32, n)
    end
    @eval function raw_call!(direct, signature, convert, packed::Array{$T,3}, out, sig_buffer, batch, len, dim, degree, method)
        if method == 3
            return ccall(direct, Cint, (Ptr{$T}, Ptr{$T}, UInt64, UInt64, UInt64, UInt64, Cint), packed, out, batch, len, dim, degree, 1)
        end
        rc = ccall(signature, Cint,
            (Ptr{$T}, Ptr{$T}, UInt64, UInt64, UInt64, UInt64, Bool, Bool, $T, Bool, Bool, Cint, Ptr{$T}, UInt64, UInt64, UInt64),
            packed, sig_buffer, batch, dim, len, degree, false, false, $T(1), true, false, 1, C_NULL, 0, 0, 0)
        rc == 0 || error("signature failed: $rc")
        return ccall(convert, Cint, (Ptr{$T}, Ptr{$T}, UInt64, UInt64, UInt64, Bool, Bool, Cint, Bool, Cint),
            sig_buffer, out, batch, dim, degree, false, false, method, false, 1)
    end
end

function compare(::Type{T}, batch, len, dim, degree, method; single=false, samples=15) where T
    data = Vector{Float64}(undef, batch * len * dim)
    scale = dim == 2 && len == 129 ? 0.1 : 1.0
    ccall(FILL, Cvoid, (Ptr{Float64}, UInt64, Float64), data, length(data), scale)
    packed = reshape(T.(data), dim, len, batch)
    public_path = single ? packed[:, :, 1] : packed
    out = Matrix{T}(undef, log_sig_length(dim, degree), batch)
    sig_buffer = Matrix{T}(undef, sig_length(dim, degree), batch)
    suffix = T == Float32 ? "f" : "d"
    direct = SigLib._cpsig_symbol(Symbol("log_sig_from_path_", suffix))
    signature = SigLib._cpsig_symbol(Symbol("signature_", suffix))
    convert = SigLib._cpsig_symbol(Symbol("sig_to_log_sig_", suffix))
    prepare_log_sig(dim, degree, method)
    native = n -> native_repeat_julia(direct, signature, convert, packed, out, sig_buffer, batch, len, dim, degree, method, n)
    public_call = () -> log_sig(public_path, degree; method, n_jobs=1)
    # Measure bare ccall as well, so the language boundary can be distinguished from allocations.
    raw_call = () -> raw_call!(direct, signature, convert, packed, out, sig_buffer, batch, len, dim, degree, method)
    @assert native(1) > 0
    expected = single ? vec(copy(out)) : copy(out)
    actual = public_call()
    @assert isapprox(actual, expected; rtol=T == Float32 ? 1e-5 : 1e-12, atol=T == Float32 ? 1e-6 : 1e-12)
    @assert raw_call() == 0
    for _ in 1:100
        public_call()
        raw_call()
    end
    julia_repeat(public_call, 10)
    julia_repeat(raw_call, 10)
    estimate = max(native(100), julia_repeat(public_call, 100))
    iterations = clamp(ceil(Int, 50_000_000 / estimate), 20, 500_000)
    standalone_cache = mktempdir(BUILD)
    standalone = parse(Float64, read(`$(joinpath(BUILD, "native_bench")) $NATIVE_LIBRARY $standalone_cache $suffix $batch $len $dim $degree $method $iterations`, String))
    native_times, raw_times, public_times = Float64[], Float64[], Float64[]
    for sample in 1:samples
        GC.gc()
        # Alternate order to reduce drift; GC remains enabled during public calls.
        if isodd(sample)
            push!(native_times, native(iterations))
            push!(raw_times, julia_repeat(raw_call, iterations))
            push!(public_times, julia_repeat(public_call, iterations))
        else
            push!(public_times, julia_repeat(public_call, iterations))
            push!(raw_times, julia_repeat(raw_call, iterations))
            push!(native_times, native(iterations))
        end
        @assert native_times[end] > 0
    end
    bytes = @allocated public_call()
    shape = single ? "$(dim)x$(len)" : "$(dim)x$(len)x$(batch)"
    c, raw, j = median(native_times), median(raw_times), median(public_times)
    delta = median(public_times .- native_times)
    @printf("%s,%s,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%d,%.6f,%.6f\n",
        T, shape, degree, method, standalone / 1000, c / 1000, raw / 1000, j / 1000,
        delta / 1000, 100 * delta / c, bytes, quantile(public_times .- native_times, .25) / 1000,
        quantile(public_times .- native_times, .75) / 1000)
    flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("# Julia ", VERSION, "; threads=", Threads.nthreads(), "; native n_jobs=1; samples=15; target=50ms/sample")
    println("# Library: ", NATIVE_LIBRARY)
    println("# Native buffers preallocated; Julia public call includes allocations, preparation checks and GC; compilation and initial cache setup excluded.")
    println("dtype,public_shape,degree,method,standalone_c_us,c_us,bare_ccall_us,julia_us,paired_overhead_us,overhead_pct,julia_alloc_bytes,overhead_q25_us,overhead_q75_us")
    compare(Float64, 1, 3, 2, 2, 3; single=true)
    compare(Float64, 1, 3, 2, 2, 1; single=true)
    compare(Float64, 4, 32, 3, 4, 3)
    compare(Float64, 4, 32, 3, 5, 3)
    compare(Float64, 4, 32, 3, 4, 1)
    compare(Float64, 4, 32, 3, 5, 1)
    compare(Float32, 32, 129, 2, 6, 3)
    compare(Float64, 1, 129, 2, 8, 3; single=true)
end
