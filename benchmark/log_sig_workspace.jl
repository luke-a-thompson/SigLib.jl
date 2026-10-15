# Benchmark the public reusable workspace against the allocating API and native C.
# Run: julia --startup-file=no --project=. benchmark/log_sig_workspace.jl
include("log_sig_overhead.jl")

function investigate(degree)
    data = Vector{Float64}(undef, 4 * 32 * 3)
    ccall(FILL, Cvoid, (Ptr{Float64}, UInt64, Float64), data, length(data), 1.0)
    path = reshape(data, 3, 32, 4)
    ws = LogSigWorkspace(path, degree)
    @assert log_sig!(ws, path) ≈ log_sig(path, degree)
    signature = SigLib._cpsig_symbol(:signature_d)
    convert = SigLib._cpsig_symbol(:sig_to_log_sig_d)
    native = n -> native_repeat_julia(C_NULL, signature, convert, path,
        ws.logarithm, ws.signature, 4, 32, 3, degree, 1, n)
    variants = (
        "log_sig" => (() -> log_sig(path, degree)),
        "log_sig!" => (() -> log_sig!(ws, path)),
        "bare_ccall" => (() -> raw_call!(C_NULL, signature, convert,
            path, ws.logarithm, ws.signature, 4, 32, 3, degree, 1)),
    )
    for (_, f) in variants
        for _ in 1:100
            f()
        end
        julia_repeat(f, 100)
    end
    estimate = max(native(100), julia_repeat(first(variants).second, 100))
    iterations = clamp(ceil(Int, 50_000_000 / estimate), 20, 500_000)
    measurements = [Float64[] for _ in variants]
    native_times = Float64[]
    for sample in 1:21
        GC.gc()
        # Rotate order to avoid favoring a variant as CPU speed changes.
        for i in circshift(collect(0:length(variants)), sample)
            if i == 0
                push!(native_times, native(iterations))
            else
                push!(measurements[i], julia_repeat(variants[i].second, iterations))
            end
        end
    end
    c = median(native_times)
    @printf("%d,native_c,%.6f,0,0,0\n", degree, c / 1000)
    for (i, (name, f)) in enumerate(variants)
        ns = median(measurements[i])
        delta = median(measurements[i] .- native_times)
        @printf("%d,%s,%.6f,%.6f,%.3f,%d\n", degree, name, ns / 1000, delta / 1000, 100 * delta / c, @allocated(f()))
    end
    @assert (@allocated log_sig!(ws, path)) == 0
    flush(stdout)
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("# Julia ", VERSION, "; cpSIG library: ", NATIVE_LIBRARY)
    println("# Float64 paths=(3,32,4), method=1, n_jobs=1; warmed; 21 rotated samples of approximately 50ms per variant")
    println("# Workspace results reuse storage and are overwritten on each call; a workspace must not be used concurrently.")
    println("degree,variant,median_us,paired_overhead_us,paired_overhead_pct,julia_alloc_bytes")
    investigate(4)
    investigate(5)
end
