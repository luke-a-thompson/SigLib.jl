using SigLib, CUDA, ChainRulesCore, Test

CUDA.functional() || error("CUDA tests require a working NVIDIA GPU")
CUDA.allowscalar(false)
@test Base.get_extension(SigLib, :SigLibCUDAExt)._libcusig[] == C_NULL

function parity(f, args...; gpu_args=map(x -> x isa AbstractArray ? CuArray(x) : x, args), kwargs...)
    cpu, cpu_pullback = rrule(f, args...; kwargs...)
    gpu_options = (; (key => (value isa AbstractArray ? CuArray(value) : value) for (key, value) in pairs(kwargs))...)
    gpu, gpu_pullback = rrule(f, gpu_args...; gpu_options...)
    T = eltype(cpu)
    atol, rtol = T == Float32 ? (2e-5, 2e-4) : (2e-11, 2e-10)
    @test gpu isa CuArray
    @test size(gpu) == size(cpu) && eltype(gpu) == T
    @test Array(gpu) ≈ cpu atol=atol rtol=rtol
    derivs = reshape(T.(cos.(1:length(cpu))), size(cpu))
    cpu_grads = cpu_pullback(derivs)
    gpu_grads = gpu_pullback(CuArray(derivs))
    for (c, g) in zip(cpu_grads, gpu_grads)
        if c isa AbstractArray
            @test g isa CuArray
            @test size(g) == size(c) && eltype(g) == eltype(c)
            @test Array(g) ≈ c atol=atol rtol=rtol
        else
            @test g isa typeof(c)
        end
    end
    return gpu
end

@testset "CUDA forward and pullback parity" begin
    for T in (Float32, Float64), shape in ((4, 2), (2, 3, 4, 2))
        path = reshape(T.(sin.(1:prod(shape)) ./ 4), shape)
        for time_aug in (false, true), lead_lag in (false, true)
            options = (; time_aug, lead_lag, end_time=0.7)
            for scalar_term in (false, true)
                parity(sig, path, 3; options..., scalar_term)
                parity(sig, path, 3; options..., scalar_term, horner=false)
                for method in 0:2
                    s = sig(path, 3; options..., scalar_term)
                    l = log_sig(path, 3; options..., scalar_term, method)
                    parity(log_sig, path, 3; options..., scalar_term, method)
                    parity(sig_to_log_sig, s, 2, 3; time_aug, lead_lag, method)
                    parity(logsig_to_sig, l, 2, 3; time_aug, lead_lag, method, scalar_term)
                end
                for planar in (false, true)
                    parity(branched_sig, path, 3; options..., planar, scalar_term)
                    b = branched_sig(path, 3; options..., planar, scalar_term)
                    for method in (planar ? (0:2) : (0:0))
                        parity(branched_log_sig, path, 3; options..., planar, scalar_term, method)
                        parity(branched_sig_to_log_sig, b, 2, 3; time_aug, lead_lag, planar, method)
                    end
                    parity(branched_sig_combine, b, b, 2, 3; time_aug, lead_lag, planar, scalar_term)
                end
            end
            parity(log_sig, path, 3; options..., method=3)
            parity(branched_log_sig, path, 3; options..., planar=true, method=3)
        end
        l = log_sig(path, 3; method=2)
        displacement = fill(T(0.2), size(path)[1:end-2]..., 2)
        parity(log_sig_combine, l, l, 2, 3)
        parity(log_sig_join, l, displacement, 2, 3)
    end
end

@testset "CUDA correction forward and pullback parity" begin
    for T in (Float32, Float64), time_aug in (false, true), scalar_term in (false, true)
        path = reshape(T.(sin.(1:48) ./ 4), 2, 3, 4, 2)
        constant = T.(cos.(1:12) ./ 20)
        shared = reshape(T.(sin.(1:36) ./ 20), 3, 12)
        specific = reshape(T.(cos.(1:216) ./ 20), 2, 3, 3, 12)
        for correction in (constant, shared, specific)
            options = (; correction, time_aug, scalar_term, end_time=0.7)
            parity(sig, path, 3; options...)
            parity(sig, path, 3; options..., horner=false)
            for method in 0:2
                parity(log_sig, path, 3; options..., method)
            end
            for planar in (false, true)
                parity(branched_sig, path, 3; options..., planar)
                for method in (planar ? (0:2) : (0:0))
                    parity(branched_log_sig, path, 3; options..., planar, method)
                end
            end
        end
    end
    path = [0.0 0.0; 0.3 -0.2; -0.1 0.5]
    gpu = CuArray(path)
    rows = reshape(sin.(1:16) ./ 20, 2, 8)
    correction = CuArray(rows)
    @test Array(sig(gpu, 3; correction=@view(correction[:, 1:2:8]))) ≈
        sig(path, 3; correction=@view(rows[:, 1:2:8]))
    @test_throws ArgumentError sig(gpu, 3; correction=ones(4))
    @test_throws ArgumentError sig(path, 3; correction=CuArray(ones(4)))
    @test_throws ArgumentError sig(gpu, 3; correction=CuArray(ones(4)), lead_lag=true)
    @test_throws ArgumentError log_sig(gpu, 3; correction=CuArray(ones(4)), method=3)
    @test_throws ArgumentError branched_log_sig(gpu, 3; correction=CuArray(ones(4)), planar=true, method=3)
end

@testset "CUDA buffers, boundaries and cache lifecycle" begin
    path = reshape(sin.(1:48) ./ 4, 3, 8, 2)
    gpu = CuArray(path)
    for f in (sig, log_sig, branched_sig, branched_log_sig)
        parity(f, @view(path[:, 1:2:8, :]), 3; gpu_args=(@view(gpu[:, 1:2:8, :]), 3))
        empty_batch = f(CuArray(zeros(Float32, 0, 4, 2)), 3)
        @test empty_batch isa CuArray && size(empty_batch, 1) == 0
        parity(f, path, 0)
        parity(f, path, 0; scalar_term=true)
        parity(f, zeros(1, 2), 3)
    end
    @test Array(sig(gpu .+ 0.1, 3)) ≈ sig(path .+ 0.1, 3)
    for method in 0:3
        parity(log_sig, path, 0; method)
        parity(branched_log_sig, path, 0; planar=true, method)
    end
    mixed = log_sig(Float32.(path), 3; method=2)
    parity(log_sig_combine, mixed, Float64.(mixed), 2, 3)
    parity(log_sig_join, mixed, fill(0.2, 3, 2), 2, 3)
    l = log_sig(gpu, 3; method=2)
    @test_throws ArgumentError log_sig_combine(l, Array(l), 2, 3)
    @test_throws ArgumentError log_sig_combine(Array(l), l, 2, 3)
    @test_throws ArgumentError log_sig_join(l, zeros(3, 2), 2, 3)
    @test_throws ArgumentError SigLib._sig_backprop(gpu, sig(gpu, 3), ones(3, 14), 3)
    @test_throws TypeError sig(CuArray(ones(Int, 4, 2)), 3)
    @test_throws ArgumentError sig(gpu, 3; correction=CuArray(ones(1)))
    @test_throws ArgumentError clear_log_sig_cache(; device=:unknown)
    @test_throws ArgumentError branched_log_sig(gpu, 21; planar=true, method=3)
    @test_throws SigLibError prepare_branched_log_sig(2, 21, 3; planar=true, device=:cuda)
    clear_log_sig_cache(; device=:both)
    @test Array(log_sig(gpu, 3; method=2)) ≈ log_sig(path, 3; method=2)
    mktempdir() do directory
        set_cache_dir(directory; device=:both)
        prepare_log_sig(2, 3, 2; use_disk=true, device=:both)
        prepare_branched_sig(2, 3; use_disk=true, device=:both)
        prepare_branched_log_sig(2, 3, 2; planar=true, use_disk=true, device=:both)
        clear_log_sig_cache(; use_disk=true, device=:both)
    end
    expected = log_sig(path, 3; method=3)
    tasks = [Threads.@spawn begin
        for _ in 1:3
            clear_log_sig_cache(; device=:cuda)
            @test Array(log_sig(gpu, 3; method=3)) ≈ expected
            GC.gc()
        end
    end for _ in 1:3]
    foreach(fetch, tasks)
end

if length(collect(CUDA.devices())) > 1
    @testset "CUDA input context overrides the active device" begin
        path = [0.0 0.0; 0.3 -0.2; -0.1 0.5]
        first_device = CuArray(path)
        l = log_sig(first_device, 3; method=2)
        CUDA.device!(collect(CUDA.devices())[2]) do
            active = CUDA.context()
            result = sig(first_device, 3)
            @test CUDA.context() == active
            @test CUDA.context(result) == CUDA.context(first_device)
            @test Array(result) ≈ sig(path, 3)
            other = CuArray(Array(l))
            @test_throws ArgumentError log_sig_combine(l, other, 2, 3)
        end
    end
end
