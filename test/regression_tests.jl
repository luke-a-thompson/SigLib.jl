@testset "validation before native calls" begin
    @test_throws OverflowError sig_length(2, 64)
    @test_throws OverflowError sig_length(typemax(UInt64), 0; time_aug=true)
    @test_throws OverflowError sig_length(typemax(UInt64), 0; lead_lag=true)
    @test_throws OverflowError log_sig_length(2, 65)
    @test_throws OverflowError log_sig_length(2, typemax(UInt64))
    @test log_sig_length(1, typemax(UInt64)) == 1
    @test_throws OverflowError log_sig_length(1, big(typemax(UInt64)) + 1)
    @test_throws OverflowError prepare_log_sig(2, 65, LogSigMethod.LyndonBasis)
    @test_throws ArgumentError prepare_branched_log_sig(2, 2, BranchedLogSigMethod.MkwWords)
    @test_throws ArgumentError sig_to_log_sig(fill(1.0), 1, 0)
    for method in (LogSigMethod.DirectBCH,)
        @test_throws ArgumentError logsig_to_sig(Float64[], 2, 0; method=method)
        @test_throws ArgumentError logsig_to_sig(zeros(3, 0), 2, 2; method=method)
        @test_throws ArgumentError sig_to_log_sig(zeros(6, 0), 2, 2; method=method)
    end
    @test_throws ArgumentError sig(zeros(2, 0), 2)
    @test_throws ArgumentError sig(zeros(0, 2), 2)
    @test_throws ArgumentError set_cache_dir("invalid\0directory")
    @test occursin("invalid argument", sprint(showerror, SigLibError(:example, 2)))
    @test occursin("status 99", sprint(showerror, SigLibError(:example, 99)))
end

@testset "batch layouts and algebra" begin
    for T in (Float32, Float64)
        paths = reshape(T.(sin.(1:48)), 2, 4, 3, 2)
        for f in (sig, branched_sig, log_sig, branched_log_sig)
            values = f(paths, 2)
            @test eltype(values) === T
            for i in 1:2, j in 1:3
                @test values[:, j, i] ≈ f(view(paths, :, :, j, i), 2)
            end
            @test size(f(zeros(T, 2, 4, 0, 2), 2))[2:end] == (0, 2)
        end
        path = T[0 0.5 1; 0 1 -0.25]
        for time_aug in (false, true), lead_lag in (false, true)
            options = (; time_aug, lead_lag, end_time=2.0)
            signature = sig(path, 3; options...)
            logarithm = log_sig(path, 3; options..., method=LogSigMethod.LyndonBasis)
            @test logarithm ≈ log_sig(path, 3; options..., method=LogSigMethod.DirectBCH)
            @test logsig_to_sig(logarithm, 2, 3; time_aug, lead_lag, method=LogSigMethod.LyndonBasis) ≈ signature
            tensor_log = log_sig(path, 3; options..., method=LogSigMethod.Expanded, scalar_term=true)
            @test logsig_to_sig(tensor_log, 2, 3; time_aug, lead_lag, method=LogSigMethod.Expanded) ≈ sig(path, 3; options..., scalar_term=true)
        end
        first = path[:, 1:2]
        second = path[:, 2:3]
        @test log_sig_combine(log_sig(first, 3; method=LogSigMethod.LyndonBasis), log_sig(second, 3; method=LogSigMethod.LyndonBasis), 2, 3) ≈ log_sig(path, 3; method=LogSigMethod.LyndonBasis)
        @test log_sig_join(log_sig(first, 3; method=LogSigMethod.LyndonBasis), vec(second[:, 2] - second[:, 1]), 2, 3) ≈ log_sig(path, 3; method=LogSigMethod.LyndonBasis)
        @test branched_sig_combine(branched_sig(first, 3), branched_sig(second, 3), 2, 3) ≈ branched_sig(path, 3)
    end
end

@testset "composed gradients and degree zero" begin
    path = [0.0 0.5 1.0; 0.0 1.0 -0.25]
    for method in instances(LogSigMethod.T), lead_lag in (false, true)
        f = x -> log_sig(x, 2; method, time_aug=true, lead_lag, end_time=2.0, scalar_term=true)
        y, pullback = rrule(log_sig, path, 2; method, time_aug=true, lead_lag, end_time=2.0, scalar_term=true)
        @test y ≈ f(path)
        ybar = collect(range(0.1, 0.5; length=length(y)))
        _test_pullback_input(f, path, ybar, pullback(ybar)[2])
        @test pullback(ZeroTangent())[2] isa ZeroTangent
    end
    batch = reshape(sin.(1:12), 2, 3, 2)
    y, pullback = rrule(log_sig, view(batch, :, :, :), 2; method=LogSigMethod.LyndonBasis)
    _test_pullback_input(x -> log_sig(x, 2; method=LogSigMethod.LyndonBasis), batch, ones(size(y)), pullback(ones(size(y)))[2])
    for f in (sig, branched_sig, log_sig, branched_log_sig), scalar_term in (false, true)
        y, pullback = rrule(f, path, 0; scalar_term)
        @test all(iszero, pullback(ones(size(y)))[2])
        @test length(y) == (scalar_term && f != log_sig ? 1 : 0)
    end
end

@testset "dense batch-last storage" begin
    for T in (Float32, Float64)
        path = reshape(T.(sin.(1:120)), 2, 5, 12)
        path_c, batch_shape, batch, dimension, len = SigLib._path_buffer(path)
        @test path_c === path
        @test (batch_shape, batch, dimension, len) == ((12,), 12, 2, 5)
        original = copy(path)
        coefficients = sig(path, 3)
        @test size(coefficients) == (sig_length(2, 3), 12)
        @test SigLib._signature_like_buffer(coefficients, T, sig_length(2, 3), "signature")[1] === coefficients
        @test SigLib._format_signature_output(coefficients, (12,)) === coefficients
        @test path == original
        for i in 1:12
            @test coefficients[:, i] ≈ sig(path[:, :, i], 3)
        end
        multi = reshape(path, 2, 5, 3, 4)
        multi_c = SigLib._path_buffer(multi)[1]
        @test multi_c === multi
        result = SigLib._format_signature_output(coefficients, (3, 4))
        @test size(result) == (sig_length(2, 3), 3, 4)
        @test pointer(result) == pointer(coefficients)
        corrections = zeros(T, 4, 4, 12)
        @test SigLib._correction_buffer(corrections, path_c, (12,), 12, 2, 5, 3, false).data === corrections
    end
end

@testset "contiguous CPU views share storage" begin
    for T in (Float32, Float64)
        storage = reshape(T.(sin.(1:240)), 2, 5, 3, 8)
        singleton_storage = reshape(storage, 2, 5, 1, 24)
        for path in (view(storage, :, :, 2, 3), view(storage, :, :, :, 3),
                     view(storage, :, :, :, 3:5), view(singleton_storage, :, :, 1:-1:1, 3:5))
            @test SigLib._path_buffer(path)[1] === path
            packed = copy(path)
            for time_aug in (false, true), lead_lag in (false, true)
                @test sig(path, 3; time_aug, lead_lag) ≈ sig(packed, 3; time_aug, lead_lag)
                @test branched_sig(path, 3; time_aug, lead_lag) ≈ branched_sig(packed, 3; time_aug, lead_lag)
                @test branched_log_sig(path, 3; time_aug, lead_lag) ≈ branched_log_sig(packed, 3; time_aug, lead_lag)
                for method in instances(LogSigMethod.T)
                    @test log_sig(path, 3; method, time_aug, lead_lag) ≈ log_sig(packed, 3; method, time_aug, lead_lag)
                end
            end
            coefficients = sig(path, 3)
            coefficient_storage = cat(coefficients, coefficients; dims=ndims(coefficients) + 1)
            coefficient_view = selectdim(coefficient_storage, ndims(coefficient_storage), 2)
            coefficient_buffer = SigLib._signature_like_buffer(coefficient_view, T, size(coefficients, 1), "signature")[1]
            @test pointer(coefficient_buffer) == pointer(coefficient_view)
            @test sig_to_log_sig(coefficient_view, size(path, 1), 3) ≈ log_sig(packed, 3)
        end
        for path in (view(storage, :, 1:2:5, :, 3:5), view(storage, :, :, :, 5:-1:3),
                     view(storage, [2, 1], :, :, 3:5), view(storage, 1:2:1, :, 2:2, 3:5))
            @test SigLib._path_buffer(path)[1] isa Array
            @test sig(path, 3) ≈ sig(copy(path), 3)
        end
        values = view(storage, :, :, :, 3)
        converted = SigLib._materialize(values, T === Float32 ? Float64 : Float32)
        @test eltype(converted) !== T
        @test converted ≈ values
    end
end

@testset "log signature return inference" begin
    for T in (Float32, Float64), shape in ((2, 5), (2, 5, 3), (2, 5, 3, 2))
        path = reshape(T.(sin.(1:prod(shape))), shape)
        @test (@inferred log_sig(path, 3)) isa Array{T,length(shape) - 1}
        @test (@inferred branched_log_sig(path, 3)) isa Array{T,length(shape) - 1}
    end
end

@testset "cache lifecycle under concurrent calls" begin
    mktempdir() do directory
        @test set_cache_dir(joinpath(directory, "native")) === nothing
        @test isdir(joinpath(directory, "native"))
        path = [0.0 0.5 1.0; 0.0 1.0 -0.25]
        expected_log = log_sig(path, 3; method=LogSigMethod.LyndonBasis)
        expected_branched = branched_log_sig(path, 3)
        tasks = map(1:8) do worker
            Threads.@spawn begin
                for iteration in 1:8
                    clear_log_sig_cache()
                    yield()
                    method = isodd(worker + iteration) ? LogSigMethod.LyndonBasis : LogSigMethod.DirectBCH
                    log_sig(path, 3; method) ≈ expected_log || error("Concurrent log signature mismatch")
                    branched_log_sig(path, 3) ≈ expected_branched || error("Concurrent branched log signature mismatch")
                end
                true
            end
        end
        @test all(fetch, tasks)
        @test clear_log_sig_cache() === nothing
        @test prepare_log_sig(2, 2, LogSigMethod.LyndonBasis; use_disk=true) === nothing
        @test clear_log_sig_cache(; use_disk=true) === nothing
    end
end
