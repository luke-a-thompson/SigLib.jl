# Check results, buffer identity, and overwriting independently of allocation counts.
function check_inplace(f, f!, path, degree, scratch...; kwargs...)
    expected = f(path, degree; kwargs...)
    out = fill!(similar(expected), NaN)
    @test f!(out, path, degree, scratch...; kwargs...) === out
    @test out ≈ expected
    clear_log_sig_cache()
    changed = path .* eltype(path)(0.7)
    @test f!(out, changed, degree, scratch...; kwargs...) === out
    @test out ≈ f(changed, degree; kwargs...)
end

@testset "in-place signatures" begin
    @test signature! === sig! && branched_signature! === branched_sig!
    for T in (Float32, Float64), shape in ((2, 4), (2, 4, 3), (2, 4, 2, 3)),
        degree in (0, 3), scalar_term in (false, true), time_aug in (false, true), lead_lag in (false, true)
        path = reshape(T.(sin.(1:prod(shape)) ./ 4), shape)
        options = (; time_aug, lead_lag, scalar_term, end_time=0.7)
        check_inplace(sig, sig!, path, degree; options..., horner=false)
        for planar in (false, true)
            check_inplace(branched_sig, branched_sig!, path, degree; options..., planar)
        end
    end
end

@testset "in-place logarithms" begin
    for T in (Float32, Float64), shape in ((2, 4), (2, 4, 3), (2, 4, 2, 3)),
        degree in (0, 3), scalar_term in (false, true), method in instances(LogSigMethod.T),
        time_aug in (false, true), lead_lag in (false, true)
        path = reshape(T.(sin.(1:prod(shape)) ./ 4), shape)
        options = (; scalar_term, time_aug, lead_lag, end_time=0.7)
        scratch = method == LogSigMethod.DirectBCH ? () :
            (similar(sig(path, degree; options..., scalar_term=method == LogSigMethod.Expanded && scalar_term)),)
        check_inplace(log_sig, log_sig!, path, degree, scratch...; options..., method)
    end
end

@testset "in-place branched logarithms" begin
    for T in (Float32, Float64), shape in ((2, 4), (2, 4, 3), (2, 4, 2, 3)),
        degree in (0, 3), scalar_term in (false, true), planar in (false, true)
        path = reshape(T.(sin.(1:prod(shape)) ./ 4), shape)
        methods = planar ? instances(BranchedLogSigMethod.T) : (BranchedLogSigMethod.Expanded,)
        for method in methods, time_aug in (false, true), lead_lag in (false, true)
            options = (; planar, scalar_term, time_aug, lead_lag, end_time=0.7)
            scratch = method == BranchedLogSigMethod.DirectBCH ? () : (similar(branched_sig(path, degree; options...)),)
            check_inplace(branched_log_sig, branched_log_sig!, path, degree, scratch...; options..., method)
        end
    end
end

@testset "in-place views, corrections and empty batches" begin
    for T in (Float32, Float64), shape in ((2, 4), (2, 4, 0), (2, 4, 2, 3))
        dense = reshape(T.(sin.(1:prod(shape))), shape)
        paths = (dense, selectdim(dense, 2, 1:2:4))
        for path in paths, time_aug in (false, true), scalar_term in (false, true),
            correction in (nothing, T[0.01, -0.02, 0.03, -0.04])
            options = (; time_aug, scalar_term, correction)
            check_inplace(sig, sig!, path, 3; options...)
            for method in (LogSigMethod.Expanded, LogSigMethod.LyndonWords, LogSigMethod.LyndonBasis)
                scratch = similar(sig(path, 3; options..., scalar_term=method == LogSigMethod.Expanded && scalar_term))
                check_inplace(log_sig, log_sig!, path, 3, scratch; options..., method)
            end
            for planar in (false, true)
                check_inplace(branched_sig, branched_sig!, path, 3; options..., planar)
                methods = planar ?
                    (BranchedLogSigMethod.Expanded, BranchedLogSigMethod.MkwWords, BranchedLogSigMethod.MkwBasis) : (BranchedLogSigMethod.Expanded,)
                scratch = similar(branched_sig(path, 3; options..., planar))
                for method in methods
                    check_inplace(branched_log_sig, branched_log_sig!, path, 3, scratch; options..., planar, method)
                end
            end
        end
    end
end

@testset "in-place buffer validation" begin
    path = [0.0 0.2 0.5; 0.0 -0.3 0.4]
    for (f, f!) in ((sig, sig!), (branched_sig, branched_sig!))
        out = f(path, 2)
        @test_throws DimensionMismatch f!(zeros(length(out) + 1), path, 2)
        @test_throws ArgumentError f!(Float32.(out), path, 2)
        @test_throws ArgumentError f!(@view(zeros(2length(out))[1:2:end]), path, 2)
        @test_throws ArgumentError f!(out, path, -1)
        @test_throws ArgumentError f!(out, path, 2; n_jobs=0)
        @test_throws ArgumentError f!(out, path, 2; correction=@view(out[1:4]))
        @test_throws ArgumentError f!(out, path, 2; correction=ones(4), lead_lag=true)
    end
    @test_throws ArgumentError sig!(reshape(path, 6), path, 2)
    @test_throws ArgumentError sig!(reshape(path, 6), @view(path[:, :]), 2)
    for (f, f!, signature) in ((log_sig, log_sig!, sig), (branched_log_sig, branched_log_sig!, branched_sig))
        out, scratch = f(path, 2), signature(path, 2)
        @test_throws ArgumentError f!(out, path, 2)
        @test_throws ArgumentError f!(scratch, path, 2, scratch; method=f === log_sig ? LogSigMethod.Expanded : BranchedLogSigMethod.Expanded)
        @test_throws DimensionMismatch f!(out, path, 2, zeros(length(scratch) + 1))
        @test_throws ArgumentError f!(out, path, 2, Float32.(scratch))
        @test_throws DimensionMismatch f!(zeros(length(out) + 1), path, 2, scratch)
        @test_throws ArgumentError f!(Float32.(out), path, 2, scratch)
        @test_throws ArgumentError f!(@view(zeros(2length(out))[1:2:end]), path, 2, scratch)
        @test_throws ArgumentError f!(out, path, -1, scratch)
        @test_throws ArgumentError f!(out, path, 2, scratch; n_jobs=0)
        @test_throws ArgumentError f!(out, path, 2, scratch; correction=@view(scratch[1:4]))
    end
    @test_throws ArgumentError log_sig!(zeros(2), path, 2; method=LogSigMethod.DirectBCH, correction=ones(4))
    out = branched_log_sig(path, 2)
    scratch = similar(branched_sig(path, 2))
    @test_throws ArgumentError branched_log_sig!(out, path, 2)
    @test_throws ArgumentError branched_log_sig!(out, path, 2, out)
    @test_throws DimensionMismatch branched_log_sig!(out, path, 2, zeros(length(scratch) + 1))
    @test_throws ArgumentError branched_log_sig!(out, path, 2, Float32.(scratch))
    @test_throws ArgumentError branched_log_sig!(out, path, 2, scratch; method=BranchedLogSigMethod.MkwWords)
    direct = branched_log_sig(path, 3; planar=true, method=BranchedLogSigMethod.DirectBCH)
    @test_throws ArgumentError branched_log_sig!(direct, path, 3; planar=true,
        method=BranchedLogSigMethod.DirectBCH, correction=ones(4))
end

@testset "in-place CPU allocations" begin
    function allocated_inplace(f!::F, out, path, args::Vararg{Any,N}; kwargs...) where {F,N}
        f!(out, path, args...; kwargs...)
        return @allocated f!(out, path, args...; kwargs...)
    end
    for T in (Float32, Float64)
        path = reshape(T.(sin.(1:120)), 2, 5, 12)
        for (f, f!) in ((sig, sig!), (branched_sig, branched_sig!))
            out = f(path, 3)
            @test allocated_inplace(f!, out, path, 3) == 0
            @test allocated_inplace(f!, out, path, 3; correction=T[0.01, 0.02, 0.03, 0.04]) == 0
        end
        for method in instances(LogSigMethod.T)
            out = log_sig(path, 3; method)
            scratch = method == LogSigMethod.DirectBCH ? () : (similar(sig(path, 3)),)
            @test allocated_inplace(log_sig!, out, path, 3, scratch...; method) == 0
            if method != LogSigMethod.DirectBCH
                @test allocated_inplace(log_sig!, out, path, 3, scratch...; method, correction=T[0.01, 0.02, 0.03, 0.04]) == 0
            end
        end
        for planar in (false, true), method in (planar ? instances(BranchedLogSigMethod.T) : (BranchedLogSigMethod.Expanded,))
            out = branched_log_sig(path, 3; planar, method)
            scratch = method == BranchedLogSigMethod.DirectBCH ? () : (similar(branched_sig(path, 3; planar)),)
            @test allocated_inplace(branched_log_sig!, out, path, 3, scratch...; planar, method) == 0
            if method != BranchedLogSigMethod.DirectBCH
                @test allocated_inplace(branched_log_sig!, out, path, 3, scratch...; planar, method, correction=T[0.01, 0.02, 0.03, 0.04]) == 0
            end
        end
    end
end

@testset "in-place log cache concurrency" begin
    path = reshape(sin.(1:120), 6, 5, 4)
    expected = log_sig(path, 3; method=LogSigMethod.LyndonBasis)
    tasks = map(1:4) do _
        Threads.@spawn begin
            out, scratch = similar(expected), similar(sig(path, 3))
            for _ in 1:8
                clear_log_sig_cache()
                yield()
                log_sig!(out, path, 3, scratch; method=LogSigMethod.LyndonBasis) ≈ expected ||
                    error("Concurrent log result mismatch")
            end
            true
        end
    end
    @test all(fetch, tasks)
end

@testset "in-place logs reuse dense and contiguous CPU storage" begin
    function allocated_log_call(out, path, degree, scratch, method)
        log_sig!(out, path, degree, scratch; method)
        return @allocated log_sig!(out, path, degree, scratch; method)
    end
    for T in (Float32, Float64), method in instances(LogSigMethod.T)
        for shape in ((3, 32), (3, 32, 4), (3, 32, 2, 2))
            path = reshape(T.(sin.(1:prod(shape))), shape)
            storage = cat(path, path; dims=length(shape) + 1)
            contiguous = selectdim(storage, ndims(storage), 2)
            expected = log_sig(path, 5; method)
            out = similar(expected)
            scratch = method == LogSigMethod.DirectBCH ? nothing : similar(sig(path, 5))
            for input in (path, contiguous)
                @test log_sig!(out, input, 5, scratch; method) === out
                @test out ≈ expected
                # The caller-owned API guarantees zero allocations for dense rank-3 paths.
                # Other shapes and views can allocate small buffer/alias wrappers.
                if input === path && ndims(path) == 3
                    allocated_log_call(out, input, 5, scratch, method)
                    @test allocated_log_call(out, input, 5, scratch, method) == 0
                end
            end
        end
    end
end
