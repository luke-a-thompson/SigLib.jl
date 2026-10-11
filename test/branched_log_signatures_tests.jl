@testset "branched log signatures" begin
    path64 = reshape(Float64[0, 1], 2, 1)
    bsig = branched_sig(path64, 3)
    blog = branched_sig_to_log_sig(bsig, 1, 3)

    @test isapprox(blog, Float64[1, 0, 0, 0]; atol=1e-14)
    @test isapprox(branched_log_sig(path64, 3), blog; atol=1e-14)

    path32 = reshape(Float32[0, 1], 2, 1)
    blog32 = branched_log_sig(path32, 2)
    @test eltype(blog32) === Float32
    @test blog32 ≈ Float32[1, 0]

    paths = Array{Float64}(undef, 2, 2, 1)
    paths[1, :, 1] .= [0.0, 1.0]
    paths[2, :, 1] .= [0.0, 2.0]
    blogs = branched_log_sig(paths, 2)
    @test size(blogs) == (2, 2)
    @test blogs[1, :] ≈ [1, 0]
    @test blogs[2, :] ≈ [2, 0]

    @test_throws TypeError branched_log_sig(reshape([0, 1], 2, 1), 2)
    @test_throws ArgumentError branched_sig_to_log_sig(bsig, 1, -1)
end

@testset "branched log Python conventions" begin
    path = [0.0 0.0; 0.2 -0.3; 0.7 0.4; 0.5 0.8]
    for planar in (false, true)
        for scalar_term in (false, true)
            bsig = branched_sig(path, 3; planar=planar, scalar_term=scalar_term)
            expanded = branched_sig_to_log_sig(bsig, 2, 3; planar=planar, method=0)
            @test expanded ≈ branched_log_sig(path, 3; planar=planar, method=0, scalar_term=scalar_term)
            @test length(expanded) == length(bsig)
            scalar_term && @test expanded[1] == 0
        end
    end
    @test branched_log_sig_length(2, 3) == branched_sig_length(2, 3)
    @test branched_log_sig_length(0, 3; planar=true) == 0
    @test branched_log_sig_length(2, 0; planar=true) == 0
    @test_throws ArgumentError branched_log_sig_length(256, 0)
    @test_throws ArgumentError branched_log_sig_length(-1, 3)
    @test_throws ArgumentError branched_log_sig_length(2, -1)
    @test_throws ArgumentError branched_log_sig(path, 3; method=1)
    @test_throws ArgumentError branched_log_sig(path, 3; planar=true, method=4)
    @test_throws ArgumentError branched_sig_to_log_sig(ones(1), 2, 0; planar=true, method=3)
    @test_throws ArgumentError branched_sig_to_log_sig(ones(2), 2, 0; method=0)
    if !SigLib._modern_abi[]
        @test_throws ArgumentError branched_log_sig(path, 3; planar=true)
        @test_throws ArgumentError branched_log_sig_length(2, 3; planar=true)
        @test_throws ArgumentError prepare_branched_log_sig(2, 3, 2; planar=true)
    else
        @test_throws OverflowError branched_log_sig_length(255, 7; planar=true)
        for T in (Float32, Float64)
            x = T.(path)
            reference = branched_log_sig(x, 4; planar=true, method=2)
            @test reference ≈ branched_log_sig(x, 4; planar=true, method=3) atol=2e-6
            @test eltype(reference) === T
            @test length(reference) == branched_log_sig_length(2, 4; planar=true)
            @test length(reference) < branched_sig_length(2, 4; planar=true)
            @test branched_log_sig(x, 4; planar=true) ≈ branched_log_sig(x, 4; planar=true, method=1)
            for method in 1:3
                @test branched_log_sig(x, 4; planar=true, method=method, scalar_term=true) ≈
                    branched_log_sig(x, 4; planar=true, method=method)
                @test isempty(branched_log_sig(x, 0; planar=true, method=method, scalar_term=true))
                @test size(branched_log_sig(zeros(T, 0, 4, 2), 3; planar=true, method=method)) ==
                    (0, branched_log_sig_length(2, 3; planar=true))
            end
            batches = repeat(reshape(x, 1, 1, 4, 2), 2, 3, 1, 1)
            @test branched_log_sig(@view(batches[1, 2, :, :]), 4; planar=true) ≈
                branched_log_sig(batches, 4; planar=true)[1, 2, :]
            for method in 1:2, scalar_term in (false, true)
                bsig = branched_sig(x, 4; planar=true, scalar_term=scalar_term)
                @test branched_sig_to_log_sig(bsig, 2, 4; planar=true, method=method) ≈
                    branched_log_sig(x, 4; planar=true, method=method)
            end
        end
        for dimension in (0, 2), method in 1:2
            degree = dimension == 0 ? 3 : 0
            prepare_branched_log_sig(dimension, degree, method; planar=true)
            @test isempty(branched_sig_to_log_sig(ones(1), dimension, degree; planar=true, method=method))
            @test SigLib._branched_sig_to_log_sig_backprop(ones(1), Float64[], dimension, degree;
                planar=true, method=method) == zeros(1)
        end
        # First coordinates from pySigLib 4.0.0's NumPy API on the path above.
        @test branched_log_sig(path, 3; planar=true, method=2)[1:9] ≈
            [0.5, 0.8, 0.325, 0, 0.325, -0.325, 0, 0.03425, 0.05316666666666667]
        @test branched_log_sig(path, 3; planar=true, method=3, correction=Float64[]) ≈
            branched_log_sig(path, 3; planar=true, method=3)
        expected = branched_log_sig(path, 3; planar=true, method=2)
        workers = map(1:4) do worker
            Threads.@spawn for _ in 1:4
                clear_log_sig_cache()
                yield()
                branched_log_sig(path, 3; planar=true, method=2 + worker % 2) ≈ expected ||
                    error("Concurrent compressed branched log mismatch")
            end
        end
        foreach(fetch, workers)
        @test all(istaskdone, workers)
    end
end
