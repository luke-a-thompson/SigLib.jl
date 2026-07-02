@testset "log signatures" begin
    @test log_sig_length(1, 0) == 0
    @test log_sig_length(1, 3) == 1
    @test log_sig_length(2, 2) == 3
    @test log_sig_length(3, 4) == 32
    @test log_sig_length(3, 4; time_aug=true, lead_lag=true) == 728
    @test prepare_log_sig(2, 2, 2) === nothing
    @test clear_log_sig_cache() === nothing
    @test_throws ArgumentError log_sig_length(0, 2)
    @test_throws ArgumentError log_sig_length(2, -1)
    @test_throws ArgumentError prepare_log_sig(2, 2, 4)

    path64 = reshape(Float64[0, 1], 2, 1)
    @test isapprox(log_sig(path64, 3; method=0), Float64[1, 0, 0]; atol=1e-14)
    @test isapprox(
        log_sig(path64, 3; method=0, scalar_term=true),
        Float64[0, 1, 0, 0];
        atol=1e-14,
    )
    @test log_sig(path64, 3; method=1) ≈ Float64[1]
    @test log_sig(path64, 3; method=2) ≈ Float64[1]
    @test log_sig(path64, 3; method=3) ≈ Float64[1]

    path32 = reshape(Float32[0, 1], 2, 1)
    logsig32 = log_sig(path32, 2; method=2)
    @test eltype(logsig32) === Float32
    @test logsig32 ≈ Float32[1]

    sig64 = sig(path64, 3)
    logsig64 = sig_to_log_sig(sig64, 1, 3; method=2)
    @test logsig64 ≈ log_sig(path64, 3; method=2)
    @test logsig_to_sig(logsig64, 1, 3; method=2) ≈ sig64
    sig64_scalar = sig(path64, 3; scalar_term=true)
    logsig64_scalar = sig_to_log_sig(sig64_scalar, 1, 3; method=0)
    @test isapprox(
        logsig64_scalar,
        log_sig(path64, 3; method=0, scalar_term=true);
        atol=1e-14,
    )
    @test logsig_to_sig(logsig64_scalar, 1, 3; method=0) ≈ sig64_scalar

    path2 = [0.0 0.0; 0.5 1.0; 1.0 -0.25]
    @test log_sig(path2, 2; method=2) ≈ Float64[1, -0.25, -0.5625]
    @test log_sig(path2, 2; method=3) ≈ log_sig(path2, 2; method=2)
    @test log_sig(path64, 2; time_aug=true, method=3) ≈ Float64[1, 1, 0]

    paths = Array{Float64}(undef, 2, 2, 1)
    paths[1, :, 1] .= [0.0, 1.0]
    paths[2, :, 1] .= [0.0, 2.0]
    logsigs = log_sig(paths, 2; method=2)
    @test size(logsigs) == (2, 1)
    @test logsigs[:, 1] ≈ [1, 2]

    joined = log_sig_join(log_sig(path64, 3; method=2), [1.0], 1, 3)
    @test joined ≈ Float64[2]
    combined = log_sig_combine(log_sig(path64, 3; method=2), log_sig(path64, 3; method=2), 1, 3)
    @test combined ≈ Float64[2]

    logsig1 = log_sig(path2, 2; method=2)

    @test_throws TypeError log_sig(reshape([0, 1], 2, 1), 2)
    @test_throws ArgumentError log_sig(path64, 2; method=4)
    @test_throws ArgumentError sig_to_log_sig(sig64, 1, 3; method=3)
    @test_throws ArgumentError logsig_to_sig(logsig64, 1, 3; method=3)
    @test_throws ArgumentError log_sig_join(logsig1, [1.0], 2, 2)
end
