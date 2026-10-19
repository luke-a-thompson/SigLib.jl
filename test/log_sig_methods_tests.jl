@testset "log signature method enums" begin
    path = [0.0 0.2 0.7; 0.0 -0.3 0.4]
    signature = sig(path, 3)
    branched = branched_sig(path, 3)

    @test Cint.((LogSigMethod.Expanded, LogSigMethod.LyndonWords,
        LogSigMethod.LyndonBasis, LogSigMethod.DirectBCH)) == (0, 1, 2, 3)
    @test Cint.((BranchedLogSigMethod.Expanded, BranchedLogSigMethod.MkwWords,
        BranchedLogSigMethod.MkwBasis, BranchedLogSigMethod.DirectBCH)) == (0, 1, 2, 3)

    @test log_sig(path, 3) == log_sig(path, 3; method=LogSigMethod.LyndonWords)
    @test sig_to_log_sig(signature, 2, 3) ==
        sig_to_log_sig(signature, 2, 3; method=LogSigMethod.LyndonWords)
    @test log_sig_basis(2, 3) == log_sig_basis(2, 3; method=LogSigMethod.LyndonWords)
    @test LogSigWorkspace(path, 3).method === LogSigMethod.LyndonWords
    @test branched_log_sig(path, 3) ==
        branched_log_sig(path, 3; method=BranchedLogSigMethod.Expanded)
    @test branched_sig_to_log_sig(branched, 2, 3) ==
        branched_sig_to_log_sig(branched, 2, 3; method=BranchedLogSigMethod.Expanded)
    @test branched_log_sig_basis(2, 3; planar=true) ==
        branched_log_sig_basis(2, 3; planar=true, method=BranchedLogSigMethod.MkwWords)

    for method in instances(LogSigMethod.T)
        @test prepare_log_sig(2, 3, method) === nothing
        @test LogSigWorkspace(path, 3; method).method === method
    end
    for method in instances(BranchedLogSigMethod.T)
        @test prepare_branched_log_sig(2, 3, method; planar=true) === nothing
    end
    planar_sig = branched_sig(path, 3; planar=true)
    @test branched_sig_to_log_sig(planar_sig, 2, 3; planar=true) ≈ branched_log_sig(path, 3; planar=true)
    @test branched_log_sig(path, 3; planar=true) ==
        branched_log_sig(path, 3; planar=true, method=BranchedLogSigMethod.MkwWords)

    # Distinguish the two method families even when their native values agree.
    for method in (nothing, 0, 1, 2, 3, BranchedLogSigMethod.Expanded, BranchedLogSigMethod.MkwBasis)
        @test_throws TypeError log_sig(path, 3; method)
        @test_throws TypeError sig_to_log_sig(signature, 2, 3; method)
        @test_throws TypeError logsig_to_sig(Float64[], 2, 0; method)
        @test_throws TypeError log_sig_basis(2, 3; method)
        @test_throws TypeError LogSigWorkspace(path, 3; method)
        @test_throws MethodError prepare_log_sig(2, 3, method)
    end
    for method in (nothing, 0, 1, 2, 3, LogSigMethod.Expanded, LogSigMethod.LyndonBasis)
        @test_throws TypeError branched_log_sig(path, 3; method)
        @test_throws TypeError branched_sig_to_log_sig(branched, 2, 3; method)
        @test_throws TypeError branched_log_sig_basis(2, 3; method)
        @test_throws MethodError prepare_branched_log_sig(2, 3, method)
    end
end
