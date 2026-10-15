@testset "branched signatures" begin
    @test branched_sig_length(1, 0) == 0
    @test branched_sig_length(1, 0; scalar_term=true) == 1
    @test branched_sig_length(1, 3) == 4
    @test branched_sig_length(1, 3; scalar_term=true) == 5
    @test branched_sig_length(2, 2) == 6
    @test branched_sig_length(1, 3; planar=true) == 8
    @test branched_sig_length(1, 2; time_aug=true) == 6
    @test branched_sig_length(1, 2; lead_lag=true) == 6
    @test prepare_branched_sig(1, 3) === nothing
    @test_throws ArgumentError branched_sig_length(0, 2)
    @test_throws ArgumentError branched_sig_length(2, -1)

    path64 = reshape(Float64[0, 1], 1, 2)
    @test branched_sig(path64, 3) ≈ Float64[1, 1 / 2, 1 / 3, 1 / 6]
    @test branched_sig(path64, 3; scalar_term=true) ≈ Float64[1, 1, 1 / 2, 1 / 3, 1 / 6]
    @test branched_signature(path64, 3) ≈ branched_sig(path64, 3)
    # cpSIG 4 changed the planar basis; fixtures follow the pinned native version.
    planar_expected = SigLib._modern_abi[] ?
        Float64[1, 1 / 2, 1 / 2, 1 / 6, 1 / 3, 1 / 6, 1 / 6, 1 / 6] :
        Float64[1, 1 / 2, 1 / 2, 1 / 6, 1 / 4, 1 / 4, 1 / 3, 1 / 6]
    @test branched_sig(path64, 3; planar=true) ≈ planar_expected

    path32 = reshape(Float32[0, 1], 1, 2)
    bsig32 = branched_sig(path32, 2)
    @test eltype(bsig32) === Float32
    @test bsig32 ≈ Float32[1, 1 / 2]

    combined = branched_sig_combine(branched_sig(path64, 3), branched_sig(path64, 3), 1, 3)
    @test combined ≈ Float64[2, 2, 8 / 3, 4 / 3]

    paths = Array{Float64}(undef, 1, 2, 2)
    paths[1, :, 1] .= [0.0, 1.0]
    paths[1, :, 2] .= [0.0, 2.0]
    bsigs = branched_sig(paths, 2)
    @test size(bsigs) == (2, 2)
    @test bsigs[:, 1] ≈ [1, 1 / 2]

    empty_batch = Array{Float64}(undef, 1, 2, 0)
    @test size(branched_sig(empty_batch, 2)) == (2, 0)

    @test_throws TypeError branched_sig(reshape([0, 1], 1, 2), 2)
    @test_throws ArgumentError branched_sig(path64, 2; n_jobs=0)
    @test branched_sig(path64, 2; correction=Float64[]) ≈ branched_sig(path64, 2)
    @test_throws ArgumentError branched_sig(Array{Float64}(undef, 1, 0, 2), 2)
end
