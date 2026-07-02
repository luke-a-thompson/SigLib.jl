@testset "basic signatures" begin
    path64 = reshape(Float64[0, 1], 2, 1)
    @test sig(path64, 3) ≈ Float64[1, 1 / 2, 1 / 6]
    @test sig(path64, 3; scalar_term=true) ≈ Float64[1, 1, 1 / 2, 1 / 6]
    @test signature(path64, 3) ≈ sig(path64, 3)
    @test sig(view(reshape(collect(Float64, 0:3), 4, 1), 1:2, :), 2) ≈ Float64[1, 1 / 2]

    path32 = reshape(Float32[0, 1], 2, 1)
    sig32 = sig(path32, 2)
    @test eltype(sig32) === Float32
    @test sig32 ≈ Float32[1, 1 / 2]

    constant_path = reshape(Float64[2, 2, 2], 3, 1)
    @test sig(constant_path, 3) ≈ Float64[0, 0, 0]
    @test sig(constant_path, 3; scalar_term=true) ≈ Float64[1, 0, 0, 0]
    @test sig(path64, 2; time_aug=true) ≈ Float64[1, 1, 1 / 2, 1 / 2, 1 / 2, 1 / 2]
    @test sig(path64, 2; lead_lag=true) ≈ Float64[1, 1, 1 / 2, 0, 1, 1 / 2]

    paths = Array{Float64}(undef, 2, 2, 1)
    paths[1, :, 1] .= [0.0, 1.0]
    paths[2, :, 1] .= [0.0, 2.0]
    sigs = sig(paths, 2)
    @test size(sigs) == (2, 2)
    @test sigs[1, :] ≈ [1, 1 / 2]
    @test sigs[2, :] ≈ [2, 2]

    empty_batch = Array{Float64}(undef, 0, 2, 1)
    @test size(sig(empty_batch, 2)) == (0, 2)

    @test_throws TypeError sig(reshape([0, 1], 2, 1), 2)
    @test_throws ArgumentError sig(path64, 2; n_jobs=0)
    @test_throws ArgumentError sig(path64, 2; correction=Float64[])
    @test_throws ArgumentError sig(Array{Float64}(undef, 2, 0, 1), 2)
end
