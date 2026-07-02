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
