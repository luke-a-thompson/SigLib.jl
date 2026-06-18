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

    function _finite_difference(path, idx, degree, sig_derivs; kwargs...)
        h = 1e-6
        plus = copy(path)
        minus = copy(path)
        plus[idx] += h
        minus[idx] -= h
        return (
            sum(sig(plus, degree; kwargs...) .* sig_derivs) -
            sum(sig(minus, degree; kwargs...) .* sig_derivs)
        ) / (2h)
    end

    function _test_sig_backprop_fd(path, degree, sig_derivs; atol=1e-7, rtol=1e-6, kwargs...)
        sig_value = sig(path, degree; kwargs...)
        path_derivs = sig_backprop(path, sig_value, sig_derivs, degree; kwargs...)
        @test size(path_derivs) == size(path)
        @test eltype(path_derivs) === eltype(path)
        for idx in CartesianIndices(path)
            @test isapprox(
                path_derivs[idx],
                _finite_difference(path, idx, degree, sig_derivs; kwargs...);
                atol=atol,
                rtol=rtol,
            )
        end
    end

    path_bp = [0.0 0.0; 0.5 1.0; 1.0 -0.25]
    _test_sig_backprop_fd(path_bp, 2, collect(0.2:0.1:0.7))
    _test_sig_backprop_fd(path_bp, 2, collect(0.1:0.1:0.7); scalar_term=true)

    path32_bp = Float32[0.0 0.0; 0.5 1.0; 1.0 -0.25]
    sig32_bp = sig(path32_bp, 2)
    grad32_bp = sig_backprop(path32_bp, sig32_bp, ones(Float32, size(sig32_bp)), 2)
    @test size(grad32_bp) == size(path32_bp)
    @test eltype(grad32_bp) === Float32

    paths_bp = Array{Float64}(undef, 2, 3, 1)
    paths_bp[1, :, 1] .= [0.0, 0.5, 1.0]
    paths_bp[2, :, 1] .= [1.0, 1.5, 2.5]
    _test_sig_backprop_fd(paths_bp, 2, [0.2 0.3; 0.5 0.7])

    path_time = reshape(Float64[0.0, 0.4, 1.0], 3, 1)
    _test_sig_backprop_fd(
        path_time, 2, collect(range(0.1, step=0.05, length=sig_length(1, 2; time_aug=true)));
        time_aug=true,
        end_time=2.0,
    )
    _test_sig_backprop_fd(
        path_time, 2, collect(range(0.1, step=0.05, length=sig_length(1, 2; lead_lag=true)));
        lead_lag=true,
    )

    paths = Array{Float64}(undef, 2, 2, 1)
    paths[1, :, 1] .= [0.0, 1.0]
    paths[2, :, 1] .= [0.0, 2.0]
    sigs = sig(paths, 2)
    @test size(sigs) == (2, 2)
    @test sigs[1, :] ≈ [1, 1 / 2]
    @test sigs[2, :] ≈ [2, 2]

    empty_batch = Array{Float64}(undef, 0, 2, 1)
    @test size(sig(empty_batch, 2)) == (0, 2)
    empty_sigs = sig(empty_batch, 2)
    @test size(sig_backprop(empty_batch, empty_sigs, zero(empty_sigs), 2)) == size(empty_batch)

    @test_throws TypeError sig(reshape([0, 1], 2, 1), 2)
    @test_throws ArgumentError sig(path64, 2; n_jobs=0)
    @test_throws ArgumentError sig(path64, 2; correction=Float64[])
    @test_throws ArgumentError sig_backprop(path64, sig(path64, 2), ones(Float64, 1), 2)
    @test_throws ArgumentError sig(Array{Float64}(undef, 2, 0, 1), 2)
end
