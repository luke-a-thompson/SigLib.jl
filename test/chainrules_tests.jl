using ChainRulesCore: NoTangent, ZeroTangent, rrule

function _finite_difference_input(f, x, idx, ybar; h=1e-6)
    plus = copy(x)
    minus = copy(x)
    plus[idx] += h
    minus[idx] -= h
    return (sum(f(plus) .* ybar) - sum(f(minus) .* ybar)) / (2h)
end

function _test_pullback_input(f, x, ybar, xbar; atol=1e-7, rtol=1e-6)
    @test size(xbar) == size(x)
    for idx in eachindex(x)
        @test isapprox(
            xbar[idx],
            _finite_difference_input(f, x, idx, ybar);
            atol=atol,
            rtol=rtol,
        )
    end
end

@testset "ChainRules extensions" begin
    path = [0.0 0.0; 0.5 1.0; 1.0 -0.25]

    sig_value, sig_pullback = rrule(sig, path, 2)
    sig_bar = collect(range(0.1, step=0.05, length=length(sig_value)))
    sig_tangents = sig_pullback(sig_bar)
    @test sig_tangents[1] isa NoTangent
    @test sig_tangents[3] isa NoTangent
    @test sig_pullback(ZeroTangent())[2] isa ZeroTangent
    _test_pullback_input(x -> sig(x, 2), path, sig_bar, sig_tangents[2])

    sig_scalar_value, sig_scalar_pullback = rrule(sig, path, 2; scalar_term=true)
    sig_scalar_bar = collect(range(0.1, step=0.04, length=length(sig_scalar_value)))
    sig_scalar_tangents = sig_scalar_pullback(sig_scalar_bar)
    _test_pullback_input(
        x -> sig(x, 2; scalar_term=true),
        path,
        sig_scalar_bar,
        sig_scalar_tangents[2],
    )

    path32 = Float32[0.0 0.0; 0.5 1.0; 1.0 -0.25]
    sig32_value, sig32_pullback = rrule(sig, path32, 2)
    sig32_tangents = sig32_pullback(ones(Float32, size(sig32_value)))
    @test size(sig32_tangents[2]) == size(path32)
    @test eltype(sig32_tangents[2]) === Float32

    bsig_value, bsig_pullback = rrule(branched_sig, path, 2)
    bsig_bar = collect(range(0.2, step=0.1, length=length(bsig_value)))
    bsig_tangents = bsig_pullback(bsig_bar)
    @test bsig_tangents[1] isa NoTangent
    @test bsig_tangents[3] isa NoTangent
    @test bsig_pullback(ZeroTangent())[2] isa ZeroTangent
    _test_pullback_input(x -> branched_sig(x, 2), path, bsig_bar, bsig_tangents[2])

    blog_value, blog_pullback = rrule(branched_sig_to_log_sig, bsig_value, 2, 2)
    blog_bar = collect(range(0.1, step=0.03, length=length(blog_value)))
    blog_tangents = blog_pullback(blog_bar)
    @test blog_tangents[1] isa NoTangent
    @test blog_tangents[3] isa NoTangent
    @test blog_tangents[4] isa NoTangent
    @test blog_pullback(ZeroTangent())[2] isa ZeroTangent
    _test_pullback_input(
        x -> branched_sig_to_log_sig(x, 2, 2),
        bsig_value,
        blog_bar,
        blog_tangents[2],
    )

    branched_log_value, branched_log_pullback = rrule(branched_log_sig, path, 2)
    branched_log_bar = collect(range(0.05, step=0.02, length=length(branched_log_value)))
    branched_log_tangents = branched_log_pullback(branched_log_bar)
    @test branched_log_tangents[1] isa NoTangent
    @test branched_log_tangents[3] isa NoTangent
    @test branched_log_pullback(ZeroTangent())[2] isa ZeroTangent
    _test_pullback_input(
        x -> branched_log_sig(x, 2),
        path,
        branched_log_bar,
        branched_log_tangents[2],
    )

    other_bsig = branched_sig([0.0 0.0; 0.25 0.5; 0.5 -0.5], 2)
    bcombine_value, bcombine_pullback = rrule(
        branched_sig_combine, bsig_value, other_bsig, 2, 2,
    )
    bcombine_bar = collect(range(0.15, step=0.04, length=length(bcombine_value)))
    bcombine_tangents = bcombine_pullback(bcombine_bar)
    @test bcombine_tangents[1] isa NoTangent
    @test bcombine_tangents[4] isa NoTangent
    @test bcombine_tangents[5] isa NoTangent
    @test bcombine_pullback(ZeroTangent())[2] isa ZeroTangent
    @test bcombine_pullback(ZeroTangent())[3] isa ZeroTangent
    _test_pullback_input(
        x -> branched_sig_combine(x, other_bsig, 2, 2),
        bsig_value,
        bcombine_bar,
        bcombine_tangents[2],
    )
    _test_pullback_input(
        x -> branched_sig_combine(bsig_value, x, 2, 2),
        other_bsig,
        bcombine_bar,
        bcombine_tangents[3],
    )

    sig_for_log = sig(path, 2)
    sig_to_log_value, sig_to_log_pullback = rrule(sig_to_log_sig, sig_for_log, 2, 2; method=2)
    sig_to_log_bar = collect(range(0.1, step=0.05, length=length(sig_to_log_value)))
    sig_to_log_tangents = sig_to_log_pullback(sig_to_log_bar)
    @test sig_to_log_tangents[1] isa NoTangent
    @test sig_to_log_tangents[3] isa NoTangent
    @test sig_to_log_tangents[4] isa NoTangent
    _test_pullback_input(
        x -> sig_to_log_sig(x, 2, 2; method=2),
        sig_for_log,
        sig_to_log_bar,
        sig_to_log_tangents[2],
    )

    logsig_value = log_sig(path, 2; method=2)
    logsig_to_sig_value, logsig_to_sig_pullback = rrule(
        logsig_to_sig, logsig_value, 2, 2; method=2,
    )
    logsig_to_sig_bar = collect(range(0.2, step=0.1, length=length(logsig_to_sig_value)))
    logsig_to_sig_tangents = logsig_to_sig_pullback(logsig_to_sig_bar)
    @test logsig_to_sig_tangents[1] isa NoTangent
    @test logsig_to_sig_tangents[3] isa NoTangent
    @test logsig_to_sig_tangents[4] isa NoTangent
    _test_pullback_input(
        x -> logsig_to_sig(x, 2, 2; method=2),
        logsig_value,
        logsig_to_sig_bar,
        logsig_to_sig_tangents[2],
    )

    for method in (2, 3)
        log_value, log_pullback = rrule(log_sig, path, 2; method=method)
        log_bar = collect(range(0.15, step=0.05, length=length(log_value)))
        log_tangents = log_pullback(log_bar)
        @test log_tangents[1] isa NoTangent
        @test log_tangents[3] isa NoTangent
        @test log_pullback(ZeroTangent())[2] isa ZeroTangent
        _test_pullback_input(
            x -> log_sig(x, 2; method=method),
            path,
            log_bar,
            log_tangents[2],
        )
    end

    other_logsig = log_sig([0.0 0.0; 0.25 0.5; 0.5 -0.5], 2; method=2)
    lcombine_value, lcombine_pullback = rrule(
        log_sig_combine, logsig_value, other_logsig, 2, 2,
    )
    lcombine_bar = collect(range(0.2, step=0.07, length=length(lcombine_value)))
    lcombine_tangents = lcombine_pullback(lcombine_bar)
    @test lcombine_tangents[1] isa NoTangent
    @test lcombine_tangents[4] isa NoTangent
    @test lcombine_tangents[5] isa NoTangent
    _test_pullback_input(
        x -> log_sig_combine(x, other_logsig, 2, 2),
        logsig_value,
        lcombine_bar,
        lcombine_tangents[2],
    )
    _test_pullback_input(
        x -> log_sig_combine(logsig_value, x, 2, 2),
        other_logsig,
        lcombine_bar,
        lcombine_tangents[3],
    )

    displacement = [0.25, -0.5]
    ljoin_value, ljoin_pullback = rrule(log_sig_join, logsig_value, displacement, 2, 2)
    ljoin_bar = collect(range(0.1, step=0.03, length=length(ljoin_value)))
    ljoin_tangents = ljoin_pullback(ljoin_bar)
    @test ljoin_tangents[1] isa NoTangent
    @test ljoin_tangents[4] isa NoTangent
    @test ljoin_tangents[5] isa NoTangent
    _test_pullback_input(
        x -> log_sig_join(x, displacement, 2, 2),
        logsig_value,
        ljoin_bar,
        ljoin_tangents[2],
    )
    _test_pullback_input(
        x -> log_sig_join(logsig_value, x, 2, 2),
        displacement,
        ljoin_bar,
        ljoin_tangents[3],
    )
end
