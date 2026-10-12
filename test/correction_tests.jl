# Independent truncated tensor series reference: each segment contributes exp(dx + c).
function _tensor_product_reference(a, b, d, degree)
    out = zeros(eltype(a), length(a))
    offsets = cumsum([0; [d^k for k in 0:degree]])
    for k in 0:degree, j in 0:k
        left = a[offsets[j + 1] + 1:offsets[j + 2]]
        right = b[offsets[k - j + 1] + 1:offsets[k - j + 2]]
        out[offsets[k + 1] + 1:offsets[k + 2]] .+= [x * y for x in left for y in right]
    end
    return out
end

function _corrected_signature_reference(path, degree, corrections)
    d = size(path, 2)
    identity = zeros(eltype(path), 1 + sum(d^k for k in 1:degree))
    identity[1] = 1
    result = copy(identity)
    for segment in 1:size(path, 1)-1
        lift = zeros(eltype(path), length(identity))
        lift[2:d + 1] = path[segment + 1, :] - path[segment, :]
        lift[d + 2:d + 1 + size(corrections, 2)] = corrections[segment, :]
        local_sig, term = copy(identity), copy(identity)
        for order in 1:degree
            term = _tensor_product_reference(term, lift, d, degree) ./ order
            local_sig .+= term
        end
        result = _tensor_product_reference(result, local_sig, d, degree)
    end
    return result[2:end]
end

@testset "corrected tensor signatures" begin
    for T in (Float32, Float64)
        path = T[0 0; 0.3 -0.2; -0.1 0.5; 0.2 0.4]
        for width in (4, 12, 28)
            rows = reshape(T.(sin.(1:3width) ./ 20), 3, width)
            expected = _corrected_signature_reference(path, 4, rows)
            for horner in (false, true)
                actual = sig(path, 4; correction=rows, horner=horner)
                @test eltype(actual) === T
                @test actual ≈ expected
                @test sig(path, 4; correction=rows, horner=horner, scalar_term=true) ≈ [one(T); expected]
            end
            constant = rows[1, :]
            @test sig(path, 4; correction=constant) ≈
                _corrected_signature_reference(path, 4, repeat(reshape(constant, 1, :), 3, 1))
        end

        # Only the original channel receives corrections when time is appended.
        p = reshape(T[0, 1], 2, 1)
        @test sig(p, 3; correction=T[0.2, 0.1]) ≈ T[1, 0.7, 1/6 + 0.2 + 0.1]
        augmented_path = T[0 0; 1 2]
        @test sig(p, 3; correction=T[0.2], time_aug=true, end_time=2) ≈
            _corrected_signature_reference(augmented_path, 3, reshape(T[0.2, 0, 0, 0], 1, 4))
        @test sig(zeros(T, 2, 1), 4; correction=T[0.2]) ≈ T[0, 0.2, 0, 0.02]
        @test log_sig(p, 3; correction=T[0.2, 0.1], method=0) ≈ T[1, 0.2, 0.1]
        @test branched_sig(zeros(T, 2, 1), 2; correction=T[0.2]) ≈ T[0, 0.2]
    end
end

@testset "correction layouts and signature families" begin
    families = (
        (sig, (;)), (log_sig, (; method=0)), (log_sig, (; method=1)), (log_sig, (; method=2)),
        (branched_sig, (;)), (branched_sig, (; planar=true)),
        (branched_log_sig, (;)), (branched_log_sig, (; planar=true, method=0)),
    )
    if SigLib._modern_abi[]
        families = (families..., (branched_log_sig, (; planar=true, method=1)),
            (branched_log_sig, (; planar=true, method=2)))
    end
    for T in (Float32, Float64), (f, options) in families, time_aug in (false, true)
        paths = reshape(T.(sin.(1:48) ./ 4), 2, 3, 4, 2)
        constant = T[0.01, -0.02, 0.03, -0.04]
        shared = reshape(T.(cos.(1:12) ./ 20), 3, 4)
        specific = reshape(T.(sin.(1:72) ./ 20), 2, 3, 3, 4)
        for correction in (constant, shared, specific), scalar_term in (false, true)
            kwargs = (; options..., correction, scalar_term, time_aug, end_time=0.7)
            batched = f(paths, 3; kwargs...)
            for i in 1:2, j in 1:3
                row = correction === specific ? specific[i, j, :, :] : correction
                expected = f(paths[i, j, :, :], 3; kwargs..., correction=row)
                @test batched[i, j, :] ≈ expected
            end
            # Views must pack exactly like contiguous arrays.
            wider = zeros(T, 2, 3, 3, 8)
            wider[:, :, :, 1:2:8] = specific
            @test f(paths, 3; kwargs..., correction=@view(wider[:, :, :, 1:2:8])) ≈
                f(paths, 3; kwargs..., correction=specific)
        end
        corrected = sig(paths, 3; correction=shared, time_aug=time_aug)
        @test log_sig(paths, 3; correction=shared, time_aug=time_aug, method=0) ≈
            sig_to_log_sig(corrected, 2, 3; time_aug=time_aug, method=0)
    end

    # Chen composition checks per-segment corrections in the branched algebra.
    path = [0.0 0.0; 0.3 -0.2; -0.1 0.5]
    rows = [0.01 -0.02 0.03 -0.04; -0.02 0.01 -0.01 0.03]
    for planar in (false, true)
        left = branched_sig(path[1:2, :], 3; correction=rows[1, :], planar=planar)
        right = branched_sig(path[2:3, :], 3; correction=rows[2, :], planar=planar)
        @test branched_sig(path, 3; correction=rows, planar=planar) ≈
            branched_sig_combine(left, right, 2, 3; planar=planar)
        @test branched_log_sig(path, 3; correction=rows, planar=planar, method=0) ≈
            branched_sig_to_log_sig(branched_sig(path, 3; correction=rows, planar=planar), 2, 3; planar=planar, method=0)
    end
    if SigLib._modern_abi[]
        for method in (1, 2)
            @test branched_log_sig(path, 3; correction=rows, planar=true, method=method) ≈
                branched_sig_to_log_sig(branched_sig(path, 3; correction=rows, planar=true), 2, 3; planar=true, method=method)
        end
    end
end

@testset "correction validation and boundaries" begin
    path = [0.0 0.0; 0.5 1.0; 1.0 -0.25]
    for f in (sig, log_sig, branched_sig, branched_log_sig)
        @test f(path, 2; correction=zeros(2, 4)) ≈ f(path, 2)
        for degree in (0, 1, 3), lead_lag in (false, true)
            @test f(path, degree; correction=Float64[], lead_lag=lead_lag) ≈ f(path, degree; lead_lag=lead_lag)
        end
        @test f(zeros(1, 2), 3; correction=ones(4)) ≈ f(zeros(1, 2), 3)
        @test size(f(zeros(0, 3, 2), 3; correction=ones(4))) == size(f(zeros(0, 3, 2), 3))
        @test size(f(zeros(0, 3, 2), 3; correction=zeros(0, 2, 4))) == size(f(zeros(0, 3, 2), 3))
        for correction in (1.0, "bad", ones(Int, 4), ones(Float32, 4), ones(3), ones(8), ones(13), ones(3, 4), ones(1, 2, 4), fill(1.0))
            @test_throws ArgumentError f(path, 3; correction=correction)
        end
        @test_throws ArgumentError f(path, 3; correction=ones(4), lead_lag=true)
        @test_throws ArgumentError f(path, 1; correction=ones(4))
        @test_throws ArgumentError f(path, 0; correction=ones(4))
    end
    @test_throws ArgumentError log_sig(path, 3; correction=ones(4), method=3)
    @test log_sig(path, 3; correction=Float64[], method=3) ≈ log_sig(path, 3; method=3)
    @test_throws ArgumentError rrule(log_sig, path, 3; correction=ones(4), method=3)
    if SigLib._modern_abi[]
        @test_throws ArgumentError branched_log_sig(path, 3; correction=ones(4), planar=true, method=3)
        @test_throws ArgumentError rrule(branched_log_sig, path, 3; correction=ones(4), planar=true, method=3)
    end
end

@testset "upstream nonplanar zero-correction discrepancy" begin
    # Verified with cpSIG 3.1 and 4.0: the correction exponential differs from
    # the straight-line fast path at degree 3. Forward native behavior unchanged.
    path = [0.0 0.0; 0.5 1.0; 1.0 -0.25]
    @test_broken branched_sig(path, 3; correction=zeros(2, 4)) ≈ branched_sig(path, 3)
    @test_broken branched_log_sig(path, 3; correction=zeros(2, 4)) ≈ branched_log_sig(path, 3)
end

@testset "corrected path pullbacks" begin
    path = [0.0 0.0; 0.3 -0.2; -0.1 0.5]
    paths = repeat(reshape(path, 1, 3, 2), 2, 1, 1)
    rows = [0.01 -0.02 0.03 -0.04; -0.02 0.01 -0.01 0.03]
    specific = repeat(reshape(rows, 1, 2, 4), 2, 1, 1)
    specific[2, :, :] .*= -2
    families = ((sig, (;)), (log_sig, (; method=0)), (log_sig, (; method=1)), (log_sig, (; method=2)),
        (branched_sig, (;)), (branched_sig, (; planar=true)),
        (branched_log_sig, (;)), (branched_log_sig, (; planar=true, method=0)))
    if SigLib._modern_abi[]
        families = (families..., (branched_log_sig, (; planar=true, method=1)),
            (branched_log_sig, (; planar=true, method=2)))
    end
    for (f, options) in families, (p, correction) in ((path, rows[1, :]), (path, rows), (paths, specific)), time_aug in (false, true)
        kwargs = (; options..., correction, time_aug, scalar_term=true, end_time=0.7)
        y, back = rrule(f, p, 3; kwargs...)
        @test y ≈ f(p, 3; kwargs...)
        ybar = reshape(cos.(1:length(y)), size(y))
        _test_pullback_input(x -> f(x, 3; kwargs...), p, ybar, back(ybar)[2])
        @test back(ZeroTangent())[2] isa ZeroTangent
    end
    for (f, options) in families
        y, back = rrule(f, Float32.(path), 3; options..., correction=Float32.(rows))
        @test eltype(back(ones(Float32, size(y)))[2]) === Float32
    end
end
