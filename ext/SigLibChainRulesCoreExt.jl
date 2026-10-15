module SigLibChainRulesCoreExt

using SigLib
import ChainRulesCore: NoTangent, ProjectTo, ZeroTangent, rrule, unthunk

_is_zero_tangent(x) = x isa NoTangent || x isa ZeroTangent

function rrule(
    ::typeof(SigLib.sig),
    path::AbstractArray{T,N},
    degree::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    end_time::Real=1.0,
    horner::Bool=true,
    scalar_term::Bool=false,
    correction=nothing,
    n_jobs::Integer=1,
) where {T<:Union{Float32,Float64},N}
    y = SigLib.sig(
        path, degree;
        time_aug=time_aug,
        lead_lag=lead_lag,
        end_time=end_time,
        horner=horner,
        scalar_term=scalar_term,
        correction=correction,
        n_jobs=n_jobs,
    )
    project_path = ProjectTo(path)

    function sig_pullback(ybar)
        ybar = unthunk(ybar)
        if _is_zero_tangent(ybar)
            return NoTangent(), ZeroTangent(), NoTangent()
        end

        pathbar = SigLib._sig_backprop(
            path, y, ybar, degree;
            time_aug=time_aug,
            lead_lag=lead_lag,
            end_time=end_time,
            scalar_term=scalar_term,
            correction=correction,
            n_jobs=n_jobs,
        )
        return NoTangent(), project_path(pathbar), NoTangent()
    end

    return y, sig_pullback
end

function rrule(
    ::typeof(SigLib.branched_sig),
    path::AbstractArray{T,N},
    max_nodes::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    end_time::Real=1.0,
    planar::Bool=false,
    scalar_term::Bool=false,
    correction=nothing,
    n_jobs::Integer=1,
) where {T<:Union{Float32,Float64},N}
    y = SigLib.branched_sig(
        path, max_nodes;
        time_aug=time_aug,
        lead_lag=lead_lag,
        end_time=end_time,
        planar=planar,
        scalar_term=scalar_term,
        correction=correction,
        n_jobs=n_jobs,
    )
    project_path = ProjectTo(path)

    function branched_sig_pullback(ybar)
        ybar = unthunk(ybar)
        if _is_zero_tangent(ybar)
            return NoTangent(), ZeroTangent(), NoTangent()
        end

        pathbar = SigLib._branched_sig_backprop(
            path, y, ybar, max_nodes;
            time_aug=time_aug,
            lead_lag=lead_lag,
            end_time=end_time,
            planar=planar,
            scalar_term=scalar_term,
            correction=correction,
            n_jobs=n_jobs,
        )
        return NoTangent(), project_path(pathbar), NoTangent()
    end

    return y, branched_sig_pullback
end

function rrule(
    ::typeof(SigLib.sig_to_log_sig),
    sig_value::AbstractArray{T,N},
    dimension::Integer,
    degree::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    method::Integer=1,
    n_jobs::Integer=1,
) where {T<:Union{Float32,Float64},N}
    y = SigLib.sig_to_log_sig(
        sig_value, dimension, degree;
        time_aug=time_aug,
        lead_lag=lead_lag,
        method=method,
        n_jobs=n_jobs,
    )
    project_sig = ProjectTo(sig_value)

    function sig_to_log_sig_pullback(ybar)
        ybar = unthunk(ybar)
        if _is_zero_tangent(ybar)
            return NoTangent(), ZeroTangent(), NoTangent(), NoTangent()
        end

        sigbar = SigLib._sig_to_log_sig_backprop(
            sig_value, ybar, dimension, degree;
            time_aug=time_aug,
            lead_lag=lead_lag,
            method=method,
            n_jobs=n_jobs,
        )
        return NoTangent(), project_sig(sigbar), NoTangent(), NoTangent()
    end

    return y, sig_to_log_sig_pullback
end

function rrule(
    ::typeof(SigLib.logsig_to_sig),
    log_sig_value::AbstractArray{T,N},
    dimension::Integer,
    degree::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    method::Integer=1,
    scalar_term::Bool=false,
    n_jobs::Integer=1,
) where {T<:Union{Float32,Float64},N}
    y = SigLib.logsig_to_sig(
        log_sig_value, dimension, degree;
        time_aug=time_aug,
        lead_lag=lead_lag,
        method=method,
        scalar_term=scalar_term,
        n_jobs=n_jobs,
    )
    project_log_sig = ProjectTo(log_sig_value)

    function logsig_to_sig_pullback(ybar)
        ybar = unthunk(ybar)
        if _is_zero_tangent(ybar)
            return NoTangent(), ZeroTangent(), NoTangent(), NoTangent()
        end

        log_sigbar = SigLib._logsig_to_sig_backprop(
            log_sig_value, ybar, dimension, degree;
            time_aug=time_aug,
            lead_lag=lead_lag,
            method=method,
            scalar_term=scalar_term,
            n_jobs=n_jobs,
        )
        return NoTangent(), project_log_sig(log_sigbar), NoTangent(), NoTangent()
    end

    return y, logsig_to_sig_pullback
end

function rrule(
    ::typeof(SigLib.log_sig),
    path::AbstractArray{T,N},
    degree::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    end_time::Real=1.0,
    method::Integer=1,
    scalar_term::Bool=false,
    correction=nothing,
    n_jobs::Integer=1,
) where {T<:Union{Float32,Float64},N}
    SigLib._as_log_sig_method(method)
    if method != 3
        sig_value, sig_pullback = rrule(
            SigLib.sig, path, degree;
            time_aug=time_aug, lead_lag=lead_lag, end_time=end_time,
            scalar_term=method == 0 && scalar_term, correction=correction, n_jobs=n_jobs,
        )
        y, convert_pullback = rrule(
            SigLib.sig_to_log_sig, sig_value, size(path, 1), degree;
            time_aug=time_aug, lead_lag=lead_lag, method=method, n_jobs=n_jobs,
        )
        return y, ybar -> sig_pullback(convert_pullback(ybar)[2])
    end
    y = SigLib.log_sig(
        path, degree;
        time_aug=time_aug,
        lead_lag=lead_lag,
        end_time=end_time,
        method=method,
        scalar_term=scalar_term,
        correction=correction,
        n_jobs=n_jobs,
    )
    project_path = ProjectTo(path)

    function log_sig_pullback(ybar)
        ybar = unthunk(ybar)
        if _is_zero_tangent(ybar)
            return NoTangent(), ZeroTangent(), NoTangent()
        end

        pathbar = SigLib._log_sig_from_path_backprop(
            path, ybar, degree;
            time_aug=time_aug,
            lead_lag=lead_lag,
            end_time=end_time,
            n_jobs=n_jobs,
        )
        return NoTangent(), project_path(pathbar), NoTangent()
    end

    return y, log_sig_pullback
end

function rrule(
    ::typeof(SigLib.log_sig_combine),
    log_sig1::AbstractArray{T,N},
    log_sig2::AbstractArray,
    dimension::Integer,
    degree::Integer;
    n_jobs::Integer=1,
) where {T<:Union{Float32,Float64},N}
    y = SigLib.log_sig_combine(log_sig1, log_sig2, dimension, degree; n_jobs=n_jobs)
    project_log_sig1 = ProjectTo(log_sig1)
    project_log_sig2 = ProjectTo(log_sig2)

    function log_sig_combine_pullback(ybar)
        ybar = unthunk(ybar)
        if _is_zero_tangent(ybar)
            return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent(), NoTangent()
        end

        log_sig1bar, log_sig2bar = SigLib._log_sig_combine_backprop(
            log_sig1, log_sig2, ybar, dimension, degree; n_jobs=n_jobs,
        )
        return (
            NoTangent(),
            project_log_sig1(log_sig1bar),
            project_log_sig2(log_sig2bar),
            NoTangent(),
            NoTangent(),
        )
    end

    return y, log_sig_combine_pullback
end

function rrule(
    ::typeof(SigLib.log_sig_join),
    log_sig_value::AbstractArray{T,N},
    displacement::AbstractArray,
    dimension::Integer,
    degree::Integer;
    n_jobs::Integer=1,
) where {T<:Union{Float32,Float64},N}
    y = SigLib.log_sig_join(log_sig_value, displacement, dimension, degree; n_jobs=n_jobs)
    project_log_sig = ProjectTo(log_sig_value)
    project_displacement = ProjectTo(displacement)

    function log_sig_join_pullback(ybar)
        ybar = unthunk(ybar)
        if _is_zero_tangent(ybar)
            return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent(), NoTangent()
        end

        log_sigbar, displacementbar = SigLib._log_sig_join_backprop(
            log_sig_value, displacement, ybar, dimension, degree; n_jobs=n_jobs,
        )
        return (
            NoTangent(),
            project_log_sig(log_sigbar),
            project_displacement(displacementbar),
            NoTangent(),
            NoTangent(),
        )
    end

    return y, log_sig_join_pullback
end

function rrule(
    ::typeof(SigLib.branched_sig_to_log_sig),
    bsig::AbstractArray{T,N},
    dimension::Integer,
    degree::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    planar::Bool=false,
    method::Union{Nothing,Integer}=nothing,
    n_jobs::Integer=1,
) where {T<:Union{Float32,Float64},N}
    y = SigLib.branched_sig_to_log_sig(
        bsig, dimension, degree;
        time_aug=time_aug,
        lead_lag=lead_lag,
        planar=planar,
        method=method,
        n_jobs=n_jobs,
    )
    project_bsig = ProjectTo(bsig)

    function branched_sig_to_log_sig_pullback(ybar)
        ybar = unthunk(ybar)
        if _is_zero_tangent(ybar)
            return NoTangent(), ZeroTangent(), NoTangent(), NoTangent()
        end

        bsigbar = SigLib._branched_sig_to_log_sig_backprop(
            bsig, ybar, dimension, degree;
            time_aug=time_aug,
            lead_lag=lead_lag,
            planar=planar,
            method=method,
            n_jobs=n_jobs,
        )
        return NoTangent(), project_bsig(bsigbar), NoTangent(), NoTangent()
    end

    return y, branched_sig_to_log_sig_pullback
end

function rrule(
    ::typeof(SigLib.branched_log_sig),
    path::AbstractArray{T,N},
    degree::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    end_time::Real=1.0,
    planar::Bool=false,
    scalar_term::Bool=false,
    method::Union{Nothing,Integer}=nothing,
    correction=nothing,
    n_jobs::Integer=1,
) where {T<:Union{Float32,Float64},N}
    m = SigLib._branched_log_sig_method(method, planar)
    if m == 3
        y = SigLib.branched_log_sig(path, degree;
            time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, planar=planar,
            scalar_term=scalar_term, method=m, correction=correction, n_jobs=n_jobs)
        project_path = ProjectTo(path)
        function direct_pullback(ybar)
            ybar = unthunk(ybar)
            _is_zero_tangent(ybar) && return NoTangent(), ZeroTangent(), NoTangent()
            pathbar = SigLib._branched_log_sig_from_path_backprop(path, ybar, degree;
                time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, n_jobs=n_jobs)
            return NoTangent(), project_path(pathbar), NoTangent()
        end
        return y, direct_pullback
    end
    bsig, sig_pullback = rrule(SigLib.branched_sig,
        path, degree;
        time_aug=time_aug,
        lead_lag=lead_lag,
        end_time=end_time,
        planar=planar,
        scalar_term=scalar_term,
        correction=correction,
        n_jobs=n_jobs,
    )
    dimension = size(path, 1)
    y, convert_pullback = rrule(SigLib.branched_sig_to_log_sig,
        bsig, dimension, degree;
        time_aug=time_aug,
        lead_lag=lead_lag,
        planar=planar,
        method=m,
        n_jobs=n_jobs,
    )
    return y, ybar -> sig_pullback(convert_pullback(ybar)[2])
end

function rrule(
    ::typeof(SigLib.branched_sig_combine),
    bsig1::AbstractArray{T,N},
    bsig2::AbstractArray,
    dimension::Integer,
    max_nodes::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    planar::Bool=false,
    scalar_term::Bool=false,
    n_jobs::Integer=1,
) where {T<:Union{Float32,Float64},N}
    y = SigLib.branched_sig_combine(
        bsig1, bsig2, dimension, max_nodes;
        time_aug=time_aug,
        lead_lag=lead_lag,
        planar=planar,
        scalar_term=scalar_term,
        n_jobs=n_jobs,
    )
    project_bsig1 = ProjectTo(bsig1)
    project_bsig2 = ProjectTo(bsig2)

    function branched_sig_combine_pullback(ybar)
        ybar = unthunk(ybar)
        if _is_zero_tangent(ybar)
            return NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent(), NoTangent()
        end

        bsig1bar, bsig2bar = SigLib._branched_sig_combine_backprop(
            bsig1, bsig2, ybar, dimension, max_nodes;
            time_aug=time_aug,
            lead_lag=lead_lag,
            planar=planar,
            scalar_term=scalar_term,
            n_jobs=n_jobs,
        )
        return (
            NoTangent(),
            project_bsig1(bsig1bar),
            project_bsig2(bsig2bar),
            NoTangent(),
            NoTangent(),
        )
    end

    return y, branched_sig_combine_pullback
end

end
