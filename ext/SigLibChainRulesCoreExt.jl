module SigLibChainRulesCoreExt

using SigLib
import ChainRulesCore: NoTangent, ProjectTo, ZeroTangent, rrule, unthunk

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
        if ybar isa NoTangent || ybar isa ZeroTangent
            return NoTangent(), ZeroTangent(), NoTangent()
        end

        pathbar = SigLib.sig_backprop(
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

end
