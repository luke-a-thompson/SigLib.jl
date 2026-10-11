"""
    sig(path, degree; time_aug=false, lead_lag=false, end_time=1.0,
        horner=true, scalar_term=false, correction=nothing, n_jobs=1)

Compute the truncated signature of one path or a batch of paths.

The public shape convention follows pySigLib:

- a single path has shape `(length, dimension)`
- a batch has shape `(batch..., length, dimension)`
- the result has shape `(signature_length,)` or `(batch..., signature_length)`
"""
function sig(
    path::AbstractArray{T,N},
    degree::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    end_time::Real=1.0,
    horner::Bool=true,
    scalar_term::Bool=false,
    correction=nothing,
    n_jobs::Integer=1,
) where {T<:Real,N}
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    _as_cint_n_jobs(n_jobs)
    _validate_standard_signature_options(correction)

    path_c, batch_shape, batch_size, dimension, len = _path_buffer(path)
    out = _sig_from_buffer(
        path_c, batch_size, dimension, len, degree;
        time_aug=time_aug, lead_lag=lead_lag, end_time=end_time,
        horner=horner, scalar_term=scalar_term, n_jobs=n_jobs,
    )
    return _format_signature_output(out, batch_shape)
end

function _sig_from_buffer(
    path_c::AbstractArray{T,3}, batch_size, dimension, len, degree;
    time_aug=false, lead_lag=false, end_time=1.0, horner=true, scalar_term=false, n_jobs=1,
) where {T<:Union{Float32,Float64}}
    siglen = sig_length(
        dimension, degree;
        time_aug=time_aug, lead_lag=lead_lag, scalar_term=scalar_term,
    )
    out = _allocate(path_c, siglen, batch_size)

    if batch_size != 0 && siglen != 0
        _signature_c!(
            out, path_c, batch_size, dimension, len, degree,
            time_aug, lead_lag, horner, scalar_term, n_jobs, end_time,
        )
    end

    return out
end

function _sig_backprop(
    path::AbstractArray{T,N},
    sig_value::AbstractArray,
    sig_derivs::AbstractArray,
    degree::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    end_time::Real=1.0,
    scalar_term::Bool=false,
    correction=nothing,
    n_jobs::Integer=1,
) where {T<:Real,N}
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    _as_cint_n_jobs(n_jobs)
    _validate_standard_signature_options(correction)

    path_c, batch_shape, batch_size, dimension, len = _path_buffer(path)
    S = eltype(path_c)
    siglen = sig_length(
        dimension, degree;
        time_aug=time_aug, lead_lag=lead_lag, scalar_term=scalar_term,
    )
    sig_c = _signature_buffer(sig_value, S, siglen, batch_shape, batch_size, "sig_value", path_c)
    sig_derivs_c = _signature_buffer(sig_derivs, S, siglen, batch_shape, batch_size, "sig_derivs", path_c)

    out_c = _allocate(path_c, dimension, len, batch_size; zero=true)
    if batch_size != 0 && siglen != 0
        _sig_backprop_c!(
            out_c, path_c, sig_derivs_c, sig_c, batch_size, dimension, len, degree,
            time_aug, lead_lag, scalar_term, n_jobs, end_time,
        )
    end

    return _format_path_output(out_c, batch_shape)
end
