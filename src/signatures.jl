"""
    sig(path, degree; time_aug=false, lead_lag=false, end_time=1.0,
        horner=true, scalar_term=false, correction=nothing, n_jobs=1)

Compute the truncated signature of one path or a batch of paths.

The public shape convention uses channels first and batch dimensions last:

- a single path has shape `(dimension, length)`
- a batch has shape `(dimension, length, batch...)`
- the result has shape `(signature_length,)` or `(signature_length, batch...)`

`correction` adds tensor levels 2 through `m <= degree` to each local path
increment before exponentiation. It accepts a constant vector `(C,)`, shared
per-segment rows `(C, length - 1)`, or batch-specific rows
`(C, length - 1, batch...)`, where `C = d^2 + ... + d^m` and `d` is the original
path dimension. Levels are concatenated in lexicographic word order (last index
varies fastest). Missing higher levels are zero; `nothing` or an empty array
means no correction. The correction must use the path's eltype and backend.
Time augmentation adds no correction to the time channel. Non-empty corrections
cannot be combined with `lead_lag=true`. Path gradients treat corrections as constants.
"""
sig(path::AbstractArray, degree::Integer; kwargs...) = _sig(nothing, path, degree; kwargs...)

"""
    sig!(out, path, degree; kwargs...)

Write [`sig`](@ref) into `out` and return it. Accepts the same keywords as `sig`.
`out` must be a dense array with the result's shape and the path's eltype and
backend, without aliasing the path or correction. Warmed rank-3 CPU calls reuse
all Julia storage; input views may be materialized.
"""
sig!(out::AbstractArray, path::AbstractArray, degree::Integer; kwargs...) = _sig(out, path, degree; kwargs...)

function _sig(
    destination, path::AbstractArray{T,N},
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
    path_c, batch_shape, batch_size, dimension, len = _path_buffer(path)
    correction_c = _correction_buffer(correction, path_c, batch_shape, batch_size, dimension, len, degree, lead_lag)
    siglen = sig_length(dimension, degree; time_aug, lead_lag, scalar_term)
    out = _output_buffer(destination, path_c, siglen, batch_shape, batch_size, "out", (path, correction))
    isempty(out) || _signature_c!(out, path_c, batch_size, dimension, len, degree,
        time_aug, lead_lag, horner, scalar_term, n_jobs, end_time, correction_c)
    return destination === nothing ? _format_signature_output(out, batch_shape) : destination
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
    path_c, batch_shape, batch_size, dimension, len = _path_buffer(path)
    correction_c = _correction_buffer(correction, path_c, batch_shape, batch_size, dimension, len, degree, lead_lag)
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
            time_aug, lead_lag, scalar_term, n_jobs, end_time, correction_c,
        )
    end

    return _format_path_output(out_c, batch_shape)
end
