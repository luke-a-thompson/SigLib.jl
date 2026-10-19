"""
    branched_sig(path, max_nodes; time_aug=false, lead_lag=false, end_time=1.0,
        planar=false, scalar_term=false, correction=nothing, n_jobs=1)

Compute the truncated branched signature of one path or a batch of paths.

The public shape convention uses channels first and batch dimensions last:

- a single path has shape `(dimension, length)`
- a batch has shape `(dimension, length, batch...)`
- the result has shape `(branched_signature_length,)` or
  `(branched_signature_length, batch...)`

`correction` uses the tensor-level layouts described in [`sig`](@ref), with
`degree = max_nodes`. Each word supplies a chain tree labeled from root to leaf
in the local Hopf exponential. Corrections are constant during path differentiation.
"""
branched_sig(path::AbstractArray, degree::Integer; kwargs...) = _branched_sig(nothing, path, degree; kwargs...)

"""
    branched_sig!(out, path, max_nodes; kwargs...)

Write [`branched_sig`](@ref) into `out` and return it, using the same keywords.
Buffer requirements match [`sig!`](@ref). Native caches prepare automatically,
including after `clear_log_sig_cache()`.
"""
branched_sig!(out::AbstractArray, path::AbstractArray, degree::Integer; kwargs...) =
    _branched_sig(out, path, degree; kwargs...)

function _branched_sig(
    destination, path::AbstractArray{T,N},
    max_nodes::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    end_time::Real=1.0,
    planar::Bool=false,
    scalar_term::Bool=false,
    correction=nothing,
    n_jobs::Integer=1,
) where {T<:Real,N}
    max_nodes >= 0 || throw(ArgumentError("max_nodes must be nonnegative"))
    _as_cint_n_jobs(n_jobs)
    path_c, batch_shape, batch_size, dimension, len = _path_buffer(path, :branched_sig)
    correction_c = _correction_buffer(correction, path_c, batch_shape, batch_size, dimension, len, max_nodes, lead_lag)
    siglen = branched_sig_length(dimension, max_nodes; time_aug, lead_lag, planar, scalar_term)
    out = _output_buffer(destination, path_c, siglen, batch_shape, batch_size, "out", (path, correction))
    _branched_sig_from_buffer!(out, path_c, batch_size, dimension, len, max_nodes,
        correction_c.data, correction_c.length, correction_c.batch_stride, correction_c.segment_stride;
        time_aug, lead_lag, end_time, planar, scalar_term, n_jobs)
    return destination === nothing ? _format_signature_output(out, batch_shape) : destination
end

# Pass correction fields across the lock callback to avoid boxing metadata.
function _branched_sig_from_buffer!(out, path_c, batch_size, dimension, len, max_nodes,
    correction_data, correction_length, correction_batch_stride, correction_segment_stride;
    time_aug=false, lead_lag=false, end_time=1.0, planar=false, scalar_term=false, n_jobs=1,
)
    isempty(out) && return out
    _with_array(path_c) do
        lock(_cache_lock) do
            _cache_call(Val(_backend(path_c)), Val(:prepare_branched_sig),
                _effective_dimension(dimension, time_aug, lead_lag), UInt64(max_nodes), planar, false)
            _branched_sig_c!(out, path_c, batch_size, dimension, len, max_nodes,
                time_aug, lead_lag, planar, scalar_term, n_jobs, end_time,
                _CorrectionBuffer(correction_data, correction_length, correction_batch_stride, correction_segment_stride))
        end
    end
    return out
end

"""
    branched_sig_combine(bsig1, bsig2, dimension, max_nodes; time_aug=false,
        lead_lag=false, planar=false, scalar_term=false, n_jobs=1)

Combine pairs of branched signatures with the same dimension, truncation, basis,
and scalar-term convention.
"""
function branched_sig_combine(
    bsig1::AbstractArray{T,N},
    bsig2::AbstractArray,
    dimension::Integer,
    max_nodes::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    planar::Bool=false,
    scalar_term::Bool=false,
    n_jobs::Integer=1,
) where {T<:Real,N}
    max_nodes >= 0 || throw(ArgumentError("max_nodes must be nonnegative"))
    _as_cint_n_jobs(n_jobs)
    S = _storage_eltype(T, :branched_sig_combine)
    siglen = branched_sig_length(
        dimension, max_nodes;
        time_aug=time_aug, lead_lag=lead_lag, planar=planar, scalar_term=scalar_term,
    )
    bsig1_c, batch_shape, batch_size = _signature_like_buffer(bsig1, S, siglen, "bsig1")
    bsig2_c = _signature_buffer(bsig2, S, siglen, batch_shape, batch_size, "bsig2", bsig1_c)
    out = _allocate(bsig1_c, siglen, batch_size)

    if batch_size != 0 && siglen != 0
        lock(_cache_lock) do
            _prepare_branched_sig_for(bsig1_c,
                dimension, max_nodes;
                time_aug=time_aug, lead_lag=lead_lag, planar=planar,
            )
            effective_dimension = _effective_dimension(dimension, time_aug, lead_lag)
            _branched_sig_combine_c!(
                out, bsig1_c, bsig2_c, batch_size, effective_dimension, max_nodes,
                planar, scalar_term, n_jobs,
            )
        end
    end

    return _format_signature_output(out, batch_shape)
end

function _branched_sig_backprop(
    path::AbstractArray{T,N},
    bsig_value::AbstractArray,
    bsig_derivs::AbstractArray,
    max_nodes::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    end_time::Real=1.0,
    planar::Bool=false,
    scalar_term::Bool=false,
    correction=nothing,
    n_jobs::Integer=1,
) where {T<:Real,N}
    max_nodes >= 0 || throw(ArgumentError("max_nodes must be nonnegative"))
    _as_cint_n_jobs(n_jobs)
    path_c, batch_shape, batch_size, dimension, len = _path_buffer(path, :branched_sig)
    correction_c = _correction_buffer(correction, path_c, batch_shape, batch_size, dimension, len, max_nodes, lead_lag)
    S = eltype(path_c)
    siglen = branched_sig_length(
        dimension, max_nodes;
        time_aug=time_aug, lead_lag=lead_lag, planar=planar, scalar_term=scalar_term,
    )
    bsig_c = _signature_buffer(bsig_value, S, siglen, batch_shape, batch_size, "bsig_value", path_c)
    bsig_derivs_c = _signature_buffer(bsig_derivs, S, siglen, batch_shape, batch_size, "bsig_derivs", path_c)

    out_c = _allocate(path_c, dimension, len, batch_size; zero=true)
    if batch_size != 0 && siglen != 0
        lock(_cache_lock) do
            _prepare_branched_sig_for(path_c,
                dimension, max_nodes;
                time_aug=time_aug, lead_lag=lead_lag, planar=planar,
            )
            _branched_sig_backprop_c!(
                out_c, path_c, bsig_derivs_c, bsig_c, batch_size, dimension, len, max_nodes,
                time_aug, lead_lag, planar, scalar_term, n_jobs, end_time, correction_c,
            )
        end
    end

    return _format_path_output(out_c, batch_shape)
end

function _branched_sig_combine_backprop(
    bsig1::AbstractArray{T,N},
    bsig2::AbstractArray,
    derivs::AbstractArray,
    dimension::Integer,
    max_nodes::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    planar::Bool=false,
    scalar_term::Bool=false,
    n_jobs::Integer=1,
) where {T<:Real,N}
    max_nodes >= 0 || throw(ArgumentError("max_nodes must be nonnegative"))
    _as_cint_n_jobs(n_jobs)
    S = _storage_eltype(T, :_branched_sig_combine_backprop)
    siglen = branched_sig_length(
        dimension, max_nodes;
        time_aug=time_aug, lead_lag=lead_lag, planar=planar, scalar_term=scalar_term,
    )
    bsig1_c, batch_shape, batch_size = _signature_like_buffer(bsig1, S, siglen, "bsig1")
    bsig2_c = _signature_buffer(bsig2, S, siglen, batch_shape, batch_size, "bsig2", bsig1_c)
    derivs_c = _signature_buffer(derivs, S, siglen, batch_shape, batch_size, "derivs", bsig1_c)
    out1 = _allocate(bsig1_c, siglen, batch_size)
    out2 = _allocate(bsig1_c, siglen, batch_size)

    if batch_size != 0 && siglen != 0
        lock(_cache_lock) do
            _prepare_branched_sig_for(bsig1_c,
                dimension, max_nodes;
                time_aug=time_aug, lead_lag=lead_lag, planar=planar,
            )
            effective_dimension = _effective_dimension(dimension, time_aug, lead_lag)
            _branched_sig_combine_backprop_c!(
                out1, out2, bsig1_c, bsig2_c, derivs_c, batch_size,
                effective_dimension, max_nodes, planar, scalar_term, n_jobs,
            )
        end
    end

    return _format_signature_output(out1, batch_shape),
        _format_signature_output(out2, batch_shape)
end
