"""
    branched_sig(path, max_nodes; time_aug=false, lead_lag=false, end_time=1.0,
        planar=false, scalar_term=false, correction=nothing, n_jobs=1)

Compute the truncated branched signature of one path or a batch of paths.

The public shape convention follows pySigLib:

- a single path has shape `(length, dimension)`
- a batch has shape `(batch..., length, dimension)`
- the result has shape `(branched_signature_length,)` or
  `(batch..., branched_signature_length)`

`correction` uses the tensor-level layouts described in [`sig`](@ref), with
`degree = max_nodes`. Each word supplies a chain tree labeled from root to leaf
in the local Hopf exponential. Corrections are constant during path differentiation.
"""
function branched_sig(
    path::AbstractArray{T,N},
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
    out = _branched_sig_from_buffer(path_c, batch_size, dimension, len, max_nodes;
        time_aug=time_aug, lead_lag=lead_lag, end_time=end_time, planar=planar,
        scalar_term=scalar_term, correction=correction_c, n_jobs=n_jobs)
    return _format_signature_output(out, batch_shape)
end

function _branched_sig_from_buffer(
    path_c::AbstractArray{T,3}, batch_size, dimension, len, max_nodes;
    time_aug=false, lead_lag=false, end_time=1.0, planar=false, scalar_term=false,
    correction::_CorrectionBuffer=_CorrectionBuffer(), n_jobs=1,
) where {T<:Union{Float32,Float64}}
    siglen = branched_sig_length(
        dimension, max_nodes;
        time_aug=time_aug, lead_lag=lead_lag, planar=planar, scalar_term=scalar_term,
    )
    out = _allocate(path_c, siglen, batch_size)

    if batch_size != 0 && siglen != 0
        lock(_cache_lock) do
            _prepare_branched_sig_for(path_c,
                dimension, max_nodes;
                time_aug=time_aug, lead_lag=lead_lag, planar=planar,
            )
            _branched_sig_c!(
                out, path_c, batch_size, dimension, len, max_nodes,
                time_aug, lead_lag, planar, scalar_term, n_jobs, end_time, correction,
            )
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
