"""
    branched_sig_to_log_sig(bsig, dimension, max_nodes; time_aug=false,
        lead_lag=false, planar=false, scalar_term=false, n_jobs=1)

Compute the truncated Hopf logarithm of a branched signature.
"""
function branched_sig_to_log_sig(
    bsig::AbstractArray{T,N},
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
    S = _storage_eltype(T, :branched_sig_to_log_sig)
    siglen = branched_sig_length(
        dimension, max_nodes;
        time_aug=time_aug, lead_lag=lead_lag, planar=planar, scalar_term=scalar_term,
    )
    bsig_c, batch_shape, batch_size = _signature_like_buffer(bsig, S, siglen, "bsig")
    out = Matrix{S}(undef, siglen, batch_size)

    if batch_size != 0 && siglen != 0
        lock(_cache_lock) do
            prepare_branched_sig(
                dimension, max_nodes;
                time_aug=time_aug, lead_lag=lead_lag, planar=planar,
            )
            effective_dimension = _effective_dimension(dimension, time_aug, lead_lag)
            _branched_sig_to_log_sig_c!(
                out, bsig_c, batch_size, effective_dimension, max_nodes,
                planar, scalar_term, n_jobs,
            )
        end
    end

    return _format_signature_output(out, batch_shape)
end

"""
    branched_log_sig(path, max_nodes; kwargs...)

Compute `branched_sig(path, max_nodes; kwargs...)` and convert it to its
truncated Hopf logarithm.
"""
function branched_log_sig(
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
    bsig = branched_sig(
        path, max_nodes;
        time_aug=time_aug,
        lead_lag=lead_lag,
        end_time=end_time,
        planar=planar,
        scalar_term=scalar_term,
        correction=correction,
        n_jobs=n_jobs,
    )
    dimension = size(path, N)
    return branched_sig_to_log_sig(
        bsig, dimension, max_nodes;
        time_aug=time_aug,
        lead_lag=lead_lag,
        planar=planar,
        scalar_term=scalar_term,
        n_jobs=n_jobs,
    )
end

function _branched_sig_to_log_sig_backprop(
    bsig::AbstractArray{T,N},
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
    S = _storage_eltype(T, :_branched_sig_to_log_sig_backprop)
    siglen = branched_sig_length(
        dimension, max_nodes;
        time_aug=time_aug, lead_lag=lead_lag, planar=planar, scalar_term=scalar_term,
    )
    bsig_c, batch_shape, batch_size = _signature_like_buffer(bsig, S, siglen, "bsig")
    derivs_c = _signature_buffer(derivs, S, siglen, batch_shape, batch_size, "derivs")
    out = Matrix{S}(undef, siglen, batch_size)

    if batch_size != 0 && siglen != 0
        lock(_cache_lock) do
            prepare_branched_sig(
                dimension, max_nodes;
                time_aug=time_aug, lead_lag=lead_lag, planar=planar,
            )
            effective_dimension = _effective_dimension(dimension, time_aug, lead_lag)
            _branched_sig_to_log_sig_backprop_c!(
                out, bsig_c, derivs_c, batch_size, effective_dimension, max_nodes,
                planar, scalar_term, n_jobs,
            )
        end
    end

    return _format_signature_output(out, batch_shape)
end
