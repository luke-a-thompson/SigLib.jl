_validate_branched_log_sig_degree(::AbstractArray, degree, method) = nothing

function _validate_branched_log_sig_method(method::BranchedLogSigMethod.T, planar::Bool; conversion::Bool=false)
    method != BranchedLogSigMethod.Expanded && !planar && throw(ArgumentError("branched log signature methods other than BranchedLogSigMethod.Expanded require planar=true"))
    conversion && method == BranchedLogSigMethod.DirectBCH && throw(ArgumentError("BranchedLogSigMethod.DirectBCH requires branched_log_sig(path, degree)"))
    return nothing
end

_branched_log_sig_dimension(dimension, time_aug, lead_lag) =
    dimension == 0 ? UInt64(time_aug) : _effective_dimension(dimension, time_aug, lead_lag)

function _branched_log_sig_input(bsig::AbstractArray{T,N}, dimension, degree, time_aug, lead_lag, planar, method) where {T<:Real,N}
    N >= 1 || throw(ArgumentError("bsig must have at least rank 1"))
    d = _branched_log_sig_dimension(dimension, time_aug, lead_lag)
    _as_uint64(degree, "degree")
    len = d == 0 ? 0 : branched_sig_length(d, degree; planar=planar)
    scalar_term = size(bsig, 1) == len + 1
    siglen = len + scalar_term
    S = _storage_eltype(T, :branched_sig_to_log_sig)
    bsig_c, batch_shape, batch_size = _signature_like_buffer(bsig, S, siglen, "bsig")
    outlen = method == BranchedLogSigMethod.Expanded ? siglen : branched_log_sig_length(d, degree; planar=true)
    return bsig_c, batch_shape, batch_size, d, scalar_term, outlen
end

"""
    branched_sig_to_log_sig(bsig, dimension, degree; time_aug=false, lead_lag=false, planar=false, method=planar ? BranchedLogSigMethod.MkwWords : BranchedLogSigMethod.Expanded, n_jobs=1)

Convert a branched signature to its Hopf logarithm using a [`BranchedLogSigMethod`](@ref). `Expanded` preserves the input's scalar-term format; `MkwWords` and `MkwBasis` return compressed scalar-free outputs. The default is `MkwWords` for planar signatures and `Expanded` otherwise. `DirectBCH` requires a path.
"""
function branched_sig_to_log_sig(
    bsig::AbstractArray{T,N}, dimension::Integer, degree::Integer;
    time_aug::Bool=false, lead_lag::Bool=false, planar::Bool=false,
    method::BranchedLogSigMethod.T=planar ? BranchedLogSigMethod.MkwWords : BranchedLogSigMethod.Expanded, n_jobs::Integer=1,
) where {T<:Real,N}
    _validate_branched_log_sig_method(method, planar; conversion=true)
    _as_cint_n_jobs(n_jobs)
    bsig_c, batch_shape, batch_size, d, scalar_term, outlen =
        _branched_log_sig_input(bsig, dimension, degree, time_aug, lead_lag, planar, method)
    out = _allocate(bsig_c, outlen, batch_size)
    _branched_sig_to_log_sig_from_buffer!(out, bsig_c, batch_size, d, degree, method, planar, scalar_term, n_jobs)
    return _format_signature_output(out, batch_shape)
end

function _branched_sig_to_log_sig_from_buffer!(out, bsig_c, batch_size, d, degree, method, planar, scalar_term, n_jobs)
    isempty(out) && return out
    _with_array(bsig_c) do
        lock(_cache_lock) do
            _cache_call(Val(_backend(bsig_c)), Val(:prepare_branched_log_sig), UInt64(d), UInt64(degree), method, planar, false)
            _branched_sig_to_log_sig_c!(out, bsig_c, batch_size, d, degree, planar, scalar_term, n_jobs, method)
        end
    end
    return out
end

"""
    branched_log_sig(path, degree; time_aug=false, lead_lag=false, end_time=1.0, planar=false, scalar_term=false, method=planar ? BranchedLogSigMethod.MkwWords : BranchedLogSigMethod.Expanded, correction=nothing, n_jobs=1)

Compute branched log signatures using a [`BranchedLogSigMethod`](@ref), defaulting to `MkwWords` for planar paths and `Expanded` otherwise. All except `Expanded` require `planar=true` and ignore `scalar_term`; `DirectBCH` computes MKW basis coordinates directly through BCH. `correction` accepts the layouts described in [`branched_sig`](@ref). Use `Expanded` to retain the full Hopf logarithm for non-Lie corrections. `DirectBCH` accepts only absent or empty corrections.
"""
branched_log_sig(path::AbstractArray, degree::Integer; kwargs...) =
    _branched_log_sig(nothing, nothing, path, degree; kwargs...)

"""
    branched_log_sig!(out, path, degree, scratch=nothing; kwargs...)

Write [`branched_log_sig`](@ref) into `out` and return it, using the same keywords.
Supply `scratch` with the shape of `branched_sig(path, degree; ...)`, including
its scalar-term format. `DirectBCH` needs no scratch. Both buffers follow
[`sig!`](@ref)'s requirements and must not alias each other. Direct BCH path
transforms may allocate; untransformed warmed rank-3 CPU calls reuse Julia storage.
"""
branched_log_sig!(out::AbstractArray, path::AbstractArray, degree::Integer, scratch=nothing; kwargs...) =
    _branched_log_sig(out, scratch, path, degree; kwargs...)

function _branched_log_sig(
    destination, scratch, path::AbstractArray{T,N}, degree::Integer;
    time_aug::Bool=false, lead_lag::Bool=false, end_time::Real=1.0,
    planar::Bool=false, scalar_term::Bool=false, method::BranchedLogSigMethod.T=planar ? BranchedLogSigMethod.MkwWords : BranchedLogSigMethod.Expanded,
    correction=nothing, n_jobs::Integer=1,
) where {T<:Real,N}
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    _validate_branched_log_sig_method(method, planar)
    _validate_branched_log_sig_degree(path, degree, method)
    _as_cint_n_jobs(n_jobs)
    path_c, batch_shape, batch_size, dimension, len = _path_buffer(path, :branched_log_sig)
    correction_c = _correction_buffer(correction, path_c, batch_shape, batch_size, dimension, len, degree, lead_lag)
    direct = method == BranchedLogSigMethod.DirectBCH
    direct && _validate_bch_correction(correction_c, :branched_log_sig)
    destination !== nothing && !direct && scratch === nothing &&
        throw(ArgumentError("supply a branched-signature scratch buffer for this method"))
    d = _effective_dimension(dimension, time_aug, lead_lag)
    siglen = direct ? 0 : branched_sig_length(d, degree; planar, scalar_term)
    outlen = method == BranchedLogSigMethod.Expanded ? siglen : branched_log_sig_length(d, degree; planar=true)
    out = _output_buffer(destination, path_c, outlen, batch_shape, batch_size, "out", (path, correction))
    bsig_c = direct ? nothing : _output_buffer(scratch, path_c, siglen, batch_shape, batch_size, "scratch", (out, path, correction))
    _branched_log_sig_from_buffers!(out, bsig_c, path_c, batch_size, dimension, len, degree, method,
        correction_c.data, correction_c.length, correction_c.batch_stride, correction_c.segment_stride;
        time_aug, lead_lag, end_time, planar, scalar_term, n_jobs)
    return destination === nothing ? _format_signature_output(out, batch_shape) : destination
end

function _branched_log_sig_from_buffers!(out, bsig_c, path_c, batch_size, dimension, len, degree, method,
    correction_data, correction_length, correction_batch_stride, correction_segment_stride;
    time_aug=false, lead_lag=false, end_time=1.0, planar=false, scalar_term=false, n_jobs=1,
)
    isempty(out) && return out
    _with_array(path_c) do
        lock(_cache_lock) do
            if method == BranchedLogSigMethod.DirectBCH
                transformed_path, d, transformed_len = _transformed_path_for_log_sig(
                    path_c, batch_size, dimension, len, time_aug, lead_lag, end_time, n_jobs)
                _cache_call(Val(_backend(path_c)), Val(:prepare_branched_log_sig), UInt64(d), UInt64(degree), method, planar, false)
                _branched_log_sig_from_path_c!(out, transformed_path, batch_size, transformed_len, d, degree, n_jobs)
            else
                _branched_sig_from_buffer!(bsig_c, path_c, batch_size, dimension, len, degree,
                    correction_data, correction_length, correction_batch_stride, correction_segment_stride;
                    time_aug, lead_lag, end_time, planar, scalar_term, n_jobs)
                _branched_sig_to_log_sig_from_buffer!(out, bsig_c, batch_size,
                    _effective_dimension(dimension, time_aug, lead_lag), degree, method, planar, scalar_term, n_jobs)
            end
        end
    end
    return out
end

function _branched_sig_to_log_sig_backprop(
    bsig::AbstractArray{T,N}, derivs::AbstractArray, dimension::Integer, degree::Integer;
    time_aug::Bool=false, lead_lag::Bool=false, planar::Bool=false,
    method::BranchedLogSigMethod.T=planar ? BranchedLogSigMethod.MkwWords : BranchedLogSigMethod.Expanded, n_jobs::Integer=1,
) where {T<:Real,N}
    _validate_branched_log_sig_method(method, planar; conversion=true)
    _as_cint_n_jobs(n_jobs)
    bsig_c, batch_shape, batch_size, d, scalar_term, outlen =
        _branched_log_sig_input(bsig, dimension, degree, time_aug, lead_lag, planar, method)
    derivs_c = _signature_buffer(derivs, eltype(bsig_c), outlen, batch_shape, batch_size, "derivs", bsig_c)
    out = _allocate(bsig_c, size(bsig_c)...; zero=true)
    if batch_size != 0 && outlen != 0
        lock(_cache_lock) do
            _prepare_branched_log_sig_for(bsig_c, d, degree, method; planar=planar)
            _branched_sig_to_log_sig_backprop_c!(out, bsig_c, derivs_c, batch_size, d, degree, planar, scalar_term, n_jobs, method)
        end
    end
    return _format_signature_output(out, batch_shape)
end

function _branched_log_sig_from_path_backprop(
    path::AbstractArray, derivs::AbstractArray, degree::Integer;
    time_aug::Bool=false, lead_lag::Bool=false, end_time::Real=1.0, n_jobs::Integer=1,
)
    _validate_branched_log_sig_method(BranchedLogSigMethod.DirectBCH, true)
    _validate_branched_log_sig_degree(path, degree, BranchedLogSigMethod.DirectBCH)
    _as_cint_n_jobs(n_jobs)
    path_c, batch_shape, batch_size, dimension, len = _path_buffer(path, :branched_log_sig)
    d = _effective_dimension(dimension, time_aug, lead_lag)
    outlen = branched_log_sig_length(d, degree; planar=true)
    derivs_c = _signature_buffer(derivs, eltype(path_c), outlen, batch_shape, batch_size, "derivs", path_c)
    out = _allocate(path_c, dimension, len, batch_size; zero=true)
    if batch_size != 0 && outlen != 0
        lock(_cache_lock) do
            transformed_path, transformed_dimension, transformed_len = _transformed_path_for_log_sig(
                path_c, batch_size, dimension, len, time_aug, lead_lag, end_time, n_jobs)
            _prepare_branched_log_sig_for(path_c, d, degree, BranchedLogSigMethod.DirectBCH; planar=true)
            transformed_out = (time_aug || lead_lag) ? _allocate(path_c, transformed_dimension, transformed_len, batch_size; zero=true) : out
            _branched_log_sig_from_path_backprop_c!(transformed_out, derivs_c, transformed_path,
                batch_size, transformed_len, transformed_dimension, degree, n_jobs)
            if time_aug || lead_lag
                _transform_path_backprop_c!(out, transformed_out, batch_size, dimension, len,
                    time_aug, lead_lag, end_time, n_jobs)
            end
        end
    end
    return _format_path_output(out, batch_shape)
end
