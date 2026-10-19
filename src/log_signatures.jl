function _log_sig_output_length(
    dimension::Integer,
    degree::Integer,
    method::LogSigMethod.T,
    scalar_term::Bool;
    time_aug::Bool=false,
    lead_lag::Bool=false,
)
    if method == LogSigMethod.Expanded
        return sig_length(
            dimension, degree;
            time_aug=time_aug, lead_lag=lead_lag, scalar_term=scalar_term,
        )
    end

    return log_sig_length(dimension, degree; time_aug=time_aug, lead_lag=lead_lag)
end

function _prepare_log_sig_if_needed(
    like::AbstractArray,
    dimension::Integer,
    degree::Integer,
    method::LogSigMethod.T;
    time_aug::Bool=false,
    lead_lag::Bool=false,
)
    if method == LogSigMethod.LyndonWords || method == LogSigMethod.LyndonBasis
        _prepare_log_sig_for(like, dimension, degree, method; time_aug=time_aug, lead_lag=lead_lag)
    end
    return nothing
end

function _signature_input_buffer(
    x::AbstractArray{T,N},
    ::Type{S},
    dimension::Integer,
    degree::Integer,
    name::AbstractString;
    time_aug::Bool=false,
    lead_lag::Bool=false,
) where {T<:Real,N,S<:Union{Float32,Float64}}
    siglen = sig_length(dimension, degree; time_aug=time_aug, lead_lag=lead_lag)
    siglen_scalar = siglen + 1
    N >= 1 || throw(ArgumentError("$name must have at least rank 1"))
    width = size(x, 1)
    if width == siglen
        buf, batch_shape, batch_size = _signature_like_buffer(x, S, siglen, name)
        return buf, batch_shape, batch_size, false
    elseif width == siglen_scalar
        buf, batch_shape, batch_size = _signature_like_buffer(x, S, siglen_scalar, name)
        return buf, batch_shape, batch_size, true
    end

    throw(ArgumentError(
        "$name must have first dimension $siglen or $siglen_scalar, got $width",
    ))
end

function _log_sig_buffer(
    x::AbstractArray{T,N},
    ::Type{S},
    dimension::Integer,
    degree::Integer,
    method::LogSigMethod.T,
    name::AbstractString;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    scalar_term::Bool=false,
) where {T<:Real,N,S<:Union{Float32,Float64}}
    if method == LogSigMethod.Expanded
        return _signature_input_buffer(
            x, S, dimension, degree, name;
            time_aug=time_aug, lead_lag=lead_lag,
        )
    end

    len = log_sig_length(dimension, degree; time_aug=time_aug, lead_lag=lead_lag)
    buf, batch_shape, batch_size = _signature_like_buffer(x, S, len, name)
    return buf, batch_shape, batch_size, scalar_term
end

function _transformed_path_for_log_sig(
    path_c::AbstractArray{T,3},
    batch_size::Integer,
    dimension::Integer,
    len::Integer,
    time_aug::Bool,
    lead_lag::Bool,
    end_time::Real,
    n_jobs::Integer,
) where {T<:Union{Float32,Float64}}
    transformed_dimension = Int(_effective_dimension(dimension, time_aug, lead_lag))
    transformed_len = lead_lag ? 2len - 1 : len
    if !time_aug && !lead_lag
        return path_c, dimension, len
    end

    out = _allocate(path_c, transformed_dimension, transformed_len, batch_size)
    _transform_path_c!(
        out, path_c, batch_size, dimension, len,
        time_aug, lead_lag, end_time, n_jobs,
    )
    return out, transformed_dimension, transformed_len
end

"""
    sig_to_log_sig(sig, dimension, degree; time_aug=false, lead_lag=false,
        method=LogSigMethod.LyndonWords, n_jobs=1)

Compute log signatures from signatures using a [`LogSigMethod`](@ref).
`LogSigMethod.DirectBCH` requires a path and is unsupported here.
"""
function sig_to_log_sig(
    sig_value::AbstractArray{T,N},
    dimension::Integer,
    degree::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    method::LogSigMethod.T=LogSigMethod.LyndonWords,
    n_jobs::Integer=1,
) where {T<:Real,N}
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    _validate_log_sig_conversion_method(method)
    _as_cint_n_jobs(n_jobs)
    S = _storage_eltype(T, :sig_to_log_sig)
    sig_c, batch_shape, batch_size, scalar_term = _signature_input_buffer(
        sig_value, S, dimension, degree, "sig";
        time_aug=time_aug, lead_lag=lead_lag,
    )
    outlen = _log_sig_output_length(
        dimension, degree, method, scalar_term;
        time_aug=time_aug, lead_lag=lead_lag,
    )
    out = _allocate(sig_c, outlen, batch_size)

    if batch_size != 0 && outlen != 0
        lock(_cache_lock) do
            _prepare_log_sig_if_needed(
                sig_c,
                dimension, degree, method;
                time_aug=time_aug, lead_lag=lead_lag,
            )
            _sig_to_log_sig_c!(
                out, sig_c, batch_size, dimension, degree,
                time_aug, lead_lag, method, scalar_term, n_jobs,
            )
        end
    end

    return _format_signature_output(out, batch_shape)
end

"""
    logsig_to_sig(log_sig, dimension, degree; time_aug=false, lead_lag=false,
        method=LogSigMethod.LyndonWords, scalar_term=false, n_jobs=1)

Compute signatures from log signatures via truncated tensor exponential.
Choose a [`LogSigMethod`](@ref) matching the input coordinates; `DirectBCH` is unsupported.
"""
function logsig_to_sig(
    log_sig_value::AbstractArray{T,N},
    dimension::Integer,
    degree::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    method::LogSigMethod.T=LogSigMethod.LyndonWords,
    scalar_term::Bool=false,
    n_jobs::Integer=1,
) where {T<:Real,N}
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    _validate_log_sig_conversion_method(method)
    _as_cint_n_jobs(n_jobs)
    S = _storage_eltype(T, :logsig_to_sig)
    log_sig_c, batch_shape, batch_size, inferred_scalar_term = _log_sig_buffer(
        log_sig_value, S, dimension, degree, method, "log_sig";
        time_aug=time_aug, lead_lag=lead_lag, scalar_term=scalar_term,
    )
    output_scalar_term = method == LogSigMethod.Expanded ? inferred_scalar_term : scalar_term
    outlen = sig_length(
        dimension, degree;
        time_aug=time_aug, lead_lag=lead_lag, scalar_term=output_scalar_term,
    )
    out = _allocate(log_sig_c, outlen, batch_size)

    if batch_size != 0 && outlen != 0
        lock(_cache_lock) do
            _prepare_log_sig_if_needed(
                log_sig_c,
                dimension, degree, method == LogSigMethod.LyndonWords ? LogSigMethod.LyndonBasis : method;
                time_aug=time_aug, lead_lag=lead_lag,
            )
            _logsig_to_sig_c!(
                out, log_sig_c, batch_size, dimension, degree,
                time_aug, lead_lag, method, output_scalar_term, n_jobs,
            )
        end
    end

    return _format_signature_output(out, batch_shape)
end

"""
    log_sig(path, degree; time_aug=false, lead_lag=false, end_time=1.0,
        method=LogSigMethod.LyndonWords, scalar_term=false, correction=nothing, n_jobs=1)

Compute the log signature of one path or a batch of paths.

`correction` accepts the layouts described in [`sig`](@ref). Use `method=LogSigMethod.Expanded`
to preserve the full tensor logarithm for non-Lie corrections such as the Ito
lift. Non-empty corrections are unsupported with the direct BCH `method=LogSigMethod.DirectBCH`.
"""
log_sig(path::AbstractArray, degree::Integer; kwargs...) = _log_sig(nothing, nothing, path, degree; kwargs...)

"""
    log_sig!(out, path, degree, scratch=nothing; kwargs...)

Write [`log_sig`](@ref) into `out`, using the same keywords. Supply `scratch`
with the shape of `sig(path, degree; ...)`; its scalar term is included only
for `Expanded` with `scalar_term=true`. `DirectBCH` needs no scratch.
Buffers follow [`sig!`](@ref)'s requirements and must not alias each other.
Direct BCH path transforms may allocate; warmed dense rank-3 CPU calls without
transforms reuse Julia storage. Native caches prepare automatically after clearing.
"""
log_sig!(out::AbstractArray, path::AbstractArray, degree::Integer, scratch=nothing; kwargs...) =
    _log_sig(out, scratch, path, degree; kwargs...)

function _log_sig(
    destination, scratch, path::AbstractArray{T,N},
    degree::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    end_time::Real=1.0,
    method::LogSigMethod.T=LogSigMethod.LyndonWords,
    scalar_term::Bool=false,
    correction=nothing,
    n_jobs::Integer=1,
) where {T<:Real,N}
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    _as_cint_n_jobs(n_jobs)
    path_c, batch_shape, batch_size, dimension, len = _path_buffer(path, :log_sig)
    correction_c = _correction_buffer(correction, path_c, batch_shape, batch_size, dimension, len, degree, lead_lag)

    scalar_term = method == LogSigMethod.Expanded && scalar_term
    method == LogSigMethod.DirectBCH && _validate_bch_correction(correction_c, :log_sig)
    outlen = _log_sig_output_length(dimension, degree, method, scalar_term; time_aug, lead_lag)
    direct = method == LogSigMethod.DirectBCH
    destination !== nothing && !direct && scratch === nothing &&
        throw(ArgumentError("supply a signature scratch buffer for this method"))
    out = _output_buffer(destination, path_c, outlen, batch_shape, batch_size, "out", (path, correction))
    sig_c = direct ? nothing : _output_buffer(scratch, path_c,
        sig_length(dimension, degree; time_aug, lead_lag, scalar_term), batch_shape, batch_size, "scratch", (out, path, correction))
    _log_sig_from_buffers!(out, sig_c, path_c, batch_size, dimension, len, degree, method,
        correction_c.data, correction_c.length, correction_c.batch_stride, correction_c.segment_stride;
        time_aug, lead_lag, end_time, scalar_term, n_jobs)
    return destination === nothing ? _format_signature_output(out, batch_shape) : destination
end

function _log_sig_from_buffers!(out, sig_c, path_c, batch_size, dimension, len, degree, method,
    correction_data, correction_length, correction_batch_stride, correction_segment_stride;
    time_aug=false, lead_lag=false, end_time=1.0, scalar_term=false, n_jobs=1,
)
    isempty(out) && return out
    degree == 0 && return fill!(out, zero(eltype(out)))
    _with_array(path_c) do
        lock(_cache_lock) do
            if method != LogSigMethod.Expanded
                _cache_call(Val(_backend(path_c)), Val(:prepare_log_sig),
                    _effective_dimension(dimension, time_aug, lead_lag), UInt64(degree), method, false)
            end
            if method == LogSigMethod.DirectBCH
                transformed_path, transformed_dimension, transformed_len = _transformed_path_for_log_sig(
                    path_c, batch_size, dimension, len, time_aug, lead_lag, end_time, n_jobs)
                _log_sig_from_path_c!(out, transformed_path, batch_size, transformed_len,
                    transformed_dimension, degree, n_jobs)
            else
                _signature_c!(sig_c, path_c, batch_size, dimension, len, degree,
                    time_aug, lead_lag, true, scalar_term, n_jobs, end_time,
                    _CorrectionBuffer(correction_data, correction_length, correction_batch_stride, correction_segment_stride))
                _sig_to_log_sig_c!(out, sig_c, batch_size, dimension, degree,
                    time_aug, lead_lag, method, scalar_term, n_jobs)
            end
        end
    end
    return out
end

function _sig_to_log_sig_backprop(
    sig_value::AbstractArray{T,N},
    log_sig_derivs::AbstractArray,
    dimension::Integer,
    degree::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    method::LogSigMethod.T=LogSigMethod.LyndonWords,
    n_jobs::Integer=1,
) where {T<:Real,N}
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    _validate_log_sig_conversion_method(method)
    _as_cint_n_jobs(n_jobs)
    S = _storage_eltype(T, :_sig_to_log_sig_backprop)
    sig_c, batch_shape, batch_size, scalar_term = _signature_input_buffer(
        sig_value, S, dimension, degree, "sig";
        time_aug=time_aug, lead_lag=lead_lag,
    )
    outlen = size(sig_c, 1)
    log_sig_len = _log_sig_output_length(
        dimension, degree, method, scalar_term;
        time_aug=time_aug, lead_lag=lead_lag,
    )
    log_sig_derivs_c = _signature_buffer(
        log_sig_derivs, S, log_sig_len, batch_shape, batch_size, "log_sig_derivs", sig_c,
    )
    out = _allocate(sig_c, outlen, batch_size)

    if batch_size != 0 && outlen != 0
        lock(_cache_lock) do
            _prepare_log_sig_if_needed(
                sig_c,
                dimension, degree, method;
                time_aug=time_aug, lead_lag=lead_lag,
            )
            _sig_to_log_sig_backprop_c!(
                out, sig_c, log_sig_derivs_c, batch_size, dimension, degree,
                time_aug, lead_lag, method, scalar_term, n_jobs,
            )
        end
    end

    return _format_signature_output(out, batch_shape)
end

function _logsig_to_sig_backprop(
    log_sig_value::AbstractArray{T,N},
    sig_derivs::AbstractArray,
    dimension::Integer,
    degree::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    method::LogSigMethod.T=LogSigMethod.LyndonWords,
    scalar_term::Bool=false,
    n_jobs::Integer=1,
) where {T<:Real,N}
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    _validate_log_sig_conversion_method(method)
    _as_cint_n_jobs(n_jobs)
    S = _storage_eltype(T, :_logsig_to_sig_backprop)
    log_sig_c, batch_shape, batch_size, inferred_scalar_term = _log_sig_buffer(
        log_sig_value, S, dimension, degree, method, "log_sig";
        time_aug=time_aug, lead_lag=lead_lag, scalar_term=scalar_term,
    )
    output_scalar_term = method == LogSigMethod.Expanded ? inferred_scalar_term : scalar_term
    siglen = sig_length(
        dimension, degree;
        time_aug=time_aug, lead_lag=lead_lag, scalar_term=output_scalar_term,
    )
    sig_derivs_c = _signature_buffer(sig_derivs, S, siglen, batch_shape, batch_size, "sig_derivs", log_sig_c)
    out = _allocate(log_sig_c, size(log_sig_c, 1), batch_size)

    if batch_size != 0 && size(out, 1) != 0
        lock(_cache_lock) do
            _prepare_log_sig_if_needed(
                log_sig_c,
                dimension, degree, method == LogSigMethod.LyndonWords ? LogSigMethod.LyndonBasis : method;
                time_aug=time_aug, lead_lag=lead_lag,
            )
            _logsig_to_sig_backprop_c!(
                out, log_sig_c, sig_derivs_c, batch_size, dimension, degree,
                time_aug, lead_lag, method, output_scalar_term, n_jobs,
            )
        end
    end

    return _format_signature_output(out, batch_shape)
end

function _log_sig_from_path_backprop(
    path::AbstractArray{T,N},
    log_sig_derivs::AbstractArray,
    degree::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    end_time::Real=1.0,
    n_jobs::Integer=1,
) where {T<:Real,N}
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    _as_cint_n_jobs(n_jobs)

    path_c, batch_shape, batch_size, dimension, len = _path_buffer(path, :log_sig)
    S = eltype(path_c)
    siglen = log_sig_length(dimension, degree; time_aug=time_aug, lead_lag=lead_lag)
    log_sig_derivs_c = _signature_buffer(
        log_sig_derivs, S, siglen, batch_shape, batch_size, "log_sig_derivs", path_c,
    )
    out_c = _allocate(path_c, dimension, len, batch_size; zero=true)
    if batch_size != 0 && siglen != 0
        lock(_cache_lock) do
            transformed_path, transformed_dimension, transformed_len = _transformed_path_for_log_sig(
                path_c, batch_size, dimension, len, time_aug, lead_lag, end_time, n_jobs,
            )
            _prepare_log_sig_for(path_c, transformed_dimension, degree, LogSigMethod.DirectBCH)
            transformed_out = (time_aug || lead_lag) ? _allocate(path_c, transformed_dimension, transformed_len, batch_size; zero=true) : out_c
            _log_sig_from_path_backprop_c!(
                transformed_out, log_sig_derivs_c, transformed_path, batch_size,
                transformed_len, transformed_dimension, degree, n_jobs,
            )
            if time_aug || lead_lag
                _transform_path_backprop_c!(
                    out_c, transformed_out, batch_size, dimension, len,
                    time_aug, lead_lag, end_time, n_jobs,
                )
            end
        end
    end
    return _format_path_output(out_c, batch_shape)
end

"""
    log_sig_combine(log_sig1, log_sig2, dimension, degree; n_jobs=1)

Combine two Lyndon-basis log signatures using the BCH formula.
"""
function log_sig_combine(
    log_sig1::AbstractArray{T,N},
    log_sig2::AbstractArray,
    dimension::Integer,
    degree::Integer;
    n_jobs::Integer=1,
) where {T<:Real,N}
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    _as_cint_n_jobs(n_jobs)
    S = _storage_eltype(T, :log_sig_combine)
    len = log_sig_length(dimension, degree)
    log_sig1_c, batch_shape, batch_size = _signature_like_buffer(log_sig1, S, len, "log_sig1")
    log_sig2_c = _signature_buffer(log_sig2, S, len, batch_shape, batch_size, "log_sig2", log_sig1_c)
    out = _allocate(log_sig1_c, len, batch_size)

    if batch_size != 0 && len != 0
        lock(_cache_lock) do
            _prepare_log_sig_for(log_sig1_c, dimension, degree, LogSigMethod.DirectBCH)
            _log_sig_combine_c!(
                out, log_sig1_c, log_sig2_c, batch_size, dimension, degree, n_jobs,
            )
        end
    end

    return _format_signature_output(out, batch_shape)
end

function _log_sig_combine_backprop(
    log_sig1::AbstractArray{T,N},
    log_sig2::AbstractArray,
    derivs::AbstractArray,
    dimension::Integer,
    degree::Integer;
    n_jobs::Integer=1,
) where {T<:Real,N}
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    _as_cint_n_jobs(n_jobs)
    S = _storage_eltype(T, :_log_sig_combine_backprop)
    len = log_sig_length(dimension, degree)
    log_sig1_c, batch_shape, batch_size = _signature_like_buffer(log_sig1, S, len, "log_sig1")
    log_sig2_c = _signature_buffer(log_sig2, S, len, batch_shape, batch_size, "log_sig2", log_sig1_c)
    derivs_c = _signature_buffer(derivs, S, len, batch_shape, batch_size, "derivs", log_sig1_c)
    out1 = _allocate(log_sig1_c, len, batch_size)
    out2 = _allocate(log_sig1_c, len, batch_size)

    if batch_size != 0 && len != 0
        lock(_cache_lock) do
            _prepare_log_sig_for(log_sig1_c, dimension, degree, LogSigMethod.DirectBCH)
            _log_sig_combine_backprop_c!(
                out1, out2, derivs_c, log_sig1_c, log_sig2_c,
                batch_size, dimension, degree, n_jobs,
            )
        end
    end

    return _format_signature_output(out1, batch_shape),
        _format_signature_output(out2, batch_shape)
end

"""
    log_sig_join(log_sig, displacement, dimension, degree; n_jobs=1)

Extend a Lyndon-basis log signature by a single displacement vector.
"""
function log_sig_join(
    log_sig_value::AbstractArray{T,N},
    displacement::AbstractArray,
    dimension::Integer,
    degree::Integer;
    n_jobs::Integer=1,
) where {T<:Real,N}
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    _as_cint_n_jobs(n_jobs)
    S = _storage_eltype(T, :log_sig_join)
    len = log_sig_length(dimension, degree)
    log_sig_c, batch_shape, batch_size = _signature_like_buffer(log_sig_value, S, len, "log_sig")
    displacement_c = _signature_buffer(displacement, S, dimension, batch_shape, batch_size, "displacement", log_sig_c)
    out = _allocate(log_sig_c, len, batch_size)

    if batch_size != 0 && len != 0
        lock(_cache_lock) do
            _prepare_log_sig_for(log_sig_c, dimension, degree, LogSigMethod.DirectBCH)
            _log_sig_join_c!(
                out, log_sig_c, displacement_c, batch_size, dimension, degree, n_jobs,
            )
        end
    end

    return _format_signature_output(out, batch_shape)
end

function _log_sig_join_backprop(
    log_sig_value::AbstractArray{T,N},
    displacement::AbstractArray,
    derivs::AbstractArray,
    dimension::Integer,
    degree::Integer;
    n_jobs::Integer=1,
) where {T<:Real,N}
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    _as_cint_n_jobs(n_jobs)
    S = _storage_eltype(T, :_log_sig_join_backprop)
    len = log_sig_length(dimension, degree)
    log_sig_c, batch_shape, batch_size = _signature_like_buffer(log_sig_value, S, len, "log_sig")
    displacement_c = _signature_buffer(displacement, S, dimension, batch_shape, batch_size, "displacement", log_sig_c)
    derivs_c = _signature_buffer(derivs, S, len, batch_shape, batch_size, "derivs", log_sig_c)
    out_log_sig = _allocate(log_sig_c, len, batch_size)
    out_displacement = _allocate(log_sig_c, dimension, batch_size)

    if batch_size != 0 && len != 0
        lock(_cache_lock) do
            _prepare_log_sig_for(log_sig_c, dimension, degree, LogSigMethod.DirectBCH)
            _log_sig_join_backprop_c!(
                out_log_sig, out_displacement, derivs_c, log_sig_c, displacement_c,
                batch_size, dimension, degree, n_jobs,
            )
        end
    end

    return _format_signature_output(out_log_sig, batch_shape),
        _format_signature_output(out_displacement, batch_shape)
end
