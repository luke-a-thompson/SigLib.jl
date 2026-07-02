function _log_sig_output_length(
    dimension::Integer,
    degree::Integer,
    method::Integer,
    scalar_term::Bool;
    time_aug::Bool=false,
    lead_lag::Bool=false,
)
    _as_log_sig_method(method)
    if method == 0
        return sig_length(
            dimension, degree;
            time_aug=time_aug, lead_lag=lead_lag, scalar_term=scalar_term,
        )
    end

    return log_sig_length(dimension, degree; time_aug=time_aug, lead_lag=lead_lag)
end

function _prepare_log_sig_if_needed(
    dimension::Integer,
    degree::Integer,
    method::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
)
    if method == 1 || method == 2
        prepare_log_sig(dimension, degree, method; time_aug=time_aug, lead_lag=lead_lag)
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
    siglen_scalar = sig_length(
        dimension, degree;
        time_aug=time_aug, lead_lag=lead_lag, scalar_term=true,
    )
    lastdim = size(x)[end]
    if lastdim == siglen
        buf, batch_shape, batch_size = _signature_like_buffer(x, S, siglen, name)
        return buf, batch_shape, batch_size, false
    elseif lastdim == siglen_scalar
        buf, batch_shape, batch_size = _signature_like_buffer(x, S, siglen_scalar, name)
        return buf, batch_shape, batch_size, true
    end

    throw(ArgumentError(
        "$name must have last dimension $siglen or $siglen_scalar, got $lastdim",
    ))
end

function _log_sig_buffer(
    x::AbstractArray{T,N},
    ::Type{S},
    dimension::Integer,
    degree::Integer,
    method::Integer,
    name::AbstractString;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    scalar_term::Bool=false,
) where {T<:Real,N,S<:Union{Float32,Float64}}
    if method == 0
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
    path_c::Array{T,3},
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

    out = Array{T,3}(undef, transformed_dimension, transformed_len, batch_size)
    _transform_path_c!(
        out, path_c, batch_size, dimension, len,
        time_aug, lead_lag, end_time, n_jobs,
    )
    return out, transformed_dimension, transformed_len
end

function _format_displacement_output(out::Matrix{T}, batch_shape::Tuple) where {T}
    dimension, _ = size(out)
    batch_first = permutedims(out, (2, 1))
    if isempty(batch_shape)
        return reshape(batch_first, dimension)
    end

    return reshape(batch_first, batch_shape..., dimension)
end

function _displacement_buffer(
    displacement::AbstractArray{T,N},
    ::Type{S},
    dimension::Integer,
    batch_shape::Tuple,
    batch_size::Integer,
    name::AbstractString,
) where {T<:Real,N,S<:Union{Float32,Float64}}
    expected_size = isempty(batch_shape) ? (dimension,) : (batch_shape..., dimension)
    size(displacement) == expected_size ||
        throw(ArgumentError("$name must have shape $expected_size, got $(size(displacement))"))

    A = displacement isa Array{S,N} ? displacement : Array{S,N}(displacement)
    if isempty(batch_shape)
        return reshape(A, dimension, batch_size)
    end

    batch_first = reshape(A, batch_size, dimension)
    return permutedims(batch_first, (2, 1))
end

"""
    sig_to_log_sig(sig, dimension, degree; time_aug=false, lead_lag=false,
        method=1, n_jobs=1)

Compute log signatures from signatures using pySigLib's method convention.
"""
function sig_to_log_sig(
    sig_value::AbstractArray{T,N},
    dimension::Integer,
    degree::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    method::Integer=1,
    n_jobs::Integer=1,
) where {T<:Real,N}
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    method == 3 && throw(ArgumentError("method=3 is only supported by log_sig"))
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
    out = Matrix{S}(undef, outlen, batch_size)

    if batch_size != 0 && outlen != 0
        _prepare_log_sig_if_needed(
            dimension, degree, method;
            time_aug=time_aug, lead_lag=lead_lag,
        )
        _sig_to_log_sig_c!(
            out, sig_c, batch_size, dimension, degree,
            time_aug, lead_lag, method, scalar_term, n_jobs,
        )
    end

    return _format_signature_output(out, batch_shape)
end

"""
    logsig_to_sig(log_sig, dimension, degree; time_aug=false, lead_lag=false,
        method=1, scalar_term=false, n_jobs=1)

Compute signatures from log signatures via truncated tensor exponential.
"""
function logsig_to_sig(
    log_sig_value::AbstractArray{T,N},
    dimension::Integer,
    degree::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    method::Integer=1,
    scalar_term::Bool=false,
    n_jobs::Integer=1,
) where {T<:Real,N}
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    method == 3 && throw(ArgumentError("method=3 is not supported by logsig_to_sig"))
    _as_cint_n_jobs(n_jobs)
    S = _storage_eltype(T, :logsig_to_sig)
    log_sig_c, batch_shape, batch_size, inferred_scalar_term = _log_sig_buffer(
        log_sig_value, S, dimension, degree, method, "log_sig";
        time_aug=time_aug, lead_lag=lead_lag, scalar_term=scalar_term,
    )
    output_scalar_term = method == 0 ? inferred_scalar_term : scalar_term
    outlen = sig_length(
        dimension, degree;
        time_aug=time_aug, lead_lag=lead_lag, scalar_term=output_scalar_term,
    )
    out = Matrix{S}(undef, outlen, batch_size)

    if batch_size != 0 && outlen != 0
        _prepare_log_sig_if_needed(
            dimension, degree, method;
            time_aug=time_aug, lead_lag=lead_lag,
        )
        _logsig_to_sig_c!(
            out, log_sig_c, batch_size, dimension, degree,
            time_aug, lead_lag, method, output_scalar_term, n_jobs,
        )
    end

    return _format_signature_output(out, batch_shape)
end

"""
    log_sig(path, degree; time_aug=false, lead_lag=false, end_time=1.0,
        method=1, scalar_term=false, n_jobs=1)

Compute the log signature of one path or a batch of paths.
"""
function log_sig(
    path::AbstractArray{T,N},
    degree::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    end_time::Real=1.0,
    method::Integer=1,
    scalar_term::Bool=false,
    n_jobs::Integer=1,
) where {T<:Real,N}
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    _as_log_sig_method(method)
    _as_cint_n_jobs(n_jobs)

    if method == 3
        path_c, batch_shape, batch_size, dimension, len = _path_buffer(path, :log_sig)
        outlen = log_sig_length(dimension, degree; time_aug=time_aug, lead_lag=lead_lag)
        out = Matrix{eltype(path_c)}(undef, outlen, batch_size)
        if batch_size != 0 && outlen != 0
            transformed_path, transformed_dimension, transformed_len = _transformed_path_for_log_sig(
                path_c, batch_size, dimension, len, time_aug, lead_lag, end_time, n_jobs,
            )
            _log_sig_from_path_c!(
                out, transformed_path, batch_size, transformed_len,
                transformed_dimension, degree, n_jobs,
            )
        end
        return _format_signature_output(out, batch_shape)
    end

    sig_scalar_term = method == 0 ? scalar_term : false
    sig_value = sig(
        path, degree;
        time_aug=time_aug,
        lead_lag=lead_lag,
        end_time=end_time,
        scalar_term=sig_scalar_term,
        n_jobs=n_jobs,
    )
    return sig_to_log_sig(
        sig_value, size(path, N), degree;
        time_aug=time_aug,
        lead_lag=lead_lag,
        method=method,
        n_jobs=n_jobs,
    )
end

function _sig_to_log_sig_backprop(
    sig_value::AbstractArray{T,N},
    log_sig_derivs::AbstractArray,
    dimension::Integer,
    degree::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    method::Integer=1,
    n_jobs::Integer=1,
) where {T<:Real,N}
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    method == 3 && throw(ArgumentError("method=3 is only supported by log_sig"))
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
        log_sig_derivs, S, log_sig_len, batch_shape, batch_size, "log_sig_derivs",
    )
    out = Matrix{S}(undef, outlen, batch_size)

    if batch_size != 0 && outlen != 0
        _prepare_log_sig_if_needed(
            dimension, degree, method;
            time_aug=time_aug, lead_lag=lead_lag,
        )
        _sig_to_log_sig_backprop_c!(
            out, sig_c, log_sig_derivs_c, batch_size, dimension, degree,
            time_aug, lead_lag, method, scalar_term, n_jobs,
        )
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
    method::Integer=1,
    scalar_term::Bool=false,
    n_jobs::Integer=1,
) where {T<:Real,N}
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    method == 3 && throw(ArgumentError("method=3 is not supported by logsig_to_sig"))
    _as_cint_n_jobs(n_jobs)
    S = _storage_eltype(T, :_logsig_to_sig_backprop)
    log_sig_c, batch_shape, batch_size, inferred_scalar_term = _log_sig_buffer(
        log_sig_value, S, dimension, degree, method, "log_sig";
        time_aug=time_aug, lead_lag=lead_lag, scalar_term=scalar_term,
    )
    output_scalar_term = method == 0 ? inferred_scalar_term : scalar_term
    siglen = sig_length(
        dimension, degree;
        time_aug=time_aug, lead_lag=lead_lag, scalar_term=output_scalar_term,
    )
    sig_derivs_c = _signature_buffer(sig_derivs, S, siglen, batch_shape, batch_size, "sig_derivs")
    out = Matrix{S}(undef, size(log_sig_c, 1), batch_size)

    if batch_size != 0 && size(out, 1) != 0
        _prepare_log_sig_if_needed(
            dimension, degree, method;
            time_aug=time_aug, lead_lag=lead_lag,
        )
        _logsig_to_sig_backprop_c!(
            out, log_sig_c, sig_derivs_c, batch_size, dimension, degree,
            time_aug, lead_lag, method, output_scalar_term, n_jobs,
        )
    end

    return _format_signature_output(out, batch_shape)
end

function _log_sig_backprop(
    path::AbstractArray{T,N},
    log_sig_value::AbstractArray,
    log_sig_derivs::AbstractArray,
    degree::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    end_time::Real=1.0,
    method::Integer=1,
    scalar_term::Bool=false,
    n_jobs::Integer=1,
) where {T<:Real,N}
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    _as_log_sig_method(method)
    _as_cint_n_jobs(n_jobs)

    if method == 3
        path_c, batch_shape, batch_size, dimension, len = _path_buffer(path, :log_sig)
        S = eltype(path_c)
        siglen = log_sig_length(dimension, degree; time_aug=time_aug, lead_lag=lead_lag)
        _signature_buffer(log_sig_value, S, siglen, batch_shape, batch_size, "log_sig_value")
        log_sig_derivs_c = _signature_buffer(
            log_sig_derivs, S, siglen, batch_shape, batch_size, "log_sig_derivs",
        )
        out_c = zeros(S, dimension, len, batch_size)
        if batch_size != 0 && siglen != 0
            transformed_path, transformed_dimension, transformed_len = _transformed_path_for_log_sig(
                path_c, batch_size, dimension, len, time_aug, lead_lag, end_time, n_jobs,
            )
            transformed_out = zeros(S, transformed_dimension, transformed_len, batch_size)
            _log_sig_from_path_backprop_c!(
                transformed_out, log_sig_derivs_c, transformed_path, batch_size,
                transformed_len, transformed_dimension, degree, n_jobs,
            )
            if time_aug || lead_lag
                _transform_path_backprop_c!(
                    out_c, transformed_out, batch_size, dimension, len,
                    time_aug, lead_lag, end_time, n_jobs,
                )
            else
                out_c .= transformed_out
            end
        end
        return _format_path_output(out_c, batch_shape)
    end

    sig_scalar_term = method == 0 ? scalar_term : false
    sig_value = sig(
        path, degree;
        time_aug=time_aug,
        lead_lag=lead_lag,
        end_time=end_time,
        scalar_term=sig_scalar_term,
        n_jobs=n_jobs,
    )
    sig_derivs = _sig_to_log_sig_backprop(
        sig_value, log_sig_derivs, size(path, N), degree;
        time_aug=time_aug,
        lead_lag=lead_lag,
        method=method,
        n_jobs=n_jobs,
    )
    # Validate the provided primal while keeping the path pullback based on the
    # recomputed signature, which is what the composed implementation needs.
    _signature_like_buffer(
        log_sig_value, eltype(path), size(log_sig_derivs, ndims(log_sig_derivs)),
        "log_sig_value",
    )
    return _sig_backprop(
        path, sig_value, sig_derivs, degree;
        time_aug=time_aug,
        lead_lag=lead_lag,
        end_time=end_time,
        scalar_term=sig_scalar_term,
        n_jobs=n_jobs,
    )
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
    log_sig2_c = _signature_buffer(log_sig2, S, len, batch_shape, batch_size, "log_sig2")
    out = Matrix{S}(undef, len, batch_size)

    if batch_size != 0 && len != 0
        prepare_log_sig(dimension, degree, 2)
        _log_sig_combine_c!(
            out, log_sig1_c, log_sig2_c, batch_size, dimension, degree, n_jobs,
        )
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
    log_sig2_c = _signature_buffer(log_sig2, S, len, batch_shape, batch_size, "log_sig2")
    derivs_c = _signature_buffer(derivs, S, len, batch_shape, batch_size, "derivs")
    out1 = Matrix{S}(undef, len, batch_size)
    out2 = Matrix{S}(undef, len, batch_size)

    if batch_size != 0 && len != 0
        prepare_log_sig(dimension, degree, 2)
        _log_sig_combine_backprop_c!(
            out1, out2, derivs_c, log_sig1_c, log_sig2_c,
            batch_size, dimension, degree, n_jobs,
        )
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
    displacement_c = _displacement_buffer(displacement, S, dimension, batch_shape, batch_size, "displacement")
    out = Matrix{S}(undef, len, batch_size)

    if batch_size != 0 && len != 0
        prepare_log_sig(dimension, degree, 2)
        _log_sig_join_c!(
            out, log_sig_c, displacement_c, batch_size, dimension, degree, n_jobs,
        )
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
    displacement_c = _displacement_buffer(displacement, S, dimension, batch_shape, batch_size, "displacement")
    derivs_c = _signature_buffer(derivs, S, len, batch_shape, batch_size, "derivs")
    out_log_sig = Matrix{S}(undef, len, batch_size)
    out_displacement = Matrix{S}(undef, dimension, batch_size)

    if batch_size != 0 && len != 0
        prepare_log_sig(dimension, degree, 2)
        _log_sig_join_backprop_c!(
            out_log_sig, out_displacement, derivs_c, log_sig_c, displacement_c,
            batch_size, dimension, degree, n_jobs,
        )
    end

    return _format_signature_output(out_log_sig, batch_shape),
        _format_displacement_output(out_displacement, batch_shape)
end
