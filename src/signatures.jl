_storage_eltype(::Type{Float32}) = Float32
_storage_eltype(::Type{Float64}) = Float64

function _storage_eltype(::Type{T}) where {T<:Real}
    throw(TypeError(:sig, "path eltype must be Float32 or Float64", Union{Float32,Float64}, T))
end

function _validate_standard_signature_options(correction)
    correction === nothing ||
        throw(ArgumentError("correction is not supported by SigLib.sig yet"))
    return nothing
end

function _path_buffer(path::AbstractArray{T,N}) where {T<:Real,N}
    N >= 2 || throw(ArgumentError("path must have at least rank 2"))

    S = _storage_eltype(T)
    A = path isa Array{S,N} ? path : Array{S,N}(path)
    path_size = size(A)
    len = path_size[end - 1]
    dimension = path_size[end]
    batch_shape = path_size[1:end - 2]
    batch_size = isempty(batch_shape) ? 1 : prod(batch_shape)

    dimension >= 1 || throw(ArgumentError("path has 0 channels; dimension must be positive"))
    len >= 1 || throw(ArgumentError("path length must be positive"))

    # Public shape follows pySigLib: (batch..., length, dimension). The C ABI is
    # row-major [batch][length][dimension], equivalent to Julia (dimension, length, batch).
    reshaped = reshape(A, batch_size, len, dimension)
    return permutedims(reshaped, (3, 2, 1)), batch_shape, batch_size, dimension, len
end

function _format_signature_output(out::Matrix{T}, batch_shape::Tuple) where {T}
    if isempty(batch_shape)
        return out[:, 1]
    end

    batch_first = permutedims(out, (2, 1))
    return reshape(batch_first, batch_shape..., size(out, 1))
end

function _signature_buffer(
    x::AbstractArray,
    ::Type{T},
    siglen::Integer,
    batch_shape::Tuple,
    batch_size::Integer,
    name::AbstractString,
) where {T<:Union{Float32,Float64}}
    expected_size = isempty(batch_shape) ? (siglen,) : (batch_shape..., siglen)
    size(x) == expected_size ||
        throw(ArgumentError("$name must have shape $expected_size, got $(size(x))"))

    A = x isa Array{T} ? x : Array{T}(x)
    if isempty(batch_shape)
        return reshape(A, siglen, batch_size)
    end

    batch_first = reshape(A, batch_size, siglen)
    return permutedims(batch_first, (2, 1))
end

function _format_path_output(out::Array{T,3}, batch_shape::Tuple) where {T}
    dimension, len, _ = size(out)
    batch_first = permutedims(out, (3, 2, 1))
    if isempty(batch_shape)
        return reshape(batch_first, len, dimension)
    end

    return reshape(batch_first, batch_shape..., len, dimension)
end

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
    siglen = sig_length(
        dimension, degree;
        time_aug=time_aug, lead_lag=lead_lag, scalar_term=scalar_term,
    )
    out = Matrix{eltype(path_c)}(undef, siglen, batch_size)

    if batch_size != 0 && siglen != 0
        _signature_c!(
            out, path_c, batch_size, dimension, len, degree,
            time_aug, lead_lag, horner, scalar_term, n_jobs, end_time,
        )
    end

    return _format_signature_output(out, batch_shape)
end

"""
    sig_backprop(path, sig_value, sig_derivs, degree; time_aug=false,
        lead_lag=false, end_time=1.0, scalar_term=false, correction=nothing,
        n_jobs=1)

Backpropagate a signature cotangent to the underlying path.

`sig_value` must be the result of `sig(path, degree; same_options...)`, and
`sig_derivs` must have the same shape. The returned array has the same shape
and element type as `path`.
"""
function sig_backprop(
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
    sig_c = _signature_buffer(sig_value, S, siglen, batch_shape, batch_size, "sig_value")
    sig_derivs_c = _signature_buffer(sig_derivs, S, siglen, batch_shape, batch_size, "sig_derivs")

    out_c = zeros(S, dimension, len, batch_size)
    if batch_size != 0 && siglen != 0
        _sig_backprop_c!(
            out_c, path_c, sig_derivs_c, sig_c, batch_size, dimension, len, degree,
            time_aug, lead_lag, scalar_term, n_jobs, end_time,
        )
    end

    return _format_path_output(out_c, batch_shape)
end
