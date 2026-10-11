function _as_uint64(x::Integer, name::AbstractString)
    x >= 0 || throw(ArgumentError("$name must be nonnegative"))
    x <= typemax(UInt64) || throw(OverflowError("$name exceeds UInt64"))
    return UInt64(x)
end

function _as_cint_n_jobs(n_jobs::Integer)
    n_jobs != 0 || throw(ArgumentError("n_jobs cannot be 0"))
    typemin(Cint) <= n_jobs <= typemax(Cint) ||
        throw(ArgumentError("n_jobs must fit in Cint"))
    return Cint(n_jobs)
end

function _effective_dimension(dimension::Integer, time_aug::Bool, lead_lag::Bool)
    dimension >= 1 || throw(ArgumentError("dimension must be positive"))
    d = _as_uint64(dimension, "dimension")
    lead_lag && (d = Base.checked_mul(UInt64(2), d))
    return time_aug ? Base.checked_add(d, UInt64(1)) : d
end

function _as_log_sig_method(method::Integer, name::AbstractString="method")
    0 <= method <= 3 || throw(ArgumentError("$name must be one of 0, 1, 2, or 3"))
    return Cint(method)
end

function _as_cached_log_sig_method(method::Integer, name::AbstractString="method")
    0 <= method <= 2 || throw(ArgumentError("$name must be one of 0, 1, or 2"))
    return Cint(method)
end

_storage_eltype(::Type{Float32}, ::Symbol) = Float32
_storage_eltype(::Type{Float64}, ::Symbol) = Float64

function _storage_eltype(::Type{T}, function_name::Symbol) where {T<:Real}
    throw(TypeError(function_name, "path eltype must be Float32 or Float64", Union{Float32,Float64}, T))
end

_storage_eltype(::Type{T}) where {T<:Real} = _storage_eltype(T, :sig)

function _validate_standard_signature_options(correction, function_name::Symbol=:sig)
    correction === nothing ||
        throw(ArgumentError("correction is not supported by SigLib.$function_name yet"))
    return nothing
end

function _path_buffer(path::AbstractArray{T,N}, function_name::Symbol=:sig) where {T<:Real,N}
    N >= 2 || throw(ArgumentError("path must have at least rank 2"))

    S = _storage_eltype(T, function_name)
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
        return vec(out)
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

function _signature_like_buffer(
    x::AbstractArray{T,N},
    ::Type{S},
    siglen::Integer,
    name::AbstractString,
) where {T<:Real,N,S<:Union{Float32,Float64}}
    N >= 1 || throw(ArgumentError("$name must have at least rank 1"))

    batch_shape = size(x)[1:end - 1]
    batch_size = isempty(batch_shape) ? 1 : prod(batch_shape)
    return _signature_buffer(x, S, siglen, batch_shape, batch_size, name), batch_shape, batch_size
end
