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

struct _CorrectionBuffer{A}
    data::A
    length::UInt64
    batch_stride::UInt64
    segment_stride::UInt64
end

_CorrectionBuffer() = _CorrectionBuffer(nothing, UInt64(0), UInt64(0), UInt64(0))

function _correction_buffer(correction, like, batch_shape, batch_size, dimension, len, degree, lead_lag)
    correction === nothing && return _CorrectionBuffer()
    correction isa AbstractArray || throw(ArgumentError("correction must be an array or nothing"))
    eltype(correction) === eltype(like) ||
        throw(ArgumentError("correction and path must have the same eltype"))
    _check_backend(like, correction)
    isempty(correction) && return _CorrectionBuffer()
    lead_lag && throw(ArgumentError("correction cannot be combined with lead_lag=true"))
    ndims(correction) >= 1 || throw(ArgumentError("correction must have at least rank 1"))
    degree >= 2 || throw(ArgumentError("correction must be empty when degree < 2"))

    width = size(correction, 1)
    # Accept only complete prefixes of tensor levels 2..degree. Stop before
    # overflowing when a level is already wider than the supplied correction.
    total, level_size = 0, dimension
    for level in 2:degree
        level_size > div(width - total, dimension) && break
        level_size *= dimension
        total += level_size
        total == width && break
    end
    total == width || throw(ArgumentError("correction width must be d^2 + ... + d^m for 2 <= m <= degree"))

    segments = len - 1
    return _with_array(like) do
        if ndims(correction) == 1
            return _CorrectionBuffer(_materialize(correction, eltype(like)), UInt64(width), UInt64(0), UInt64(0))
        elseif ndims(correction) == 2 && size(correction, 2) == segments
            packed = _materialize(correction, eltype(like))
            return _CorrectionBuffer(packed, UInt64(width), UInt64(0), UInt64(width))
        end

        expected = (width, segments, batch_shape...)
        size(correction) == expected || throw(ArgumentError(
            "correction must have shape ($width,), ($width, $segments), or $expected, got $(size(correction))",
        ))
        packed = _reshape_buffer(_materialize(correction, eltype(like)), (width, segments, batch_size))
        return _CorrectionBuffer(packed, UInt64(width), UInt64(Base.checked_mul(segments, width)), UInt64(width))
    end
end

function _validate_bch_correction(correction::_CorrectionBuffer, function_name::Symbol)
    correction.length == 0 || throw(ArgumentError("correction is not supported with $function_name method=3"))
    return nothing
end

# Backend hooks keep layout, validation and numerical flows shared by CPU and CUDA.
_backend(::AbstractArray) = :cpu
_with_array(f, ::AbstractArray) = f()
_materialize(x::AbstractArray, ::Type{T}) where {T} = x isa Array{T} ? x : Array{T}(x)
_reshape_buffer(x, dims::Tuple) = size(x) == dims ? x : reshape(x, dims)

function _allocate(like::AbstractArray, dims::Integer...; zero::Bool=false)
    return _with_array(like) do
        out = similar(like, dims...)
        zero && fill!(out, Base.zero(eltype(out)))
        out
    end
end

function _check_backend(like::AbstractArray, x::AbstractArray)
    _backend(like) == _backend(x) || throw(ArgumentError("all inputs must use the same CPU or CUDA backend"))
    return nothing
end

function _path_buffer(path::AbstractArray{T,N}, function_name::Symbol=:sig) where {T<:Real,N}
    return _with_array(path) do
        N >= 2 || throw(ArgumentError("path must have at least rank 2"))

        S = _storage_eltype(T, function_name)
        path_size = size(path)
        dimension = path_size[1]
        len = path_size[2]
        batch_shape = path_size[3:end]
        batch_size = isempty(batch_shape) ? 1 : prod(batch_shape)

        dimension >= 1 || throw(ArgumentError("path has 0 channels; dimension must be positive"))
        len >= 1 || throw(ArgumentError("path length must be positive"))

        # Julia's channel-first, batch-last storage matches the row-major C ABI.
        # Dense inputs share storage; only views and precision conversions copy.
        return _reshape_buffer(_materialize(path, S), (dimension, len, batch_size)),
            batch_shape, batch_size, dimension, len
    end
end

function _format_signature_output(out::AbstractMatrix{T}, batch_shape::Tuple) where {T}
    return _with_array(out) do
        if isempty(batch_shape)
            return vec(out)
        end

        return _reshape_buffer(out, (size(out, 1), batch_shape...))
    end
end

function _signature_buffer(
    x::AbstractArray,
    ::Type{T},
    siglen::Integer,
    batch_shape::Tuple,
    batch_size::Integer,
    name::AbstractString,
    like::AbstractArray=x,
) where {T<:Union{Float32,Float64}}
    return _with_array(x) do
        expected_size = (siglen, batch_shape...)
        size(x) == expected_size ||
            throw(ArgumentError("$name must have shape $expected_size, got $(size(x))"))

        _check_backend(like, x)
        return _reshape_buffer(_materialize(x, T), (siglen, batch_size))
    end
end

function _format_path_output(out::AbstractArray{T,3}, batch_shape::Tuple) where {T}
    return _with_array(out) do
        dimension, len, _ = size(out)
        return _reshape_buffer(out, (dimension, len, batch_shape...))
    end
end

function _signature_like_buffer(
    x::AbstractArray{T,N},
    ::Type{S},
    siglen::Integer,
    name::AbstractString,
) where {T<:Real,N,S<:Union{Float32,Float64}}
    N >= 1 || throw(ArgumentError("$name must have at least rank 1"))

    batch_shape = size(x)[2:end]
    batch_size = isempty(batch_shape) ? 1 : prod(batch_shape)
    return _signature_buffer(x, S, siglen, batch_shape, batch_size, name), batch_shape, batch_size
end
