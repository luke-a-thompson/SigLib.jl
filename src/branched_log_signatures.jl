_validate_branched_log_sig_degree(::AbstractArray, degree, method) = nothing

function _branched_log_sig_method(method::Union{Nothing,Integer}, planar::Bool; conversion::Bool=false)
    m = method === nothing ? Cint(planar) : _as_log_sig_method(method)
    m != 0 && !planar && throw(ArgumentError("branched log signature methods 1–3 require planar=true"))
    conversion && m == 3 && throw(ArgumentError("method=3 requires branched_log_sig(path, degree)"))
    m != 0 && !_modern_abi[] && throw(ArgumentError("compressed branched log signatures require a pySigLib 4.x native artifact"))
    return m
end

_branched_log_sig_dimension(dimension, time_aug, lead_lag) =
    dimension == 0 ? UInt64(time_aug) : _effective_dimension(dimension, time_aug, lead_lag)

function _branched_log_sig_input(bsig::AbstractArray{T,N}, dimension, degree, time_aug, lead_lag, planar, method) where {T<:Real,N}
    N >= 1 || throw(ArgumentError("bsig must have at least rank 1"))
    d = _branched_log_sig_dimension(dimension, time_aug, lead_lag)
    _as_uint64(degree, "degree")
    len = d == 0 ? 0 : branched_sig_length(d, degree; planar=planar)
    scalar_term = size(bsig, N) == len + 1
    siglen = len + scalar_term
    S = _storage_eltype(T, :branched_sig_to_log_sig)
    bsig_c, batch_shape, batch_size = _signature_like_buffer(bsig, S, siglen, "bsig")
    outlen = method == 0 ? siglen : branched_log_sig_length(d, degree; planar=true)
    return bsig_c, batch_shape, batch_size, d, scalar_term, outlen
end

"""
    branched_sig_to_log_sig(bsig, dimension, degree; time_aug=false, lead_lag=false, planar=false, method=nothing, n_jobs=1)

Convert a branched signature to its Hopf logarithm. Method 0 preserves the input's scalar-term format; planar methods 1 (extracted coordinates) and 2 (MKW basis coordinates) return compressed scalar-free outputs. The default is 1 for planar signatures and 0 otherwise. Method 3 requires a path.
"""
function branched_sig_to_log_sig(
    bsig::AbstractArray{T,N}, dimension::Integer, degree::Integer;
    time_aug::Bool=false, lead_lag::Bool=false, planar::Bool=false,
    method::Union{Nothing,Integer}=nothing, n_jobs::Integer=1,
) where {T<:Real,N}
    m = _branched_log_sig_method(method, planar; conversion=true)
    _as_cint_n_jobs(n_jobs)
    bsig_c, batch_shape, batch_size, d, scalar_term, outlen =
        _branched_log_sig_input(bsig, dimension, degree, time_aug, lead_lag, planar, m)
    out = _allocate(bsig_c, outlen, batch_size)
    if batch_size != 0 && outlen != 0
        lock(_cache_lock) do
            _prepare_branched_log_sig_for(bsig_c, d, degree, m; planar=planar)
            _branched_sig_to_log_sig_c!(out, bsig_c, batch_size, d, degree, planar, scalar_term, n_jobs, m)
        end
    end
    return _format_signature_output(out, batch_shape)
end

"""
    branched_log_sig(path, degree; time_aug=false, lead_lag=false, end_time=1.0, planar=false, scalar_term=false, method=nothing, correction=nothing, n_jobs=1)

Compute branched log signatures, defaulting to method 1 for planar paths and method 0 otherwise. Methods 1–3 require `planar=true` and ignore `scalar_term`; method 3 computes MKW basis coordinates directly through BCH. Non-empty corrections are currently unsupported.
"""
function branched_log_sig(
    path::AbstractArray{T,N}, degree::Integer;
    time_aug::Bool=false, lead_lag::Bool=false, end_time::Real=1.0,
    planar::Bool=false, scalar_term::Bool=false, method::Union{Nothing,Integer}=nothing,
    correction=nothing, n_jobs::Integer=1,
) where {T<:Real,N}
    m = _branched_log_sig_method(method, planar)
    _validate_branched_log_sig_degree(path, degree, m)
    _as_cint_n_jobs(n_jobs)
    correction isa AbstractArray && isempty(correction) && (correction = nothing)
    _validate_standard_signature_options(correction, :branched_log_sig)
    if m == 3
        path_c, batch_shape, batch_size, dimension, len = _path_buffer(path, :branched_log_sig)
        d = _effective_dimension(dimension, time_aug, lead_lag)
        outlen = branched_log_sig_length(d, degree; planar=true)
        out = _allocate(path_c, outlen, batch_size)
        if batch_size != 0 && outlen != 0
            lock(_cache_lock) do
                transformed_path, d, len = _transformed_path_for_log_sig(
                    path_c, batch_size, dimension, len, time_aug, lead_lag, end_time, n_jobs)
                _prepare_branched_log_sig_for(path_c, d, degree, 3; planar=true)
                _branched_log_sig_from_path_c!(out, transformed_path, batch_size, len, d, degree, n_jobs)
            end
        end
        return _format_signature_output(out, batch_shape)
    end
    bsig = branched_sig(path, degree;
        time_aug=time_aug, lead_lag=lead_lag, end_time=end_time,
        planar=planar, scalar_term=scalar_term, n_jobs=n_jobs)
    return branched_sig_to_log_sig(bsig, size(path, N), degree;
        time_aug=time_aug, lead_lag=lead_lag, planar=planar, method=m, n_jobs=n_jobs)
end

function _branched_sig_to_log_sig_backprop(
    bsig::AbstractArray{T,N}, derivs::AbstractArray, dimension::Integer, degree::Integer;
    time_aug::Bool=false, lead_lag::Bool=false, planar::Bool=false,
    method::Union{Nothing,Integer}=nothing, n_jobs::Integer=1,
) where {T<:Real,N}
    m = _branched_log_sig_method(method, planar; conversion=true)
    _as_cint_n_jobs(n_jobs)
    bsig_c, batch_shape, batch_size, d, scalar_term, outlen =
        _branched_log_sig_input(bsig, dimension, degree, time_aug, lead_lag, planar, m)
    derivs_c = _signature_buffer(derivs, eltype(bsig_c), outlen, batch_shape, batch_size, "derivs", bsig_c)
    out = _allocate(bsig_c, size(bsig_c)...; zero=true)
    if batch_size != 0 && outlen != 0
        lock(_cache_lock) do
            _prepare_branched_log_sig_for(bsig_c, d, degree, m; planar=planar)
            _branched_sig_to_log_sig_backprop_c!(out, bsig_c, derivs_c, batch_size, d, degree, planar, scalar_term, n_jobs, m)
        end
    end
    return _format_signature_output(out, batch_shape)
end

function _branched_log_sig_from_path_backprop(
    path::AbstractArray, derivs::AbstractArray, degree::Integer;
    time_aug::Bool=false, lead_lag::Bool=false, end_time::Real=1.0, n_jobs::Integer=1,
)
    _branched_log_sig_method(3, true)
    _validate_branched_log_sig_degree(path, degree, 3)
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
            _prepare_branched_log_sig_for(path_c, d, degree, 3; planar=true)
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
