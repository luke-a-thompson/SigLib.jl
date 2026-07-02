using cpsig_jll: libcpsig

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
    d = big(dimension)
    d >= 1 || throw(ArgumentError("dimension must be positive"))
    return (lead_lag ? 2d : d) + (time_aug ? 1 : 0)
end

"""
    sig_length(dimension, degree; time_aug=false, lead_lag=false, scalar_term=false)

Return the length of the truncated signature for `dimension` channels and
truncation `degree`. By default this matches pySigLib and excludes the leading
scalar term.
"""
function sig_length(
    dimension::Integer,
    degree::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    scalar_term::Bool=false,
)
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    effective_dimension = _effective_dimension(dimension, time_aug, lead_lag)

    full_length = ccall(
        (:sig_length, libcpsig),
        UInt64,
        (UInt64, UInt64),
        _as_uint64(effective_dimension, "dimension"),
        _as_uint64(degree, "degree"),
    )

    full_length != 0 || throw(OverflowError("cpsig signature length overflowed"))
    full_length <= UInt64(typemax(Int)) ||
        throw(OverflowError("signature length exceeds Int"))

    return Int(full_length) - (scalar_term ? 0 : 1)
end

function _as_log_sig_method(method::Integer, name::AbstractString="method")
    0 <= method <= 3 || throw(ArgumentError("$name must be one of 0, 1, 2, or 3"))
    return Cint(method)
end

function _as_cached_log_sig_method(method::Integer, name::AbstractString="method")
    0 <= method <= 2 || throw(ArgumentError("$name must be one of 0, 1, or 2"))
    return Cint(method)
end

"""
    log_sig_length(dimension, degree; time_aug=false, lead_lag=false)

Return the length of a truncated compressed log signature for `dimension`
channels and truncation `degree`.
"""
function log_sig_length(
    dimension::Integer,
    degree::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
)
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    effective_dimension = _effective_dimension(dimension, time_aug, lead_lag)

    len = ccall(
        (:log_sig_length, libcpsig),
        UInt64,
        (UInt64, UInt64),
        _as_uint64(effective_dimension, "dimension"),
        _as_uint64(degree, "degree"),
    )

    len <= UInt64(typemax(Int)) || throw(OverflowError("log signature length exceeds Int"))
    return Int(len)
end

"""
    prepare_log_sig(dimension, degree, method; time_aug=false, lead_lag=false,
        use_disk=false)

Prepare cpsig's cache for normal log-signature computations.
"""
function prepare_log_sig(
    dimension::Integer,
    degree::Integer,
    method::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    use_disk::Bool=false,
)
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    effective_dimension = _effective_dimension(dimension, time_aug, lead_lag)
    status = ccall(
        (:prepare_log_sig, libcpsig),
        Cint,
        (UInt64, UInt64, Cint, Bool),
        _as_uint64(effective_dimension, "dimension"),
        _as_uint64(degree, "degree"),
        _as_log_sig_method(method),
        use_disk,
    )
    check_status(status, :prepare_log_sig)
    return nothing
end

function clear_log_sig_cache(; use_disk::Bool=false)
    status = ccall((:clear_cache, libcpsig), Cint, (Bool,), use_disk)
    check_status(status, :clear_cache)
    return nothing
end

"""
    branched_sig_length(dimension, max_nodes; time_aug=false, lead_lag=false,
        planar=false, scalar_term=false)

Return the length of the truncated branched signature for `dimension` channels
and decorated trees with at most `max_nodes` nodes. By default this matches
pySigLib and excludes the leading scalar term.
"""
function branched_sig_length(
    dimension::Integer,
    max_nodes::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    planar::Bool=false,
    scalar_term::Bool=false,
)
    max_nodes >= 0 || throw(ArgumentError("max_nodes must be nonnegative"))
    effective_dimension = _effective_dimension(dimension, time_aug, lead_lag)

    full_length = ccall(
        (:branched_sig_length, libcpsig),
        UInt64,
        (UInt64, UInt64, Bool),
        _as_uint64(effective_dimension, "dimension"),
        _as_uint64(max_nodes, "max_nodes"),
        planar,
    )

    full_length != 0 || throw(OverflowError("cpsig branched signature length overflowed"))
    full_length <= UInt64(typemax(Int)) ||
        throw(OverflowError("branched signature length exceeds Int"))

    return Int(full_length) - (scalar_term ? 0 : 1)
end

"""
    prepare_branched_sig(dimension, max_nodes; time_aug=false, lead_lag=false,
        planar=false, use_disk=false)

Prepare cpsig's cache for branched signature computations. High-level branched
signature functions call this automatically, but explicit preparation can be
useful when pre-warming a cache.
"""
function prepare_branched_sig(
    dimension::Integer,
    max_nodes::Integer;
    time_aug::Bool=false,
    lead_lag::Bool=false,
    planar::Bool=false,
    use_disk::Bool=false,
)
    max_nodes >= 0 || throw(ArgumentError("max_nodes must be nonnegative"))
    effective_dimension = _effective_dimension(dimension, time_aug, lead_lag)
    status = ccall(
        (:prepare_branched_sig, libcpsig),
        Cint,
        (UInt64, UInt64, Bool, Bool),
        _as_uint64(effective_dimension, "dimension"),
        _as_uint64(max_nodes, "max_nodes"),
        use_disk,
        planar,
    )
    check_status(status, :prepare_branched_sig)
    return nothing
end

function _signature_c!(
    out::Matrix{Float64},
    path::Array{Float64,3},
    batch_size::Integer,
    dimension::Integer,
    len::Integer,
    degree::Integer,
    time_aug::Bool,
    lead_lag::Bool,
    horner::Bool,
    scalar_term::Bool,
    n_jobs::Integer,
    end_time::Real,
)
    status = ccall(
        (:signature_d, libcpsig),
        Cint,
        (
            Ptr{Cdouble}, Ptr{Cdouble},
            UInt64, UInt64, UInt64, UInt64,
            Bool, Bool, Cdouble, Bool, Bool, Cint,
            Ptr{Cdouble}, UInt64, UInt64, UInt64,
        ),
        path, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(len, "length"),
        _as_uint64(degree, "degree"),
        time_aug, lead_lag, Cdouble(end_time), horner, scalar_term, _as_cint_n_jobs(n_jobs),
        Ptr{Cdouble}(C_NULL), UInt64(0), UInt64(0), UInt64(0),
    )
    check_status(status, :signature_d)
    return out
end

function _signature_c!(
    out::Matrix{Float32},
    path::Array{Float32,3},
    batch_size::Integer,
    dimension::Integer,
    len::Integer,
    degree::Integer,
    time_aug::Bool,
    lead_lag::Bool,
    horner::Bool,
    scalar_term::Bool,
    n_jobs::Integer,
    end_time::Real,
)
    status = ccall(
        (:signature_f, libcpsig),
        Cint,
        (
            Ptr{Cfloat}, Ptr{Cfloat},
            UInt64, UInt64, UInt64, UInt64,
            Bool, Bool, Cfloat, Bool, Bool, Cint,
            Ptr{Cfloat}, UInt64, UInt64, UInt64,
        ),
        path, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(len, "length"),
        _as_uint64(degree, "degree"),
        time_aug, lead_lag, Cfloat(end_time), horner, scalar_term, _as_cint_n_jobs(n_jobs),
        Ptr{Cfloat}(C_NULL), UInt64(0), UInt64(0), UInt64(0),
    )
    check_status(status, :signature_f)
    return out
end

function _transform_path_c!(
    out::Array{Float64,3},
    path::Array{Float64,3},
    batch_size::Integer,
    dimension::Integer,
    len::Integer,
    time_aug::Bool,
    lead_lag::Bool,
    end_time::Real,
    n_jobs::Integer,
)
    status = ccall(
        (:transform_path_d, libcpsig),
        Cint,
        (Ptr{Cdouble}, Ptr{Cdouble}, UInt64, UInt64, UInt64, Bool, Bool, Cdouble, Cint),
        path, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(len, "length"),
        time_aug, lead_lag, Cdouble(end_time), _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :transform_path_d)
    return out
end

function _transform_path_c!(
    out::Array{Float32,3},
    path::Array{Float32,3},
    batch_size::Integer,
    dimension::Integer,
    len::Integer,
    time_aug::Bool,
    lead_lag::Bool,
    end_time::Real,
    n_jobs::Integer,
)
    status = ccall(
        (:transform_path_f, libcpsig),
        Cint,
        (Ptr{Cfloat}, Ptr{Cfloat}, UInt64, UInt64, UInt64, Bool, Bool, Cfloat, Cint),
        path, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(len, "length"),
        time_aug, lead_lag, Cfloat(end_time), _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :transform_path_f)
    return out
end

function _transform_path_backprop_c!(
    out::Array{Float64,3},
    derivs::Array{Float64,3},
    batch_size::Integer,
    dimension::Integer,
    len::Integer,
    time_aug::Bool,
    lead_lag::Bool,
    end_time::Real,
    n_jobs::Integer,
)
    status = ccall(
        (:transform_path_backprop_d, libcpsig),
        Cint,
        (Ptr{Cdouble}, Ptr{Cdouble}, UInt64, UInt64, UInt64, Bool, Bool, Cdouble, Cint),
        derivs, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(len, "length"),
        time_aug, lead_lag, Cdouble(end_time), _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :transform_path_backprop_d)
    return out
end

function _transform_path_backprop_c!(
    out::Array{Float32,3},
    derivs::Array{Float32,3},
    batch_size::Integer,
    dimension::Integer,
    len::Integer,
    time_aug::Bool,
    lead_lag::Bool,
    end_time::Real,
    n_jobs::Integer,
)
    status = ccall(
        (:transform_path_backprop_f, libcpsig),
        Cint,
        (Ptr{Cfloat}, Ptr{Cfloat}, UInt64, UInt64, UInt64, Bool, Bool, Cfloat, Cint),
        derivs, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(len, "length"),
        time_aug, lead_lag, Cfloat(end_time), _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :transform_path_backprop_f)
    return out
end

function _sig_to_log_sig_c!(
    out::Matrix{Float64},
    sig_value::Matrix{Float64},
    batch_size::Integer,
    dimension::Integer,
    degree::Integer,
    time_aug::Bool,
    lead_lag::Bool,
    method::Integer,
    scalar_term::Bool,
    n_jobs::Integer,
)
    status = ccall(
        (:sig_to_log_sig_d, libcpsig),
        Cint,
        (Ptr{Cdouble}, Ptr{Cdouble}, UInt64, UInt64, UInt64, Bool, Bool, Cint, Bool, Cint),
        sig_value, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(degree, "degree"),
        time_aug, lead_lag, _as_cached_log_sig_method(method), scalar_term, _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :sig_to_log_sig_d)
    return out
end

function _sig_to_log_sig_c!(
    out::Matrix{Float32},
    sig_value::Matrix{Float32},
    batch_size::Integer,
    dimension::Integer,
    degree::Integer,
    time_aug::Bool,
    lead_lag::Bool,
    method::Integer,
    scalar_term::Bool,
    n_jobs::Integer,
)
    status = ccall(
        (:sig_to_log_sig_f, libcpsig),
        Cint,
        (Ptr{Cfloat}, Ptr{Cfloat}, UInt64, UInt64, UInt64, Bool, Bool, Cint, Bool, Cint),
        sig_value, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(degree, "degree"),
        time_aug, lead_lag, _as_cached_log_sig_method(method), scalar_term, _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :sig_to_log_sig_f)
    return out
end

function _sig_to_log_sig_backprop_c!(
    out::Matrix{Float64},
    sig_value::Matrix{Float64},
    log_sig_derivs::Matrix{Float64},
    batch_size::Integer,
    dimension::Integer,
    degree::Integer,
    time_aug::Bool,
    lead_lag::Bool,
    method::Integer,
    scalar_term::Bool,
    n_jobs::Integer,
)
    status = ccall(
        (:sig_to_log_sig_backprop_d, libcpsig),
        Cint,
        (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, UInt64, UInt64, UInt64, Bool, Bool, Cint, Bool, Cint),
        sig_value, out, log_sig_derivs,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(degree, "degree"),
        time_aug, lead_lag, _as_cached_log_sig_method(method), scalar_term, _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :sig_to_log_sig_backprop_d)
    return out
end

function _sig_to_log_sig_backprop_c!(
    out::Matrix{Float32},
    sig_value::Matrix{Float32},
    log_sig_derivs::Matrix{Float32},
    batch_size::Integer,
    dimension::Integer,
    degree::Integer,
    time_aug::Bool,
    lead_lag::Bool,
    method::Integer,
    scalar_term::Bool,
    n_jobs::Integer,
)
    status = ccall(
        (:sig_to_log_sig_backprop_f, libcpsig),
        Cint,
        (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, UInt64, UInt64, UInt64, Bool, Bool, Cint, Bool, Cint),
        sig_value, out, log_sig_derivs,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(degree, "degree"),
        time_aug, lead_lag, _as_cached_log_sig_method(method), scalar_term, _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :sig_to_log_sig_backprop_f)
    return out
end

function _logsig_to_sig_c!(
    out::Matrix{Float64},
    log_sig_value::Matrix{Float64},
    batch_size::Integer,
    dimension::Integer,
    degree::Integer,
    time_aug::Bool,
    lead_lag::Bool,
    method::Integer,
    scalar_term::Bool,
    n_jobs::Integer,
)
    status = ccall(
        (:logsig_to_sig_d, libcpsig),
        Cint,
        (Ptr{Cdouble}, Ptr{Cdouble}, UInt64, UInt64, UInt64, Bool, Bool, Cint, Bool, Cint),
        log_sig_value, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(degree, "degree"),
        time_aug, lead_lag, _as_cached_log_sig_method(method), scalar_term, _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :logsig_to_sig_d)
    return out
end

function _logsig_to_sig_c!(
    out::Matrix{Float32},
    log_sig_value::Matrix{Float32},
    batch_size::Integer,
    dimension::Integer,
    degree::Integer,
    time_aug::Bool,
    lead_lag::Bool,
    method::Integer,
    scalar_term::Bool,
    n_jobs::Integer,
)
    status = ccall(
        (:logsig_to_sig_f, libcpsig),
        Cint,
        (Ptr{Cfloat}, Ptr{Cfloat}, UInt64, UInt64, UInt64, Bool, Bool, Cint, Bool, Cint),
        log_sig_value, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(degree, "degree"),
        time_aug, lead_lag, _as_cached_log_sig_method(method), scalar_term, _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :logsig_to_sig_f)
    return out
end

function _logsig_to_sig_backprop_c!(
    out::Matrix{Float64},
    log_sig_value::Matrix{Float64},
    sig_derivs::Matrix{Float64},
    batch_size::Integer,
    dimension::Integer,
    degree::Integer,
    time_aug::Bool,
    lead_lag::Bool,
    method::Integer,
    scalar_term::Bool,
    n_jobs::Integer,
)
    status = ccall(
        (:logsig_to_sig_backprop_d, libcpsig),
        Cint,
        (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, UInt64, UInt64, UInt64, Bool, Bool, Cint, Bool, Cint),
        log_sig_value, out, sig_derivs,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(degree, "degree"),
        time_aug, lead_lag, _as_cached_log_sig_method(method), scalar_term, _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :logsig_to_sig_backprop_d)
    return out
end

function _logsig_to_sig_backprop_c!(
    out::Matrix{Float32},
    log_sig_value::Matrix{Float32},
    sig_derivs::Matrix{Float32},
    batch_size::Integer,
    dimension::Integer,
    degree::Integer,
    time_aug::Bool,
    lead_lag::Bool,
    method::Integer,
    scalar_term::Bool,
    n_jobs::Integer,
)
    status = ccall(
        (:logsig_to_sig_backprop_f, libcpsig),
        Cint,
        (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, UInt64, UInt64, UInt64, Bool, Bool, Cint, Bool, Cint),
        log_sig_value, out, sig_derivs,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(degree, "degree"),
        time_aug, lead_lag, _as_cached_log_sig_method(method), scalar_term, _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :logsig_to_sig_backprop_f)
    return out
end

function _log_sig_from_path_c!(
    out::Matrix{Float64},
    path::Array{Float64,3},
    batch_size::Integer,
    len::Integer,
    dimension::Integer,
    degree::Integer,
    n_jobs::Integer,
)
    status = ccall(
        (:log_sig_from_path_d, libcpsig),
        Cint,
        (Ptr{Cdouble}, Ptr{Cdouble}, UInt64, UInt64, UInt64, UInt64, Cint),
        path, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(len, "length"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(degree, "degree"),
        _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :log_sig_from_path_d)
    return out
end

function _log_sig_from_path_c!(
    out::Matrix{Float32},
    path::Array{Float32,3},
    batch_size::Integer,
    len::Integer,
    dimension::Integer,
    degree::Integer,
    n_jobs::Integer,
)
    status = ccall(
        (:log_sig_from_path_f, libcpsig),
        Cint,
        (Ptr{Cfloat}, Ptr{Cfloat}, UInt64, UInt64, UInt64, UInt64, Cint),
        path, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(len, "length"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(degree, "degree"),
        _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :log_sig_from_path_f)
    return out
end

function _log_sig_from_path_backprop_c!(
    out::Array{Float64,3},
    log_sig_derivs::Matrix{Float64},
    path::Array{Float64,3},
    batch_size::Integer,
    len::Integer,
    dimension::Integer,
    degree::Integer,
    n_jobs::Integer,
)
    status = ccall(
        (:log_sig_from_path_backprop_d, libcpsig),
        Cint,
        (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, UInt64, UInt64, UInt64, UInt64, Cint),
        log_sig_derivs, out, path,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(len, "length"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(degree, "degree"),
        _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :log_sig_from_path_backprop_d)
    return out
end

function _log_sig_from_path_backprop_c!(
    out::Array{Float32,3},
    log_sig_derivs::Matrix{Float32},
    path::Array{Float32,3},
    batch_size::Integer,
    len::Integer,
    dimension::Integer,
    degree::Integer,
    n_jobs::Integer,
)
    status = ccall(
        (:log_sig_from_path_backprop_f, libcpsig),
        Cint,
        (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, UInt64, UInt64, UInt64, UInt64, Cint),
        log_sig_derivs, out, path,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(len, "length"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(degree, "degree"),
        _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :log_sig_from_path_backprop_f)
    return out
end

function _log_sig_combine_c!(
    out::Matrix{Float64},
    log_sig1::Matrix{Float64},
    log_sig2::Matrix{Float64},
    batch_size::Integer,
    dimension::Integer,
    degree::Integer,
    n_jobs::Integer,
)
    status = ccall(
        (:log_sig_combine_d, libcpsig),
        Cint,
        (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, UInt64, UInt64, UInt64, Cint),
        log_sig1, log_sig2, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(degree, "degree"),
        _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :log_sig_combine_d)
    return out
end

function _log_sig_combine_c!(
    out::Matrix{Float32},
    log_sig1::Matrix{Float32},
    log_sig2::Matrix{Float32},
    batch_size::Integer,
    dimension::Integer,
    degree::Integer,
    n_jobs::Integer,
)
    status = ccall(
        (:log_sig_combine_f, libcpsig),
        Cint,
        (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, UInt64, UInt64, UInt64, Cint),
        log_sig1, log_sig2, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(degree, "degree"),
        _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :log_sig_combine_f)
    return out
end

function _log_sig_combine_backprop_c!(
    out1::Matrix{Float64},
    out2::Matrix{Float64},
    derivs::Matrix{Float64},
    log_sig1::Matrix{Float64},
    log_sig2::Matrix{Float64},
    batch_size::Integer,
    dimension::Integer,
    degree::Integer,
    n_jobs::Integer,
)
    status = ccall(
        (:log_sig_combine_backprop_d, libcpsig),
        Cint,
        (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, UInt64, UInt64, UInt64, Cint),
        derivs, out1, out2, log_sig1, log_sig2,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(degree, "degree"),
        _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :log_sig_combine_backprop_d)
    return out1, out2
end

function _log_sig_combine_backprop_c!(
    out1::Matrix{Float32},
    out2::Matrix{Float32},
    derivs::Matrix{Float32},
    log_sig1::Matrix{Float32},
    log_sig2::Matrix{Float32},
    batch_size::Integer,
    dimension::Integer,
    degree::Integer,
    n_jobs::Integer,
)
    status = ccall(
        (:log_sig_combine_backprop_f, libcpsig),
        Cint,
        (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, UInt64, UInt64, UInt64, Cint),
        derivs, out1, out2, log_sig1, log_sig2,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(degree, "degree"),
        _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :log_sig_combine_backprop_f)
    return out1, out2
end

function _log_sig_join_c!(
    out::Matrix{Float64},
    log_sig_value::Matrix{Float64},
    displacement::Matrix{Float64},
    batch_size::Integer,
    dimension::Integer,
    degree::Integer,
    n_jobs::Integer,
)
    status = ccall(
        (:log_sig_join_d, libcpsig),
        Cint,
        (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, UInt64, UInt64, UInt64, Cint),
        log_sig_value, displacement, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(degree, "degree"),
        _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :log_sig_join_d)
    return out
end

function _log_sig_join_c!(
    out::Matrix{Float32},
    log_sig_value::Matrix{Float32},
    displacement::Matrix{Float32},
    batch_size::Integer,
    dimension::Integer,
    degree::Integer,
    n_jobs::Integer,
)
    status = ccall(
        (:log_sig_join_f, libcpsig),
        Cint,
        (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, UInt64, UInt64, UInt64, Cint),
        log_sig_value, displacement, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(degree, "degree"),
        _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :log_sig_join_f)
    return out
end

function _log_sig_join_backprop_c!(
    out_log_sig::Matrix{Float64},
    out_displacement::Matrix{Float64},
    derivs::Matrix{Float64},
    log_sig_value::Matrix{Float64},
    displacement::Matrix{Float64},
    batch_size::Integer,
    dimension::Integer,
    degree::Integer,
    n_jobs::Integer,
)
    status = ccall(
        (:log_sig_join_backprop_d, libcpsig),
        Cint,
        (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, UInt64, UInt64, UInt64, Cint),
        derivs, out_log_sig, out_displacement, log_sig_value, displacement,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(degree, "degree"),
        _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :log_sig_join_backprop_d)
    return out_log_sig, out_displacement
end

function _log_sig_join_backprop_c!(
    out_log_sig::Matrix{Float32},
    out_displacement::Matrix{Float32},
    derivs::Matrix{Float32},
    log_sig_value::Matrix{Float32},
    displacement::Matrix{Float32},
    batch_size::Integer,
    dimension::Integer,
    degree::Integer,
    n_jobs::Integer,
)
    status = ccall(
        (:log_sig_join_backprop_f, libcpsig),
        Cint,
        (Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, UInt64, UInt64, UInt64, Cint),
        derivs, out_log_sig, out_displacement, log_sig_value, displacement,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(degree, "degree"),
        _as_cint_n_jobs(n_jobs),
    )
    check_status(status, :log_sig_join_backprop_f)
    return out_log_sig, out_displacement
end

function _branched_sig_c!(
    out::Matrix{Float64},
    path::Array{Float64,3},
    batch_size::Integer,
    dimension::Integer,
    len::Integer,
    max_nodes::Integer,
    time_aug::Bool,
    lead_lag::Bool,
    planar::Bool,
    scalar_term::Bool,
    n_jobs::Integer,
    end_time::Real,
)
    status = ccall(
        (:branched_sig_d, libcpsig),
        Cint,
        (
            Ptr{Cdouble}, Ptr{Cdouble},
            UInt64, UInt64, UInt64, UInt64,
            Cint, Bool, Bool, Cdouble, Bool, Bool,
            Ptr{Cdouble}, UInt64, UInt64, UInt64,
        ),
        path, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(len, "length"),
        _as_uint64(max_nodes, "max_nodes"),
        _as_cint_n_jobs(n_jobs), time_aug, lead_lag, Cdouble(end_time), planar, scalar_term,
        Ptr{Cdouble}(C_NULL), UInt64(0), UInt64(0), UInt64(0),
    )
    check_status(status, :branched_sig_d)
    return out
end

function _branched_sig_c!(
    out::Matrix{Float32},
    path::Array{Float32,3},
    batch_size::Integer,
    dimension::Integer,
    len::Integer,
    max_nodes::Integer,
    time_aug::Bool,
    lead_lag::Bool,
    planar::Bool,
    scalar_term::Bool,
    n_jobs::Integer,
    end_time::Real,
)
    status = ccall(
        (:branched_sig_f, libcpsig),
        Cint,
        (
            Ptr{Cfloat}, Ptr{Cfloat},
            UInt64, UInt64, UInt64, UInt64,
            Cint, Bool, Bool, Cfloat, Bool, Bool,
            Ptr{Cfloat}, UInt64, UInt64, UInt64,
        ),
        path, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(len, "length"),
        _as_uint64(max_nodes, "max_nodes"),
        _as_cint_n_jobs(n_jobs), time_aug, lead_lag, Cfloat(end_time), planar, scalar_term,
        Ptr{Cfloat}(C_NULL), UInt64(0), UInt64(0), UInt64(0),
    )
    check_status(status, :branched_sig_f)
    return out
end

function _branched_sig_combine_c!(
    out::Matrix{Float64},
    bsig1::Matrix{Float64},
    bsig2::Matrix{Float64},
    batch_size::Integer,
    dimension::Integer,
    max_nodes::Integer,
    planar::Bool,
    scalar_term::Bool,
    n_jobs::Integer,
)
    status = ccall(
        (:branched_sig_combine_d, libcpsig),
        Cint,
        (
            Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
            UInt64, UInt64, UInt64, Cint, Bool, Bool,
        ),
        bsig1, bsig2, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(max_nodes, "max_nodes"),
        _as_cint_n_jobs(n_jobs), planar, scalar_term,
    )
    check_status(status, :branched_sig_combine_d)
    return out
end

function _branched_sig_combine_c!(
    out::Matrix{Float32},
    bsig1::Matrix{Float32},
    bsig2::Matrix{Float32},
    batch_size::Integer,
    dimension::Integer,
    max_nodes::Integer,
    planar::Bool,
    scalar_term::Bool,
    n_jobs::Integer,
)
    status = ccall(
        (:branched_sig_combine_f, libcpsig),
        Cint,
        (
            Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},
            UInt64, UInt64, UInt64, Cint, Bool, Bool,
        ),
        bsig1, bsig2, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(max_nodes, "max_nodes"),
        _as_cint_n_jobs(n_jobs), planar, scalar_term,
    )
    check_status(status, :branched_sig_combine_f)
    return out
end

function _branched_sig_to_log_sig_c!(
    out::Matrix{Float64},
    bsig::Matrix{Float64},
    batch_size::Integer,
    dimension::Integer,
    max_nodes::Integer,
    planar::Bool,
    scalar_term::Bool,
    n_jobs::Integer,
)
    status = ccall(
        (:branched_sig_to_log_sig_d, libcpsig),
        Cint,
        (
            Ptr{Cdouble}, Ptr{Cdouble},
            UInt64, UInt64, UInt64, Cint, Bool, Bool,
        ),
        bsig, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(max_nodes, "max_nodes"),
        _as_cint_n_jobs(n_jobs), planar, scalar_term,
    )
    check_status(status, :branched_sig_to_log_sig_d)
    return out
end

function _branched_sig_to_log_sig_c!(
    out::Matrix{Float32},
    bsig::Matrix{Float32},
    batch_size::Integer,
    dimension::Integer,
    max_nodes::Integer,
    planar::Bool,
    scalar_term::Bool,
    n_jobs::Integer,
)
    status = ccall(
        (:branched_sig_to_log_sig_f, libcpsig),
        Cint,
        (
            Ptr{Cfloat}, Ptr{Cfloat},
            UInt64, UInt64, UInt64, Cint, Bool, Bool,
        ),
        bsig, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(max_nodes, "max_nodes"),
        _as_cint_n_jobs(n_jobs), planar, scalar_term,
    )
    check_status(status, :branched_sig_to_log_sig_f)
    return out
end

function _sig_backprop_c!(
    out::Array{Float64,3},
    path::Array{Float64,3},
    sig_derivs::Matrix{Float64},
    sig_value::Matrix{Float64},
    batch_size::Integer,
    dimension::Integer,
    len::Integer,
    degree::Integer,
    time_aug::Bool,
    lead_lag::Bool,
    scalar_term::Bool,
    n_jobs::Integer,
    end_time::Real,
)
    status = ccall(
        (:sig_backprop_d, libcpsig),
        Cint,
        (
            Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
            UInt64, UInt64, UInt64, UInt64,
            Bool, Bool, Cdouble, Bool, Cint,
            Ptr{Cdouble}, UInt64, UInt64, UInt64,
        ),
        path, out, sig_derivs, sig_value,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(len, "length"),
        _as_uint64(degree, "degree"),
        time_aug, lead_lag, Cdouble(end_time), scalar_term, _as_cint_n_jobs(n_jobs),
        Ptr{Cdouble}(C_NULL), UInt64(0), UInt64(0), UInt64(0),
    )
    check_status(status, :sig_backprop_d)
    return out
end

function _sig_backprop_c!(
    out::Array{Float32,3},
    path::Array{Float32,3},
    sig_derivs::Matrix{Float32},
    sig_value::Matrix{Float32},
    batch_size::Integer,
    dimension::Integer,
    len::Integer,
    degree::Integer,
    time_aug::Bool,
    lead_lag::Bool,
    scalar_term::Bool,
    n_jobs::Integer,
    end_time::Real,
)
    status = ccall(
        (:sig_backprop_f, libcpsig),
        Cint,
        (
            Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},
            UInt64, UInt64, UInt64, UInt64,
            Bool, Bool, Cfloat, Bool, Cint,
            Ptr{Cfloat}, UInt64, UInt64, UInt64,
        ),
        path, out, sig_derivs, sig_value,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(len, "length"),
        _as_uint64(degree, "degree"),
        time_aug, lead_lag, Cfloat(end_time), scalar_term, _as_cint_n_jobs(n_jobs),
        Ptr{Cfloat}(C_NULL), UInt64(0), UInt64(0), UInt64(0),
    )
    check_status(status, :sig_backprop_f)
    return out
end

function _branched_sig_backprop_c!(
    out::Array{Float64,3},
    path::Array{Float64,3},
    bsig_derivs::Matrix{Float64},
    bsig_value::Matrix{Float64},
    batch_size::Integer,
    dimension::Integer,
    len::Integer,
    max_nodes::Integer,
    time_aug::Bool,
    lead_lag::Bool,
    planar::Bool,
    scalar_term::Bool,
    n_jobs::Integer,
    end_time::Real,
)
    status = ccall(
        (:branched_sig_backprop_d, libcpsig),
        Cint,
        (
            Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
            UInt64, UInt64, UInt64, UInt64,
            Cint, Bool, Bool, Cdouble, Bool, Bool,
            Ptr{Cdouble}, UInt64, UInt64, UInt64,
        ),
        path, out, bsig_derivs, bsig_value,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(len, "length"),
        _as_uint64(max_nodes, "max_nodes"),
        _as_cint_n_jobs(n_jobs), time_aug, lead_lag, Cdouble(end_time), planar, scalar_term,
        Ptr{Cdouble}(C_NULL), UInt64(0), UInt64(0), UInt64(0),
    )
    check_status(status, :branched_sig_backprop_d)
    return out
end

function _branched_sig_backprop_c!(
    out::Array{Float32,3},
    path::Array{Float32,3},
    bsig_derivs::Matrix{Float32},
    bsig_value::Matrix{Float32},
    batch_size::Integer,
    dimension::Integer,
    len::Integer,
    max_nodes::Integer,
    time_aug::Bool,
    lead_lag::Bool,
    planar::Bool,
    scalar_term::Bool,
    n_jobs::Integer,
    end_time::Real,
)
    status = ccall(
        (:branched_sig_backprop_f, libcpsig),
        Cint,
        (
            Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},
            UInt64, UInt64, UInt64, UInt64,
            Cint, Bool, Bool, Cfloat, Bool, Bool,
            Ptr{Cfloat}, UInt64, UInt64, UInt64,
        ),
        path, out, bsig_derivs, bsig_value,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(len, "length"),
        _as_uint64(max_nodes, "max_nodes"),
        _as_cint_n_jobs(n_jobs), time_aug, lead_lag, Cfloat(end_time), planar, scalar_term,
        Ptr{Cfloat}(C_NULL), UInt64(0), UInt64(0), UInt64(0),
    )
    check_status(status, :branched_sig_backprop_f)
    return out
end

function _branched_sig_to_log_sig_backprop_c!(
    out::Matrix{Float64},
    bsig::Matrix{Float64},
    derivs::Matrix{Float64},
    batch_size::Integer,
    dimension::Integer,
    max_nodes::Integer,
    planar::Bool,
    scalar_term::Bool,
    n_jobs::Integer,
)
    status = ccall(
        (:branched_sig_to_log_sig_backprop_d, libcpsig),
        Cint,
        (
            Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
            UInt64, UInt64, UInt64, Cint, Bool, Bool,
        ),
        bsig, derivs, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(max_nodes, "max_nodes"),
        _as_cint_n_jobs(n_jobs), planar, scalar_term,
    )
    check_status(status, :branched_sig_to_log_sig_backprop_d)
    return out
end

function _branched_sig_to_log_sig_backprop_c!(
    out::Matrix{Float32},
    bsig::Matrix{Float32},
    derivs::Matrix{Float32},
    batch_size::Integer,
    dimension::Integer,
    max_nodes::Integer,
    planar::Bool,
    scalar_term::Bool,
    n_jobs::Integer,
)
    status = ccall(
        (:branched_sig_to_log_sig_backprop_f, libcpsig),
        Cint,
        (
            Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},
            UInt64, UInt64, UInt64, Cint, Bool, Bool,
        ),
        bsig, derivs, out,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(max_nodes, "max_nodes"),
        _as_cint_n_jobs(n_jobs), planar, scalar_term,
    )
    check_status(status, :branched_sig_to_log_sig_backprop_f)
    return out
end

function _branched_sig_combine_backprop_c!(
    out1::Matrix{Float64},
    out2::Matrix{Float64},
    bsig1::Matrix{Float64},
    bsig2::Matrix{Float64},
    derivs::Matrix{Float64},
    batch_size::Integer,
    dimension::Integer,
    max_nodes::Integer,
    planar::Bool,
    scalar_term::Bool,
    n_jobs::Integer,
)
    status = ccall(
        (:branched_sig_combine_backprop_d, libcpsig),
        Cint,
        (
            Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble},
            UInt64, UInt64, UInt64, Cint, Bool, Bool,
        ),
        bsig1, bsig2, derivs, out1, out2,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(max_nodes, "max_nodes"),
        _as_cint_n_jobs(n_jobs), planar, scalar_term,
    )
    check_status(status, :branched_sig_combine_backprop_d)
    return out1, out2
end

function _branched_sig_combine_backprop_c!(
    out1::Matrix{Float32},
    out2::Matrix{Float32},
    bsig1::Matrix{Float32},
    bsig2::Matrix{Float32},
    derivs::Matrix{Float32},
    batch_size::Integer,
    dimension::Integer,
    max_nodes::Integer,
    planar::Bool,
    scalar_term::Bool,
    n_jobs::Integer,
)
    status = ccall(
        (:branched_sig_combine_backprop_f, libcpsig),
        Cint,
        (
            Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat},
            UInt64, UInt64, UInt64, Cint, Bool, Bool,
        ),
        bsig1, bsig2, derivs, out1, out2,
        _as_uint64(batch_size, "batch_size"),
        _as_uint64(dimension, "dimension"),
        _as_uint64(max_nodes, "max_nodes"),
        _as_cint_n_jobs(n_jobs), planar, scalar_term,
    )
    check_status(status, :branched_sig_combine_backprop_f)
    return out1, out2
end
