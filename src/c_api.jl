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
