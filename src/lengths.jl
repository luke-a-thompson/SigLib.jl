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
        _cpsig_symbol(:sig_length),
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

"""
    branched_log_sig_length(dimension, degree; planar=false)

Return the scalar-free branched-log length: decorated rooted trees for nonplanar signatures, or weighted Lyndon forests for compressed planar signatures.
"""
function branched_log_sig_length(dimension::Integer, degree::Integer; planar::Bool=false)
    d = _as_uint64(dimension, "dimension")
    n = _as_uint64(degree, "degree")
    d <= 255 || throw(ArgumentError("branched log signature dimension must be <= 255"))
    (d == 0 || n == 0) && return 0
    if !_modern_abi[]
        planar && throw(ArgumentError("compressed branched log signatures require a pySigLib 4.x native artifact"))
        return branched_sig_length(d, n)
    end
    len = ccall(_cpsig_symbol(:branched_log_sig_length), UInt64, (UInt64, UInt64, Bool), d, n, planar)
    0 < len <= typemax(Int) || throw(OverflowError("branched log signature length overflowed"))
    return Int(len)
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

    d = _as_uint64(effective_dimension, "dimension")
    n = _as_uint64(degree, "degree")
    n == 0 && return 0
    d == 1 && return 1
    # Native Witt-formula intermediates are signed 64-bit powers. Bound allocations too.
    n < 63 || throw(OverflowError("cpsig log signature length overflowed"))
    len = ccall(
        _cpsig_symbol(:log_sig_length),
        UInt64,
        (UInt64, UInt64),
        d, n,
    )

    len != 0 || throw(OverflowError("cpsig log signature length overflowed"))
    len <= UInt64(typemax(Int)) || throw(OverflowError("log signature length exceeds Int"))
    return Int(len)
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
        _cpsig_symbol(:branched_sig_length),
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
