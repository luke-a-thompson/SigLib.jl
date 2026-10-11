const _cache_dir_configured = Ref(false)

"""
    set_cache_dir(directory)

Create and select the parent directory for native disk caches. Cache operations are synchronized with computations made through SigLib.
"""
function set_cache_dir(directory::AbstractString)
    occursin('\0', directory) && throw(ArgumentError("cache directory contains a NUL character"))
    lock(_cache_lock) do
        path = abspath(directory)
        mkpath(path)
        check_status(ccall(_cpsig_symbol(:set_cache_dir), Cint, (Cstring,), path), :set_cache_dir)
        _cache_dir_configured[] = true
    end
    return nothing
end

function _prepare_cache_directory(use_disk::Bool)
    # Older cpSIG creates a directory even for in-memory caches. Keep it temporary.
    if !_modern_abi[] && !_cache_dir_configured[] && !use_disk
        set_cache_dir(mktempdir())
    end
    return nothing
end

"""
    prepare_log_sig(dimension, degree, method; time_aug=false, lead_lag=false, use_disk=false)

Prepare the native log-signature cache. Methods 0–3 select tensor, extracted Lyndon coordinates, Lyndon basis coordinates, and direct BCH computations respectively. High-level functions prepare their caches automatically.
"""
function prepare_log_sig(
    dimension::Integer, degree::Integer, method::Integer;
    time_aug::Bool=false, lead_lag::Bool=false, use_disk::Bool=false,
)
    d = _as_uint64(_effective_dimension(dimension, time_aug, lead_lag), "dimension")
    n = _as_uint64(degree, "degree")
    m = _as_log_sig_method(method)
    (m == 0 || n == 0) && return nothing
    log_sig_length(d, n)
    # 3.1's direct method uses implicit BCH preparation; method 3 would corrupt its basis cache.
    !_modern_abi[] && m == 3 && (m = Cint(2))
    lock(_cache_lock) do
        _prepare_cache_directory(use_disk)
        check_status(ccall(_cpsig_symbol(:prepare_log_sig), Cint, (UInt64, UInt64, Cint, Bool), d, n, m, use_disk), :prepare_log_sig)
    end
    return nothing
end

"""
    prepare_branched_log_sig(dimension, degree, method; time_aug=false, lead_lag=false, planar=false, use_disk=false)

Prepare branched-log-signature caches. Methods 1–3 require `planar=true` and a 4.x native artifact. High-level functions prepare automatically.
"""
function prepare_branched_log_sig(
    dimension::Integer, degree::Integer, method::Integer;
    time_aug::Bool=false, lead_lag::Bool=false, planar::Bool=false, use_disk::Bool=false,
)
    m = _branched_log_sig_method(method, planar)
    d = _branched_log_sig_dimension(dimension, time_aug, lead_lag)
    n = _as_uint64(degree, "degree")
    d != 0 && branched_sig_length(d, n; planar=planar)
    lock(_cache_lock) do
        if _modern_abi[]
            check_status(ccall(_cpsig_symbol(:prepare_branched_log_sig), Cint, (UInt64, UInt64, Cint, Bool, Bool), d, n, m, use_disk, planar), :prepare_branched_log_sig)
        else
            d != 0 && prepare_branched_sig(d, n; planar=planar, use_disk=use_disk)
        end
    end
    return nothing
end

"""
    clear_log_sig_cache(; use_disk=false)

Clear native caches, waiting for active cache-dependent computations. With `use_disk=true`, also remove native disk cache files under the configured cache directory.
"""
function clear_log_sig_cache(; use_disk::Bool=false)
    lock(_cache_lock) do
        _prepare_cache_directory(use_disk)
        check_status(ccall(_cpsig_symbol(:clear_cache), Cint, (Bool,), use_disk), :clear_cache)
    end
    return nothing
end

"""
    prepare_branched_sig(dimension, max_nodes; time_aug=false, lead_lag=false, planar=false, use_disk=false)

Prepare the native branched-signature cache. High-level functions call this automatically.
"""
function prepare_branched_sig(
    dimension::Integer, max_nodes::Integer;
    time_aug::Bool=false, lead_lag::Bool=false, planar::Bool=false, use_disk::Bool=false,
)
    d = _as_uint64(_effective_dimension(dimension, time_aug, lead_lag), "dimension")
    n = _as_uint64(max_nodes, "max_nodes")
    branched_sig_length(d, n; planar=planar)
    lock(_cache_lock) do
        _prepare_cache_directory(use_disk)
        check_status(ccall(_cpsig_symbol(:prepare_branched_sig), Cint, (UInt64, UInt64, Bool, Bool), d, n, use_disk, planar), :prepare_branched_sig)
    end
    return nothing
end
