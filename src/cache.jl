const _cache_dir_configured = Ref(false)

# CUDA supplies the same cache operations through its optional extension.
_cache_call(::Val{:cuda}, args...) = throw(ArgumentError("load CUDA.jl to use device=:cuda"))
_cache_call(::Val, args...) = throw(ArgumentError("device must be :cpu, :cuda, or :both"))
function _cache_call(::Val{:both}, args...)
    _cache_call(Val(:cpu), args...)
    _cache_call(Val(:cuda), args...)
    return nothing
end

function _cache_call(::Val{:cpu}, ::Val{:set_cache_dir}, path)
    check_status(ccall(_cpsig_symbol(:set_cache_dir), Cint, (Cstring,), path), :set_cache_dir)
    _cache_dir_configured[] = true
end
function _cache_call(::Val{:cpu}, ::Val{:clear_cache}, use_disk)
    _prepare_cache_directory(use_disk)
    check_status(ccall(_cpsig_symbol(:clear_cache), Cint, (Bool,), use_disk), :clear_cache)
end
function _cache_call(::Val{:cpu}, ::Val{:prepare_log_sig}, d, n, m, use_disk)
    # 3.1's direct method uses implicit BCH preparation; method 3 corrupts its basis cache.
    !_modern_abi[] && m == 3 && (m = Cint(2))
    _prepare_cache_directory(use_disk)
    check_status(ccall(_cpsig_symbol(:prepare_log_sig), Cint, (UInt64, UInt64, Cint, Bool), d, n, m, use_disk), :prepare_log_sig)
end
function _cache_call(::Val{:cpu}, ::Val{:prepare_branched_sig}, d, n, planar, use_disk)
    _prepare_cache_directory(use_disk)
    check_status(ccall(_cpsig_symbol(:prepare_branched_sig), Cint, (UInt64, UInt64, Bool, Bool), d, n, use_disk, planar), :prepare_branched_sig)
end
function _cache_call(::Val{:cpu}, ::Val{:prepare_branched_log_sig}, d, n, m, planar, use_disk)
    if _modern_abi[]
        check_status(ccall(_cpsig_symbol(:prepare_branched_log_sig), Cint, (UInt64, UInt64, Cint, Bool, Bool), d, n, m, use_disk, planar), :prepare_branched_log_sig)
    else
        d != 0 && _cache_call(Val(:cpu), Val(:prepare_branched_sig), d, n, planar, use_disk)
    end
end

for name in (:prepare_log_sig, :prepare_branched_sig, :prepare_branched_log_sig)
    @eval function $(Symbol(:_, name, :_for))(like::AbstractArray, args...; kwargs...)
        return _with_array(like) do
            $name(args...; device=_backend(like), kwargs...)
        end
    end
end

"""
    set_cache_dir(directory; device=:cpu)

Create and select the parent directory for native disk caches. Cache operations are synchronized with computations made through SigLib. `device` selects `:cpu`, `:cuda`, or `:both`; CUDA operations use the active device and require CUDA.jl.
"""
function set_cache_dir(directory::AbstractString; device::Symbol=:cpu)
    occursin('\0', directory) && throw(ArgumentError("cache directory contains a NUL character"))
    lock(_cache_lock) do
        path = abspath(directory)
        mkpath(path)
        _cache_call(Val(device), Val(:set_cache_dir), path)
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
    prepare_log_sig(dimension, degree, method; time_aug=false, lead_lag=false, use_disk=false, device=:cpu)

Prepare the native log-signature cache. Methods 0–3 select tensor, extracted Lyndon coordinates, Lyndon basis coordinates, and direct BCH computations respectively. High-level functions prepare their caches automatically on the input device.
"""
function prepare_log_sig(
    dimension::Integer, degree::Integer, method::Integer;
    time_aug::Bool=false, lead_lag::Bool=false, use_disk::Bool=false, device::Symbol=:cpu,
)
    d = _as_uint64(_effective_dimension(dimension, time_aug, lead_lag), "dimension")
    n = _as_uint64(degree, "degree")
    m = _as_log_sig_method(method)
    (m == 0 || n == 0) && return nothing
    log_sig_length(d, n)
    lock(_cache_lock) do
        _cache_call(Val(device), Val(:prepare_log_sig), d, n, m, use_disk)
    end
    return nothing
end

"""
    prepare_branched_log_sig(dimension, degree, method; time_aug=false, lead_lag=false, planar=false, use_disk=false, device=:cpu)

Prepare branched-log-signature caches. Methods 1–3 require `planar=true` and a 4.x native artifact. High-level functions prepare automatically on the input device.
"""
function prepare_branched_log_sig(
    dimension::Integer, degree::Integer, method::Integer;
    time_aug::Bool=false, lead_lag::Bool=false, planar::Bool=false, use_disk::Bool=false, device::Symbol=:cpu,
)
    m = _branched_log_sig_method(method, planar)
    d = _branched_log_sig_dimension(dimension, time_aug, lead_lag)
    n = _as_uint64(degree, "degree")
    d != 0 && branched_sig_length(d, n; planar=planar)
    lock(_cache_lock) do
        _cache_call(Val(device), Val(:prepare_branched_log_sig), d, n, m, planar, use_disk)
    end
    return nothing
end

"""
    clear_log_sig_cache(; use_disk=false, device=:cpu)

Clear native caches, waiting for active cache-dependent computations. With `use_disk=true`, also remove native disk cache files under the configured cache directory.
"""
function clear_log_sig_cache(; use_disk::Bool=false, device::Symbol=:cpu)
    lock(_cache_lock) do
        _cache_call(Val(device), Val(:clear_cache), use_disk)
    end
    return nothing
end

"""
    prepare_branched_sig(dimension, max_nodes; time_aug=false, lead_lag=false, planar=false, use_disk=false, device=:cpu)

Prepare the native branched-signature cache. High-level functions call this automatically on the input device.
"""
function prepare_branched_sig(
    dimension::Integer, max_nodes::Integer;
    time_aug::Bool=false, lead_lag::Bool=false, planar::Bool=false, use_disk::Bool=false, device::Symbol=:cpu,
)
    d = _as_uint64(_effective_dimension(dimension, time_aug, lead_lag), "dimension")
    n = _as_uint64(max_nodes, "max_nodes")
    branched_sig_length(d, n; planar=planar)
    lock(_cache_lock) do
        _cache_call(Val(device), Val(:prepare_branched_sig), d, n, planar, use_disk)
    end
    return nothing
end
