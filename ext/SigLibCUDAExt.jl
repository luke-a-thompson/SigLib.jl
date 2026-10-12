module SigLibCUDAExt

using SigLib, CUDA
using SigLib.LazyArtifacts, SigLib.Libdl

const _libcusig = Ref{Ptr{Cvoid}}(C_NULL)
const _last_error = Ref{Ptr{Cvoid}}(C_NULL)

_context(x::CuArray) = CUDA.context(x)
_context(x::AnyCuArray) = _context(parent(x))
SigLib._backend(::AnyCuArray) = :cuda
SigLib._with_array(f, x::AnyCuArray) = CUDA.context!(f, _context(x))
SigLib._materialize(x::AnyCuArray, ::Type{T}) where {T} = x isa CuArray{T} ? x : CuArray{T}(x)
SigLib._permuted_buffer(x::AnyCuArray, ::Type{T}, dims, axes) where {T} =
    SigLib._permute(reshape(SigLib._materialize(x, T), dims), axes)
function SigLib._check_backend(like::AnyCuArray, x::AnyCuArray)
    _context(like) == _context(x) || throw(ArgumentError("all CUDA inputs must belong to the same context"))
    return nothing
end

function SigLib._validate_branched_log_sig_degree(::AnyCuArray, degree, method)
    method == 3 && degree > 20 && throw(ArgumentError("CUDA branched BCH supports degree at most 20"))
    return nothing
end

function _library()
    if _libcusig[] == C_NULL
        SigLib._modern_abi[] || throw(ArgumentError("CUDA requires matching pySigLib 4.x CPU and CUDA artifacts"))
        manifest = joinpath(pkgdir(SigLib), "Artifacts.toml")
        hash = artifact_hash("pysiglib_cuda12", manifest)
        hash === nothing && throw(ArgumentError("no CUDA artifact for this platform; import the upstream release manifest with scripts/update_artifacts.jl"))
        root = LazyArtifacts.ensure_artifact_installed("pysiglib_cuda12", manifest; pkg_uuid=Base.PkgId(SigLib).uuid)
        directory = joinpath(root, "pysiglib_cuda")
        Sys.iswindows() && Libdl.dlopen(joinpath(directory, "cudart64_12.dll"))
        filename = Sys.iswindows() ? "cusig.dll" : "libcusig.$(Libdl.dlext)"
        _libcusig[] = Libdl.dlopen(joinpath(directory, filename))
        _last_error[] = Libdl.dlsym(_libcusig[], :cusig_last_error_message)
    end
    return _libcusig[]
end

# cuSIG owns streams and exposes no stream argument. Device barriers order Julia
# producers/consumers and keep every buffer alive until native work has finished.
function _native(f, name::Symbol, arrays::CuArray...)
    lock(SigLib._cache_lock) do
        symbol = Libdl.dlsym(_library(), name)
        ctx = isempty(arrays) ? CUDA.context() : _context(first(arrays))
        all(x -> _context(x) == ctx, arrays) || throw(ArgumentError("all CUDA inputs must belong to the same context"))
        CUDA.context!(ctx) do
            GC.@preserve arrays f begin
                CUDA.device_synchronize()
                status = f(symbol)
                message = status == 0 ? C_NULL : ccall(_last_error[], Cstring, ())
                detail = message == C_NULL ? "" : unsafe_string(message)
                CUDA.device_synchronize()
                status == 0 || throw(SigLib.SigLibError(name, status, detail))
            end
        end
    end
    return nothing
end

function SigLib._cache_call(::Val{:cuda}, ::Val{:set_cache_dir}, path)
    _native(:set_cache_dir_cuda) do symbol
        ccall(symbol, Cint, (Cstring,), path)
    end
end
function SigLib._cache_call(::Val{:cuda}, ::Val{:clear_cache}, use_disk)
    _native(:clear_cache_cuda) do symbol
        ccall(symbol, Cint, (Bool,), use_disk)
    end
end
function SigLib._cache_call(::Val{:cuda}, ::Val{:prepare_log_sig}, d, n, m, use_disk)
    _native(:prepare_log_sig_cuda) do symbol
        ccall(symbol, Cint, (UInt64, UInt64, Cint, Bool), d, n, m, use_disk)
    end
end
function SigLib._cache_call(::Val{:cuda}, ::Val{:prepare_branched_sig}, d, n, planar, use_disk)
    _native(:prepare_branched_sig_cuda) do symbol
        ccall(symbol, Cint, (UInt64, UInt64, Bool, Bool), d, n, planar, use_disk)
    end
end
function SigLib._cache_call(::Val{:cuda}, ::Val{:prepare_branched_log_sig}, d, n, m, planar, use_disk)
    _native(:prepare_branched_log_sig_cuda) do symbol
        ccall(symbol, Cint, (UInt64, UInt64, Cint, Bool, Bool), d, n, m, planar, use_disk)
    end
end

include("cuda/c_api.jl")

end
