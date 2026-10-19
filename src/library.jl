using LazyArtifacts, Libdl

const _libcpsig = Ref{Ptr{Cvoid}}(C_NULL)
const _last_error = Ref{Ptr{Cvoid}}(C_NULL)
const _cache_lock = ReentrantLock()

_cpsig_symbol(name::Symbol) = Libdl.dlsym(_libcpsig[], name)

function __init__()
    root = artifact"pysiglib_cpu"
    directory = joinpath(root, "pysiglib")
    Sys.iswindows() && Libdl.dlopen(joinpath(directory, "tbb12.dll"))
    filename = Sys.iswindows() ? "cpsig.dll" : "libcpsig.$(Libdl.dlext)"
    _libcpsig[] = Libdl.dlopen(joinpath(directory, filename))
    _last_error[] = Libdl.dlsym(_libcpsig[], :cpsig_last_error_message)
    return nothing
end
