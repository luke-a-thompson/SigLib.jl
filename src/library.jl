using LazyArtifacts, Libdl

const _libcpsig = Ref{Ptr{Cvoid}}(C_NULL)
const _last_error = Ref{Ptr{Cvoid}}(C_NULL)
const _modern_abi = Ref(false)
const _cache_lock = ReentrantLock()

_cpsig_symbol(name::Symbol) = Libdl.dlsym(_libcpsig[], name)

function __init__()
    root = artifact"pysiglib_cpu"
    # The legacy 3.1 archive uses lib/; upstream wheel artifacts use pysiglib/.
    directory = joinpath(root, isdir(joinpath(root, "pysiglib")) ? "pysiglib" : "lib")
    Sys.iswindows() && Libdl.dlopen(joinpath(directory, "tbb12.dll"))
    filename = Sys.iswindows() ? "cpsig.dll" : "libcpsig.$(Libdl.dlext)"
    _libcpsig[] = Libdl.dlopen(joinpath(directory, filename))
    _last_error[] = Libdl.dlsym_e(_libcpsig[], :cpsig_last_error_message)
    _modern_abi[] = Libdl.dlsym_e(_libcpsig[], :branched_log_sig_length) != C_NULL
    return nothing
end
