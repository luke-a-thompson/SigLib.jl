module cpsig_jll

using LazyArtifacts
using Libdl

export libcpsig, cpsig_h, cppch_h

const cpsig = artifact"cpsig"
const libcpsig = joinpath(cpsig, "lib", "libcpsig.so")
const cpsig_h = joinpath(cpsig, "include", "cpsig.h")
const cppch_h = joinpath(cpsig, "include", "cppch.h")

function __init__()
    isfile(libcpsig) || error("libcpsig not found at $libcpsig")
    Libdl.dlopen(libcpsig)
    return nothing
end

end
