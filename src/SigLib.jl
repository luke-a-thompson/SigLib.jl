module SigLib

include("errors.jl")
include("c_api.jl")
include("signatures.jl")

const signature = sig
const signature_length = sig_length

export sig, sig_backprop, sig_length, signature, signature_length, SigLibError, CPSigError

end
