struct SigLibError <: Exception
    function_name::Symbol
    status::Cint
end

const CPSigError = SigLibError

function Base.showerror(io::IO, err::SigLibError)
    print(io, "cpsig call ", err.function_name, " failed with status ", err.status)
end

function check_status(status::Cint, function_name::Symbol)
    status == 0 || throw(SigLibError(function_name, status))
    return nothing
end
