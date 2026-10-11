struct SigLibError <: Exception
    function_name::Symbol
    status::Cint
    detail::String
end

SigLibError(function_name::Symbol, status::Integer) = SigLibError(function_name, Cint(status), "")
const CPSigError = SigLibError

const _status_messages = (
    "allocation failed", "invalid argument", "out of range", "filesystem error",
    "cache not prepared", "directory does not exist", "default cache directory unavailable",
    "invalid cache directory", "invalid cache file", "runtime error", "unknown exception",
)

function Base.showerror(io::IO, err::SigLibError)
    print(io, "native call ", err.function_name, " failed with status ", err.status)
    if !isempty(err.detail)
        print(io, ": ", err.detail)
    elseif 1 <= err.status <= length(_status_messages)
        print(io, ": ", _status_messages[err.status])
    end
end

function check_status(status::Cint, function_name::Symbol)
    if status != 0
        message = _last_error[] == C_NULL ? C_NULL : ccall(_last_error[], Cstring, ())
        detail = message == C_NULL ? "" : unsafe_string(message)
        throw(SigLibError(function_name, status, detail))
    end
    return nothing
end
