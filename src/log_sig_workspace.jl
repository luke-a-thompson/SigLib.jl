"""
    LogSigWorkspace(path, degree; method=1)

Allocate reusable CPU or CUDA buffers for [`log_sig!`](@ref). Inputs must have the same shape, backend, and `Float32` or `Float64` element type as `path`. Methods 0–3 are supported, without time augmentation, lead-lag, corrections, or a scalar term.

The workspace owns its output, which is overwritten on each call. Use a separate workspace for each concurrent computation.
"""
struct LogSigWorkspace{T,N,A<:AbstractMatrix{T},R<:AbstractArray{T}}
    signature::A
    logarithm::A
    result::R
    path_shape::NTuple{N,Int}
    degree::Int
    method::Int
end

function LogSigWorkspace(path::AbstractArray{T,N}, degree::Integer; method::Integer=1) where {T<:Real,N}
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    _as_log_sig_method(method)
    path_c, batch_shape, batch, dim, _ = _path_buffer(path, :log_sig!)
    siglen = method == 3 ? 0 : sig_length(dim, degree)
    outlen = _log_sig_output_length(dim, degree, method, false)
    signature = _allocate(path_c, siglen, batch)
    logarithm = _allocate(path_c, outlen, batch)
    return LogSigWorkspace(signature, logarithm, _format_signature_output(logarithm, batch_shape),
        size(path), Int(degree), Int(method))
end

"""
    log_sig!(workspace::LogSigWorkspace, path; n_jobs=1)

Compute a log signature using reusable buffers and return the workspace's output. Paths use `(dimension, length, batch...)`; results use `(coefficients, batch...)`. Dense inputs are passed directly to the native library; views are materialized when needed. The method and degree are fixed when the workspace is constructed.

The returned array is overwritten by the next call; copy it to retain a result. Do not use the same workspace concurrently. Native caches are checked on each call, including after [`clear_log_sig_cache`](@ref).
"""
function log_sig!(ws::LogSigWorkspace{T}, path::AbstractArray; n_jobs::Integer=1) where {T}
    size(path) == ws.path_shape || throw(DimensionMismatch("path shape must match the workspace"))
    eltype(path) === T || throw(ArgumentError("path eltype must match the workspace"))
    _check_backend(ws.logarithm, path)
    _as_cint_n_jobs(n_jobs)
    isempty(ws.result) && return ws.result

    path_c, _, batch, dim, len = _path_buffer(path, :log_sig!)
    _with_array(path_c) do
        lock(_cache_lock) do
            if ws.method != 0
                # Recheck preparation so clearing the native cache remains safe.
                _cache_call(Val(_backend(path_c)), Val(:prepare_log_sig), UInt64(dim), UInt64(ws.degree), Cint(ws.method), false)
            end
            if ws.method == 3
                _log_sig_from_path_c!(ws.logarithm, path_c, batch, len, dim, ws.degree, n_jobs)
            else
                _signature_c!(ws.signature, path_c, batch, dim, len, ws.degree,
                    false, false, true, false, n_jobs, one(T))
                _sig_to_log_sig_c!(ws.logarithm, ws.signature, batch, dim, ws.degree,
                    false, false, ws.method, false, n_jobs)
            end
        end
    end
    return ws.result
end
