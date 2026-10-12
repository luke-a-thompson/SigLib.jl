_correction_arrays(correction) = correction.data === nothing ? () : (correction.data,)

# Concrete element types make every CUDA ABI signature explicit.
for (T, suffix) in ((Float32, :f), (Float64, :d))
    @eval begin
        function SigLib._signature_c!(out::CuArray{$T,2}, path::CuArray{$T,3}, batch, d, len, n, ta, ll, horner, scalar, jobs, end_time, correction::SigLib._CorrectionBuffer=SigLib._CorrectionBuffer())
            _native($(QuoteNode(Symbol(:signature_cuda_, suffix))), out, path, _correction_arrays(correction)...) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, UInt64, Bool, Bool, $T, Bool, Bool, CuPtr{$T}, UInt64, UInt64, UInt64),
                    path, out, batch, d, len, n, ta, ll, end_time, horner, scalar, correction.data === nothing ? CuPtr{$T}(0) : correction.data,
                    correction.length, correction.batch_stride, correction.segment_stride)
            end
            return out
        end
        function SigLib._sig_backprop_c!(out::CuArray{$T,3}, path::CuArray{$T,3}, derivs::CuArray{$T,2}, sig::CuArray{$T,2}, batch, d, len, n, ta, ll, scalar, jobs, end_time, correction::SigLib._CorrectionBuffer=SigLib._CorrectionBuffer())
            _native($(QuoteNode(Symbol(:sig_backprop_cuda_, suffix))), out, path, derivs, sig, _correction_arrays(correction)...) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, UInt64, Bool, Bool, $T, Bool, CuPtr{$T}, UInt64, UInt64, UInt64),
                    path, out, derivs, sig, batch, d, len, n, ta, ll, end_time, scalar, correction.data === nothing ? CuPtr{$T}(0) : correction.data,
                    correction.length, correction.batch_stride, correction.segment_stride)
            end
            return out
        end
        function SigLib._transform_path_c!(out::CuArray{$T,3}, path::CuArray{$T,3}, batch, d, len, ta, ll, end_time, jobs)
            _native($(QuoteNode(Symbol(:transform_path_cuda_, suffix))), out, path) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, Bool, Bool, $T),
                    path, out, batch, d, len, ta, ll, end_time)
            end
            return out
        end
        function SigLib._transform_path_backprop_c!(out::CuArray{$T,3}, derivs::CuArray{$T,3}, batch, d, len, ta, ll, end_time, jobs)
            _native($(QuoteNode(Symbol(:transform_path_backprop_cuda_, suffix))), out, derivs) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, Bool, Bool, $T),
                    derivs, out, batch, d, len, ta, ll, end_time)
            end
            return out
        end
        function SigLib._sig_to_log_sig_c!(out::CuArray{$T,2}, sig::CuArray{$T,2}, batch, d, n, ta, ll, method, scalar, jobs)
            _native($(QuoteNode(Symbol(:sig_to_log_sig_cuda_, suffix))), out, sig) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, Cint, Bool),
                    sig, out, batch, SigLib._effective_dimension(d, ta, ll), n, method, scalar)
            end
            return out
        end
        function SigLib._sig_to_log_sig_backprop_c!(out::CuArray{$T,2}, sig::CuArray{$T,2}, derivs::CuArray{$T,2}, batch, d, n, ta, ll, method, scalar, jobs)
            _native($(QuoteNode(Symbol(:sig_to_log_sig_backprop_cuda_, suffix))), out, sig, derivs) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, Cint, Bool),
                    sig, out, derivs, batch, SigLib._effective_dimension(d, ta, ll), n, method, scalar)
            end
            return out
        end
        function SigLib._logsig_to_sig_c!(out::CuArray{$T,2}, log::CuArray{$T,2}, batch, d, n, ta, ll, method, scalar, jobs)
            _native($(QuoteNode(Symbol(:logsig_to_sig_cuda_, suffix))), out, log) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, Cint, Bool),
                    log, out, batch, SigLib._effective_dimension(d, ta, ll), n, method, scalar)
            end
            return out
        end
        function SigLib._logsig_to_sig_backprop_c!(out::CuArray{$T,2}, log::CuArray{$T,2}, derivs::CuArray{$T,2}, batch, d, n, ta, ll, method, scalar, jobs)
            _native($(QuoteNode(Symbol(:logsig_to_sig_backprop_cuda_, suffix))), out, log, derivs) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, Cint, Bool),
                    log, out, derivs, batch, SigLib._effective_dimension(d, ta, ll), n, method, scalar)
            end
            return out
        end
        function SigLib._log_sig_from_path_c!(out::CuArray{$T,2}, path::CuArray{$T,3}, batch, len, d, n, jobs)
            _native($(QuoteNode(Symbol(:log_sig_from_path_cuda_, suffix))), out, path) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, UInt64), path, out, batch, len, d, n)
            end
            return out
        end
        function SigLib._log_sig_from_path_backprop_c!(out::CuArray{$T,3}, derivs::CuArray{$T,2}, path::CuArray{$T,3}, batch, len, d, n, jobs)
            _native($(QuoteNode(Symbol(:log_sig_from_path_backprop_cuda_, suffix))), out, derivs, path) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, UInt64), derivs, out, path, batch, len, d, n)
            end
            return out
        end
        function SigLib._log_sig_combine_c!(out::CuArray{$T,2}, log1::CuArray{$T,2}, log2::CuArray{$T,2}, batch, d, n, jobs)
            _native($(QuoteNode(Symbol(:log_sig_combine_cuda_, suffix))), out, log1, log2) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64), log1, log2, out, batch, d, n)
            end
            return out
        end
        function SigLib._log_sig_combine_backprop_c!(out1::CuArray{$T,2}, out2::CuArray{$T,2}, derivs::CuArray{$T,2}, log1::CuArray{$T,2}, log2::CuArray{$T,2}, batch, d, n, jobs)
            _native($(QuoteNode(Symbol(:log_sig_combine_backprop_cuda_, suffix))), out1, out2, derivs, log1, log2) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64), derivs, out1, out2, log1, log2, batch, d, n)
            end
            return out1, out2
        end
        function SigLib._log_sig_join_c!(out::CuArray{$T,2}, log::CuArray{$T,2}, displacement::CuArray{$T,2}, batch, d, n, jobs)
            _native($(QuoteNode(Symbol(:log_sig_join_cuda_, suffix))), out, log, displacement) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64), log, displacement, out, batch, d, n)
            end
            return out
        end
        function SigLib._log_sig_join_backprop_c!(out_log::CuArray{$T,2}, out_displacement::CuArray{$T,2}, derivs::CuArray{$T,2}, log::CuArray{$T,2}, displacement::CuArray{$T,2}, batch, d, n, jobs)
            _native($(QuoteNode(Symbol(:log_sig_join_backprop_cuda_, suffix))), out_log, out_displacement, derivs, log, displacement) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64), derivs, out_log, out_displacement, log, displacement, batch, d, n)
            end
            return out_log, out_displacement
        end
        function SigLib._branched_sig_c!(out::CuArray{$T,2}, path::CuArray{$T,3}, batch, d, len, n, ta, ll, planar, scalar, jobs, end_time, correction::SigLib._CorrectionBuffer=SigLib._CorrectionBuffer())
            _native($(QuoteNode(Symbol(:branched_sig_cuda_, suffix))), out, path, _correction_arrays(correction)...) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, UInt64, Bool, Bool, $T, Bool, Bool, CuPtr{$T}, UInt64, UInt64, UInt64),
                    path, out, batch, d, len, n, ta, ll, end_time, planar, scalar, correction.data === nothing ? CuPtr{$T}(0) : correction.data,
                    correction.length, correction.batch_stride, correction.segment_stride)
            end
            return out
        end
        function SigLib._branched_sig_backprop_c!(out::CuArray{$T,3}, path::CuArray{$T,3}, derivs::CuArray{$T,2}, sig::CuArray{$T,2}, batch, d, len, n, ta, ll, planar, scalar, jobs, end_time, correction::SigLib._CorrectionBuffer=SigLib._CorrectionBuffer())
            _native($(QuoteNode(Symbol(:branched_sig_backprop_cuda_, suffix))), out, path, derivs, sig, _correction_arrays(correction)...) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, UInt64, Bool, Bool, $T, Bool, Bool, CuPtr{$T}, UInt64, UInt64, UInt64),
                    path, out, derivs, sig, batch, d, len, n, ta, ll, end_time, planar, scalar, correction.data === nothing ? CuPtr{$T}(0) : correction.data,
                    correction.length, correction.batch_stride, correction.segment_stride)
            end
            return out
        end
        function SigLib._branched_sig_combine_c!(out::CuArray{$T,2}, sig1::CuArray{$T,2}, sig2::CuArray{$T,2}, batch, d, n, planar, scalar, jobs)
            _native($(QuoteNode(Symbol(:branched_sig_combine_cuda_, suffix))), out, sig1, sig2) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, Bool, Bool), sig1, sig2, out, batch, d, n, planar, scalar)
            end
            return out
        end
        function SigLib._branched_sig_combine_backprop_c!(out1::CuArray{$T,2}, out2::CuArray{$T,2}, sig1::CuArray{$T,2}, sig2::CuArray{$T,2}, derivs::CuArray{$T,2}, batch, d, n, planar, scalar, jobs)
            _native($(QuoteNode(Symbol(:branched_sig_combine_backprop_cuda_, suffix))), out1, out2, sig1, sig2, derivs) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, Bool, Bool), sig1, sig2, derivs, out1, out2, batch, d, n, planar, scalar)
            end
            return out1, out2
        end
        function SigLib._branched_sig_to_log_sig_c!(out::CuArray{$T,2}, sig::CuArray{$T,2}, batch, d, n, planar, scalar, jobs, method=0)
            _native($(QuoteNode(Symbol(:branched_sig_to_log_sig_cuda_, suffix))), out, sig) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, Cint, Bool, Bool), sig, out, batch, d, n, method, planar, scalar)
            end
            return out
        end
        function SigLib._branched_sig_to_log_sig_backprop_c!(out::CuArray{$T,2}, sig::CuArray{$T,2}, derivs::CuArray{$T,2}, batch, d, n, planar, scalar, jobs, method=0)
            _native($(QuoteNode(Symbol(:branched_sig_to_log_sig_backprop_cuda_, suffix))), out, sig, derivs) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, Cint, Bool, Bool), sig, derivs, out, batch, d, n, method, planar, scalar)
            end
            return out
        end
        function SigLib._branched_log_sig_from_path_c!(out::CuArray{$T,2}, path::CuArray{$T,3}, batch, len, d, n, jobs)
            _native($(QuoteNode(Symbol(:branched_log_sig_from_path_cuda_, suffix))), out, path) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, UInt64), path, out, batch, len, d, n)
            end
            return out
        end
        function SigLib._branched_log_sig_from_path_backprop_c!(out::CuArray{$T,3}, derivs::CuArray{$T,2}, path::CuArray{$T,3}, batch, len, d, n, jobs)
            _native($(QuoteNode(Symbol(:branched_log_sig_from_path_backprop_cuda_, suffix))), out, derivs, path) do symbol
                ccall(symbol, Cint, (CuPtr{$T}, CuPtr{$T}, CuPtr{$T}, UInt64, UInt64, UInt64, UInt64), derivs, out, path, batch, len, d, n)
            end
            return out
        end
    end
end
