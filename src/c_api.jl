# Concrete Float32/Float64 methods keep the ABI explicit without duplicate wrappers.

for (T, suffix) in ((Float32, :f), (Float64, :d))
    symbol = QuoteNode(Symbol(:signature_, suffix))
    @eval function _signature_c!(
        out::Matrix{$T},
        path::Array{$T,3},
        batch_size::Integer,
        dimension::Integer,
        len::Integer,
        degree::Integer,
        time_aug::Bool,
        lead_lag::Bool,
        horner::Bool,
        scalar_term::Bool,
        n_jobs::Integer,
        end_time::Real,
    )
        status = ccall(
            _cpsig_symbol($symbol),
            Cint,
            (
                Ptr{$T}, Ptr{$T},
                UInt64, UInt64, UInt64, UInt64,
                Bool, Bool, $T, Bool, Bool, Cint,
                Ptr{$T}, UInt64, UInt64, UInt64,
            ),
            path, out,
            _as_uint64(batch_size, "batch_size"),
            _as_uint64(dimension, "dimension"),
            _as_uint64(len, "length"),
            _as_uint64(degree, "degree"),
            time_aug, lead_lag, $T(end_time), horner, scalar_term, _as_cint_n_jobs(n_jobs),
            Ptr{$T}(C_NULL), UInt64(0), UInt64(0), UInt64(0),
        )
        check_status(status, $symbol)
        return out
    end
end

for (T, suffix) in ((Float32, :f), (Float64, :d))
    symbol = QuoteNode(Symbol(:transform_path_, suffix))
    @eval function _transform_path_c!(
        out::Array{$T,3},
        path::Array{$T,3},
        batch_size::Integer,
        dimension::Integer,
        len::Integer,
        time_aug::Bool,
        lead_lag::Bool,
        end_time::Real,
        n_jobs::Integer,
    )
        status = ccall(
            _cpsig_symbol($symbol),
            Cint,
            (Ptr{$T}, Ptr{$T}, UInt64, UInt64, UInt64, Bool, Bool, $T, Cint),
            path, out,
            _as_uint64(batch_size, "batch_size"),
            _as_uint64(dimension, "dimension"),
            _as_uint64(len, "length"),
            time_aug, lead_lag, $T(end_time), _as_cint_n_jobs(n_jobs),
        )
        check_status(status, $symbol)
        return out
    end
end

for (T, suffix) in ((Float32, :f), (Float64, :d))
    symbol = QuoteNode(Symbol(:transform_path_backprop_, suffix))
    @eval function _transform_path_backprop_c!(
        out::Array{$T,3},
        derivs::Array{$T,3},
        batch_size::Integer,
        dimension::Integer,
        len::Integer,
        time_aug::Bool,
        lead_lag::Bool,
        end_time::Real,
        n_jobs::Integer,
    )
        status = ccall(
            _cpsig_symbol($symbol),
            Cint,
            (Ptr{$T}, Ptr{$T}, UInt64, UInt64, UInt64, Bool, Bool, $T, Cint),
            derivs, out,
            _as_uint64(batch_size, "batch_size"),
            _as_uint64(dimension, "dimension"),
            _as_uint64(len, "length"),
            time_aug, lead_lag, $T(end_time), _as_cint_n_jobs(n_jobs),
        )
        check_status(status, $symbol)
        return out
    end
end

for (T, suffix) in ((Float32, :f), (Float64, :d))
    symbol = QuoteNode(Symbol(:sig_to_log_sig_, suffix))
    @eval function _sig_to_log_sig_c!(
        out::Matrix{$T},
        sig_value::Matrix{$T},
        batch_size::Integer,
        dimension::Integer,
        degree::Integer,
        time_aug::Bool,
        lead_lag::Bool,
        method::Integer,
        scalar_term::Bool,
        n_jobs::Integer,
    )
        status = ccall(
            _cpsig_symbol($symbol),
            Cint,
            (Ptr{$T}, Ptr{$T}, UInt64, UInt64, UInt64, Bool, Bool, Cint, Bool, Cint),
            sig_value, out,
            _as_uint64(batch_size, "batch_size"),
            _as_uint64(dimension, "dimension"),
            _as_uint64(degree, "degree"),
            time_aug, lead_lag, _as_cached_log_sig_method(method), scalar_term, _as_cint_n_jobs(n_jobs),
        )
        check_status(status, $symbol)
        return out
    end
end

for (T, suffix) in ((Float32, :f), (Float64, :d))
    symbol = QuoteNode(Symbol(:sig_to_log_sig_backprop_, suffix))
    @eval function _sig_to_log_sig_backprop_c!(
        out::Matrix{$T},
        sig_value::Matrix{$T},
        log_sig_derivs::Matrix{$T},
        batch_size::Integer,
        dimension::Integer,
        degree::Integer,
        time_aug::Bool,
        lead_lag::Bool,
        method::Integer,
        scalar_term::Bool,
        n_jobs::Integer,
    )
        status = ccall(
            _cpsig_symbol($symbol),
            Cint,
            (Ptr{$T}, Ptr{$T}, Ptr{$T}, UInt64, UInt64, UInt64, Bool, Bool, Cint, Bool, Cint),
            sig_value, out, log_sig_derivs,
            _as_uint64(batch_size, "batch_size"),
            _as_uint64(dimension, "dimension"),
            _as_uint64(degree, "degree"),
            time_aug, lead_lag, _as_cached_log_sig_method(method), scalar_term, _as_cint_n_jobs(n_jobs),
        )
        check_status(status, $symbol)
        return out
    end
end

for (T, suffix) in ((Float32, :f), (Float64, :d))
    symbol = QuoteNode(Symbol(:logsig_to_sig_, suffix))
    @eval function _logsig_to_sig_c!(
        out::Matrix{$T},
        log_sig_value::Matrix{$T},
        batch_size::Integer,
        dimension::Integer,
        degree::Integer,
        time_aug::Bool,
        lead_lag::Bool,
        method::Integer,
        scalar_term::Bool,
        n_jobs::Integer,
    )
        status = ccall(
            _cpsig_symbol($symbol),
            Cint,
            (Ptr{$T}, Ptr{$T}, UInt64, UInt64, UInt64, Bool, Bool, Cint, Bool, Cint),
            log_sig_value, out,
            _as_uint64(batch_size, "batch_size"),
            _as_uint64(dimension, "dimension"),
            _as_uint64(degree, "degree"),
            time_aug, lead_lag, _as_cached_log_sig_method(method), scalar_term, _as_cint_n_jobs(n_jobs),
        )
        check_status(status, $symbol)
        return out
    end
end

for (T, suffix) in ((Float32, :f), (Float64, :d))
    symbol = QuoteNode(Symbol(:logsig_to_sig_backprop_, suffix))
    @eval function _logsig_to_sig_backprop_c!(
        out::Matrix{$T},
        log_sig_value::Matrix{$T},
        sig_derivs::Matrix{$T},
        batch_size::Integer,
        dimension::Integer,
        degree::Integer,
        time_aug::Bool,
        lead_lag::Bool,
        method::Integer,
        scalar_term::Bool,
        n_jobs::Integer,
    )
        status = ccall(
            _cpsig_symbol($symbol),
            Cint,
            (Ptr{$T}, Ptr{$T}, Ptr{$T}, UInt64, UInt64, UInt64, Bool, Bool, Cint, Bool, Cint),
            log_sig_value, out, sig_derivs,
            _as_uint64(batch_size, "batch_size"),
            _as_uint64(dimension, "dimension"),
            _as_uint64(degree, "degree"),
            time_aug, lead_lag, _as_cached_log_sig_method(method), scalar_term, _as_cint_n_jobs(n_jobs),
        )
        check_status(status, $symbol)
        return out
    end
end

for (T, suffix) in ((Float32, :f), (Float64, :d))
    symbol = QuoteNode(Symbol(:log_sig_from_path_, suffix))
    @eval function _log_sig_from_path_c!(
        out::Matrix{$T},
        path::Array{$T,3},
        batch_size::Integer,
        len::Integer,
        dimension::Integer,
        degree::Integer,
        n_jobs::Integer,
    )
        status = ccall(
            _cpsig_symbol($symbol),
            Cint,
            (Ptr{$T}, Ptr{$T}, UInt64, UInt64, UInt64, UInt64, Cint),
            path, out,
            _as_uint64(batch_size, "batch_size"),
            _as_uint64(len, "length"),
            _as_uint64(dimension, "dimension"),
            _as_uint64(degree, "degree"),
            _as_cint_n_jobs(n_jobs),
        )
        check_status(status, $symbol)
        return out
    end
end

for (T, suffix) in ((Float32, :f), (Float64, :d))
    symbol = QuoteNode(Symbol(:log_sig_from_path_backprop_, suffix))
    @eval function _log_sig_from_path_backprop_c!(
        out::Array{$T,3},
        log_sig_derivs::Matrix{$T},
        path::Array{$T,3},
        batch_size::Integer,
        len::Integer,
        dimension::Integer,
        degree::Integer,
        n_jobs::Integer,
    )
        status = ccall(
            _cpsig_symbol($symbol),
            Cint,
            (Ptr{$T}, Ptr{$T}, Ptr{$T}, UInt64, UInt64, UInt64, UInt64, Cint),
            log_sig_derivs, out, path,
            _as_uint64(batch_size, "batch_size"),
            _as_uint64(len, "length"),
            _as_uint64(dimension, "dimension"),
            _as_uint64(degree, "degree"),
            _as_cint_n_jobs(n_jobs),
        )
        check_status(status, $symbol)
        return out
    end
end

for (T, suffix) in ((Float32, :f), (Float64, :d))
    symbol = QuoteNode(Symbol(:log_sig_combine_, suffix))
    @eval function _log_sig_combine_c!(
        out::Matrix{$T},
        log_sig1::Matrix{$T},
        log_sig2::Matrix{$T},
        batch_size::Integer,
        dimension::Integer,
        degree::Integer,
        n_jobs::Integer,
    )
        status = ccall(
            _cpsig_symbol($symbol),
            Cint,
            (Ptr{$T}, Ptr{$T}, Ptr{$T}, UInt64, UInt64, UInt64, Cint),
            log_sig1, log_sig2, out,
            _as_uint64(batch_size, "batch_size"),
            _as_uint64(dimension, "dimension"),
            _as_uint64(degree, "degree"),
            _as_cint_n_jobs(n_jobs),
        )
        check_status(status, $symbol)
        return out
    end
end

for (T, suffix) in ((Float32, :f), (Float64, :d))
    symbol = QuoteNode(Symbol(:log_sig_combine_backprop_, suffix))
    @eval function _log_sig_combine_backprop_c!(
        out1::Matrix{$T},
        out2::Matrix{$T},
        derivs::Matrix{$T},
        log_sig1::Matrix{$T},
        log_sig2::Matrix{$T},
        batch_size::Integer,
        dimension::Integer,
        degree::Integer,
        n_jobs::Integer,
    )
        status = ccall(
            _cpsig_symbol($symbol),
            Cint,
            (Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, UInt64, UInt64, UInt64, Cint),
            derivs, out1, out2, log_sig1, log_sig2,
            _as_uint64(batch_size, "batch_size"),
            _as_uint64(dimension, "dimension"),
            _as_uint64(degree, "degree"),
            _as_cint_n_jobs(n_jobs),
        )
        check_status(status, $symbol)
        return out1, out2
    end
end

for (T, suffix) in ((Float32, :f), (Float64, :d))
    symbol = QuoteNode(Symbol(:log_sig_join_, suffix))
    @eval function _log_sig_join_c!(
        out::Matrix{$T},
        log_sig_value::Matrix{$T},
        displacement::Matrix{$T},
        batch_size::Integer,
        dimension::Integer,
        degree::Integer,
        n_jobs::Integer,
    )
        status = ccall(
            _cpsig_symbol($symbol),
            Cint,
            (Ptr{$T}, Ptr{$T}, Ptr{$T}, UInt64, UInt64, UInt64, Cint),
            log_sig_value, displacement, out,
            _as_uint64(batch_size, "batch_size"),
            _as_uint64(dimension, "dimension"),
            _as_uint64(degree, "degree"),
            _as_cint_n_jobs(n_jobs),
        )
        check_status(status, $symbol)
        return out
    end
end

for (T, suffix) in ((Float32, :f), (Float64, :d))
    symbol = QuoteNode(Symbol(:log_sig_join_backprop_, suffix))
    @eval function _log_sig_join_backprop_c!(
        out_log_sig::Matrix{$T},
        out_displacement::Matrix{$T},
        derivs::Matrix{$T},
        log_sig_value::Matrix{$T},
        displacement::Matrix{$T},
        batch_size::Integer,
        dimension::Integer,
        degree::Integer,
        n_jobs::Integer,
    )
        status = ccall(
            _cpsig_symbol($symbol),
            Cint,
            (Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, UInt64, UInt64, UInt64, Cint),
            derivs, out_log_sig, out_displacement, log_sig_value, displacement,
            _as_uint64(batch_size, "batch_size"),
            _as_uint64(dimension, "dimension"),
            _as_uint64(degree, "degree"),
            _as_cint_n_jobs(n_jobs),
        )
        check_status(status, $symbol)
        return out_log_sig, out_displacement
    end
end

for (T, suffix) in ((Float32, :f), (Float64, :d))
    symbol = QuoteNode(Symbol(:branched_sig_, suffix))
    @eval function _branched_sig_c!(
        out::Matrix{$T},
        path::Array{$T,3},
        batch_size::Integer,
        dimension::Integer,
        len::Integer,
        max_nodes::Integer,
        time_aug::Bool,
        lead_lag::Bool,
        planar::Bool,
        scalar_term::Bool,
        n_jobs::Integer,
        end_time::Real,
    )
        status = ccall(
            _cpsig_symbol($symbol),
            Cint,
            (
                Ptr{$T}, Ptr{$T},
                UInt64, UInt64, UInt64, UInt64,
                Cint, Bool, Bool, $T, Bool, Bool,
                Ptr{$T}, UInt64, UInt64, UInt64,
            ),
            path, out,
            _as_uint64(batch_size, "batch_size"),
            _as_uint64(dimension, "dimension"),
            _as_uint64(len, "length"),
            _as_uint64(max_nodes, "max_nodes"),
            _as_cint_n_jobs(n_jobs), time_aug, lead_lag, $T(end_time), planar, scalar_term,
            Ptr{$T}(C_NULL), UInt64(0), UInt64(0), UInt64(0),
        )
        check_status(status, $symbol)
        return out
    end
end

for (T, suffix) in ((Float32, :f), (Float64, :d))
    symbol = QuoteNode(Symbol(:branched_sig_combine_, suffix))
    @eval function _branched_sig_combine_c!(
        out::Matrix{$T},
        bsig1::Matrix{$T},
        bsig2::Matrix{$T},
        batch_size::Integer,
        dimension::Integer,
        max_nodes::Integer,
        planar::Bool,
        scalar_term::Bool,
        n_jobs::Integer,
    )
        status = ccall(
            _cpsig_symbol($symbol),
            Cint,
            (
                Ptr{$T}, Ptr{$T}, Ptr{$T},
                UInt64, UInt64, UInt64, Cint, Bool, Bool,
            ),
            bsig1, bsig2, out,
            _as_uint64(batch_size, "batch_size"),
            _as_uint64(dimension, "dimension"),
            _as_uint64(max_nodes, "max_nodes"),
            _as_cint_n_jobs(n_jobs), planar, scalar_term,
        )
        check_status(status, $symbol)
        return out
    end
end

for (T, suffix) in ((Float32, :f), (Float64, :d))
    symbol = QuoteNode(Symbol(:branched_sig_to_log_sig_, suffix))
    @eval function _branched_sig_to_log_sig_c!(
        out::Matrix{$T},
        bsig::Matrix{$T},
        batch_size::Integer,
        dimension::Integer,
        max_nodes::Integer,
        planar::Bool,
        scalar_term::Bool,
        n_jobs::Integer,
        method::Integer,
    )
        status = if _modern_abi[]
            ccall(
                _cpsig_symbol($symbol),
                Cint,
                (
                    Ptr{$T}, Ptr{$T},
                    UInt64, UInt64, UInt64, Cint, Cint, Bool, Bool,
                ),
                bsig, out,
                _as_uint64(batch_size, "batch_size"),
                _as_uint64(dimension, "dimension"),
                _as_uint64(max_nodes, "max_nodes"),
                Cint(method), _as_cint_n_jobs(n_jobs), planar, scalar_term,
            )
        else
            ccall(
                _cpsig_symbol($symbol),
                Cint,
                (
                    Ptr{$T}, Ptr{$T},
                    UInt64, UInt64, UInt64, Cint, Bool, Bool,
                ),
                bsig, out,
                _as_uint64(batch_size, "batch_size"),
                _as_uint64(dimension, "dimension"),
                _as_uint64(max_nodes, "max_nodes"),
                _as_cint_n_jobs(n_jobs), planar, scalar_term,
            )
        end
        check_status(status, $symbol)
        return out
    end
end

for (T, suffix) in ((Float32, :f), (Float64, :d))
    symbol = QuoteNode(Symbol(:sig_backprop_, suffix))
    @eval function _sig_backprop_c!(
        out::Array{$T,3},
        path::Array{$T,3},
        sig_derivs::Matrix{$T},
        sig_value::Matrix{$T},
        batch_size::Integer,
        dimension::Integer,
        len::Integer,
        degree::Integer,
        time_aug::Bool,
        lead_lag::Bool,
        scalar_term::Bool,
        n_jobs::Integer,
        end_time::Real,
    )
        status = ccall(
            _cpsig_symbol($symbol),
            Cint,
            (
                Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T},
                UInt64, UInt64, UInt64, UInt64,
                Bool, Bool, $T, Bool, Cint,
                Ptr{$T}, UInt64, UInt64, UInt64,
            ),
            path, out, sig_derivs, sig_value,
            _as_uint64(batch_size, "batch_size"),
            _as_uint64(dimension, "dimension"),
            _as_uint64(len, "length"),
            _as_uint64(degree, "degree"),
            time_aug, lead_lag, $T(end_time), scalar_term, _as_cint_n_jobs(n_jobs),
            Ptr{$T}(C_NULL), UInt64(0), UInt64(0), UInt64(0),
        )
        check_status(status, $symbol)
        return out
    end
end

for (T, suffix) in ((Float32, :f), (Float64, :d))
    symbol = QuoteNode(Symbol(:branched_sig_backprop_, suffix))
    @eval function _branched_sig_backprop_c!(
        out::Array{$T,3},
        path::Array{$T,3},
        bsig_derivs::Matrix{$T},
        bsig_value::Matrix{$T},
        batch_size::Integer,
        dimension::Integer,
        len::Integer,
        max_nodes::Integer,
        time_aug::Bool,
        lead_lag::Bool,
        planar::Bool,
        scalar_term::Bool,
        n_jobs::Integer,
        end_time::Real,
    )
        status = ccall(
            _cpsig_symbol($symbol),
            Cint,
            (
                Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T},
                UInt64, UInt64, UInt64, UInt64,
                Cint, Bool, Bool, $T, Bool, Bool,
                Ptr{$T}, UInt64, UInt64, UInt64,
            ),
            path, out, bsig_derivs, bsig_value,
            _as_uint64(batch_size, "batch_size"),
            _as_uint64(dimension, "dimension"),
            _as_uint64(len, "length"),
            _as_uint64(max_nodes, "max_nodes"),
            _as_cint_n_jobs(n_jobs), time_aug, lead_lag, $T(end_time), planar, scalar_term,
            Ptr{$T}(C_NULL), UInt64(0), UInt64(0), UInt64(0),
        )
        check_status(status, $symbol)
        return out
    end
end

for (T, suffix) in ((Float32, :f), (Float64, :d))
    symbol = QuoteNode(Symbol(:branched_sig_to_log_sig_backprop_, suffix))
    @eval function _branched_sig_to_log_sig_backprop_c!(
        out::Matrix{$T},
        bsig::Matrix{$T},
        derivs::Matrix{$T},
        batch_size::Integer,
        dimension::Integer,
        max_nodes::Integer,
        planar::Bool,
        scalar_term::Bool,
        n_jobs::Integer,
        method::Integer,
    )
        status = if _modern_abi[]
            ccall(
                _cpsig_symbol($symbol),
                Cint,
                (
                    Ptr{$T}, Ptr{$T}, Ptr{$T},
                    UInt64, UInt64, UInt64, Cint, Cint, Bool, Bool,
                ),
                bsig, derivs, out,
                _as_uint64(batch_size, "batch_size"),
                _as_uint64(dimension, "dimension"),
                _as_uint64(max_nodes, "max_nodes"),
                Cint(method), _as_cint_n_jobs(n_jobs), planar, scalar_term,
            )
        else
            ccall(
                _cpsig_symbol($symbol),
                Cint,
                (
                    Ptr{$T}, Ptr{$T}, Ptr{$T},
                    UInt64, UInt64, UInt64, Cint, Bool, Bool,
                ),
                bsig, derivs, out,
                _as_uint64(batch_size, "batch_size"),
                _as_uint64(dimension, "dimension"),
                _as_uint64(max_nodes, "max_nodes"),
                _as_cint_n_jobs(n_jobs), planar, scalar_term,
            )
        end
        check_status(status, $symbol)
        return out
    end
end

for (T, suffix) in ((Float32, :f), (Float64, :d))
    symbol = QuoteNode(Symbol(:branched_sig_combine_backprop_, suffix))
    @eval function _branched_sig_combine_backprop_c!(
        out1::Matrix{$T},
        out2::Matrix{$T},
        bsig1::Matrix{$T},
        bsig2::Matrix{$T},
        derivs::Matrix{$T},
        batch_size::Integer,
        dimension::Integer,
        max_nodes::Integer,
        planar::Bool,
        scalar_term::Bool,
        n_jobs::Integer,
    )
        status = ccall(
            _cpsig_symbol($symbol),
            Cint,
            (
                Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ptr{$T},
                UInt64, UInt64, UInt64, Cint, Bool, Bool,
            ),
            bsig1, bsig2, derivs, out1, out2,
            _as_uint64(batch_size, "batch_size"),
            _as_uint64(dimension, "dimension"),
            _as_uint64(max_nodes, "max_nodes"),
            _as_cint_n_jobs(n_jobs), planar, scalar_term,
        )
        check_status(status, $symbol)
        return out1, out2
    end
end

for (T, suffix) in ((Float32, :f), (Float64, :d))
    forward = QuoteNode(Symbol(:branched_log_sig_from_path_, suffix))
    backward = QuoteNode(Symbol(:branched_log_sig_from_path_backprop_, suffix))
    @eval function _branched_log_sig_from_path_c!(out::Matrix{$T}, path::Array{$T,3}, batch_size, len, dimension, degree, n_jobs)
        status = ccall(_cpsig_symbol($forward), Cint,
            (Ptr{$T}, Ptr{$T}, UInt64, UInt64, UInt64, UInt64, Cint),
            path, out, _as_uint64(batch_size, "batch_size"), _as_uint64(len, "length"),
            _as_uint64(dimension, "dimension"), _as_uint64(degree, "degree"), _as_cint_n_jobs(n_jobs))
        check_status(status, $forward)
        return out
    end
    @eval function _branched_log_sig_from_path_backprop_c!(out::Array{$T,3}, derivs::Matrix{$T}, path::Array{$T,3}, batch_size, len, dimension, degree, n_jobs)
        status = ccall(_cpsig_symbol($backward), Cint,
            (Ptr{$T}, Ptr{$T}, Ptr{$T}, UInt64, UInt64, UInt64, UInt64, Cint),
            derivs, out, path, _as_uint64(batch_size, "batch_size"), _as_uint64(len, "length"),
            _as_uint64(dimension, "dimension"), _as_uint64(degree, "degree"), _as_cint_n_jobs(n_jobs))
        check_status(status, $backward)
        return out
    end
end
