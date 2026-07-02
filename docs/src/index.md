# SigLib.jl

`SigLib.jl` is a small Julia wrapper around `cpsig_jll`.

The public API mirrors pySigLib's standard and branched signature functions:

```julia
sig_length(dimension, degree; scalar_term=false)
sig(path, degree; scalar_term=false, n_jobs=1)
log_sig_length(dimension, degree; time_aug=false, lead_lag=false)
prepare_log_sig(dimension, degree, method; time_aug=false, lead_lag=false)
sig_to_log_sig(sig, dimension, degree; time_aug=false, lead_lag=false, method=1, n_jobs=1)
logsig_to_sig(log_sig, dimension, degree; time_aug=false, lead_lag=false, method=1, scalar_term=false, n_jobs=1)
log_sig(path, degree; time_aug=false, lead_lag=false, end_time=1.0, method=1, scalar_term=false, n_jobs=1)
log_sig_combine(log_sig1, log_sig2, dimension, degree; n_jobs=1)
log_sig_join(log_sig, displacement, dimension, degree; n_jobs=1)
branched_sig_length(dimension, max_nodes; planar=false, scalar_term=false)
prepare_branched_sig(dimension, max_nodes; planar=false)
branched_sig(path, max_nodes; planar=false, scalar_term=false, n_jobs=1)
branched_sig_combine(bsig1, bsig2, dimension, max_nodes; planar=false)
branched_sig_to_log_sig(bsig, dimension, max_nodes; planar=false)
branched_log_sig(path, max_nodes; planar=false, scalar_term=false, n_jobs=1)
```

Paths use pySigLib's shape convention: `(length, dimension)` for one path and
`(batch..., length, dimension)` for batches.

Derivatives are provided through ChainRules AD rules.
