# SigLib.jl

`SigLib.jl` is a small Julia wrapper around `cpsig_jll`.

The initial public API mirrors pySigLib's standard signature functions:

```julia
sig_length(dimension, degree; scalar_term=false)
sig(path, degree; scalar_term=false, n_jobs=1)
```

Paths use pySigLib's shape convention: `(length, dimension)` for one path and
`(batch..., length, dimension)` for batches.
