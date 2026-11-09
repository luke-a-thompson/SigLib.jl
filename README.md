# SigLib.jl

**SigLib.jl** calls [pySigLib's native cpSIG/cuSIG libraries](https://github.com/daniil-shmelev/pySigLib/tree/main/siglib) through Julia artifacts. Python and a compiler are unnecessary. The interface uses Julia's channel-first, batch-last arrays and pySigLib's signature conventions, with optional CUDA and ChainRules extensions.

```julia
using SigLib
path = [0.0 0.5 1.0; 0.0 1.0 -0.25] # (dimension, length)
s = sig(path, 2)
sig!(s, path, 2) # reuse output storage
l = log_sig(path, 2; method=LogSigMethod.LyndonBasis)
log_sig!(l, path, 2, s; method=LogSigMethod.LyndonBasis) # reuse output and scratch
bs = branched_sig(path, 2)
branched_sig!(bs, path, 2)
bl = similar(bs)
branched_log_sig!(bl, path, 2, bs) # bs is reusable scratch storage
sig(path, 3; correction=[-0.01, 0.0, 0.0, -0.01]) # local level-2 correction
lyndon_brackets(2, 3) # standard brackets in LyndonBasis/DirectBCH coordinate order
trees(2, 3) # BCK trees; forests(2, 3) gives MKW ordered forests
lyndon_forest_to_idx((((0,), 0),), 2, 3) # compressed MKW log index
branched_log_sig_basis(2, 3; planar=true, method=BranchedLogSigMethod.MkwBasis) # compressed MKW brackets
```

Methods use scoped enum values. Signatures use `SigMethod.Auto` (default), `.Horner`, or `.PrefixSuffix`. Log signatures use `LogSigMethod.Expanded`, `.LyndonWords` (default), `.LyndonBasis`, or `.DirectBCH`. Branched logs use `BranchedLogSigMethod.Expanded`, `.MkwWords`, `.MkwBasis`, or `.DirectBCH`; the default is `Expanded` for nonplanar paths and `MkwWords` for planar paths.

## Documentation

See [`docs/src/index.md`](docs/src/index.md) for supported operations, installation, native artifact updates, and extension conventions. The bundled artifact uses a pinned GitHub checkout with runtime SIMD dispatch and requires Linux x86-64 with glibc 2.38 or newer. SigLib requires the 4.x native ABI.

## License

See [`LICENSE`](LICENSE).
