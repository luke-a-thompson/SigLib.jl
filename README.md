# SigLib.jl

**SigLib.jl** calls [pySigLib's native cpSIG/cuSIG libraries](https://github.com/daniil-shmelev/pySigLib/tree/main/siglib) through Julia artifacts. Python and a compiler are unnecessary. The interface uses Julia's channel-first, batch-last arrays and pySigLib's signature conventions, with optional CUDA and ChainRules extensions.

```julia
using SigLib
path = [0.0 0.5 1.0; 0.0 1.0 -0.25] # (dimension, length)
sig(path, 2)
log_sig(path, 2; method=LogSigMethod.LyndonBasis)
workspace = LogSigWorkspace(path, 2; method=LogSigMethod.LyndonBasis)
log_sig!(workspace, path) # reuse buffers for subsequent paths of the same shape
branched_sig(path, 2)
sig(path, 3; correction=[-0.01, 0.0, 0.0, -0.01]) # local level-2 correction
lyndon_brackets(2, 3) # standard brackets in LyndonBasis/DirectBCH coordinate order
trees(2, 3) # BCK trees; planar=true gives MKW ordered forests
branched_log_sig_basis(2, 3; planar=true, method=BranchedLogSigMethod.MkwBasis) # compressed MKW brackets
```

Methods use scoped enum values: `LogSigMethod.Expanded`, `.LyndonWords` (default), `.LyndonBasis`, or `.DirectBCH`. Branched logs use `BranchedLogSigMethod.Expanded`, `.MkwWords`, `.MkwBasis`, or `.DirectBCH`; the default is `Expanded` for nonplanar paths and `MkwWords` for planar paths.

## Documentation

See [`docs/src/index.md`](docs/src/index.md) for supported operations, installation, native artifact updates, and extension conventions. The bundled artifact uses a pinned GitHub checkout with runtime SIMD dispatch and requires Linux x86-64 with glibc 2.38 or newer. SigLib requires the 4.x native ABI.

## License

See [`LICENSE`](LICENSE).
