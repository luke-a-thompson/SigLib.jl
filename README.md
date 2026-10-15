# SigLib.jl

**SigLib.jl** calls [pySigLib's native cpSIG/cuSIG libraries](https://github.com/daniil-shmelev/pySigLib/tree/main/siglib) through Julia artifacts. Python and a compiler are unnecessary. The interface uses Julia's channel-first, batch-last arrays and pySigLib's signature conventions, with optional CUDA and ChainRules extensions.

```julia
using SigLib
path = [0.0 0.5 1.0; 0.0 1.0 -0.25] # (dimension, length)
sig(path, 2)
log_sig(path, 2; method=2)
workspace = LogSigWorkspace(path, 2; method=2)
log_sig!(workspace, path) # reuse buffers for subsequent paths of the same shape
branched_sig(path, 2)
sig(path, 3; correction=[-0.01, 0.0, 0.0, -0.01]) # local level-2 correction
lyndon_brackets(2, 3) # standard brackets in method-2/3 coordinate order
trees(2, 3) # BCK trees; planar=true gives MKW ordered forests
branched_log_sig_basis(2, 3; planar=true, method=2) # compressed MKW brackets
```

## Documentation

See [`docs/src/index.md`](docs/src/index.md) for supported operations, installation, native artifact updates, and extension conventions. The bootstrap artifact currently supports Linux x86-64 with glibc; upstream release manifests also provide Windows x86-64 and Apple Silicon macOS binaries.

## License

See [`LICENSE`](LICENSE).
