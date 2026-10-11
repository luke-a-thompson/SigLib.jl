# SigLib.jl

**SigLib.jl** calls [pySigLib's native cpSIG library](https://github.com/daniil-shmelev/pySigLib/tree/main/siglib) through Julia artifacts. Python and a compiler are unnecessary. The interface follows pySigLib's array shapes and signature conventions, with optional ChainRules derivatives.

```julia
using SigLib
path = [0.0 0.0; 0.5 1.0; 1.0 -0.25] # (length, dimension)
sig(path, 2)
log_sig(path, 2; method=2)
branched_sig(path, 2)
```

## Documentation

See [`docs/src/index.md`](docs/src/index.md) for supported operations, installation, native artifact updates, and extension conventions. The bootstrap artifact currently supports Linux x86-64 with glibc; upstream release manifests also provide Windows x86-64 and Apple Silicon macOS binaries.

## License

See [`LICENSE`](LICENSE).
