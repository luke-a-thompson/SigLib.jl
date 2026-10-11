# SigLib.jl

`SigLib.jl` wraps the native cpSIG/cuSIG libraries using checked, version-pinned Julia artifacts. Install this unregistered package with `Pkg.add(url="https://github.com/luke-a-thompson/SigLib.jl")`, or use `Pkg.develop(path=".")` from a local checkout.

The implemented API covers standard and branched signatures, log signatures, conversions, and selected composition operations:

```julia
sig_length(dimension, degree; scalar_term=false)
sig(path, degree; scalar_term=false, n_jobs=1)
log_sig_length(dimension, degree; time_aug=false, lead_lag=false)
prepare_log_sig(dimension, degree, method; time_aug=false, lead_lag=false)
set_cache_dir(directory)
clear_log_sig_cache(; use_disk=false)
sig_to_log_sig(sig, dimension, degree; time_aug=false, lead_lag=false, method=1, n_jobs=1)
logsig_to_sig(log_sig, dimension, degree; time_aug=false, lead_lag=false, method=1, scalar_term=false, n_jobs=1)
log_sig(path, degree; time_aug=false, lead_lag=false, end_time=1.0, method=1, scalar_term=false, n_jobs=1)
log_sig_combine(log_sig1, log_sig2, dimension, degree; n_jobs=1)
log_sig_join(log_sig, displacement, dimension, degree; n_jobs=1)
branched_sig_length(dimension, max_nodes; planar=false, scalar_term=false)
prepare_branched_sig(dimension, max_nodes; planar=false)
branched_log_sig_length(dimension, degree; planar=false)
prepare_branched_log_sig(dimension, degree, method; planar=false)
branched_sig(path, max_nodes; planar=false, scalar_term=false, n_jobs=1)
branched_sig_combine(bsig1, bsig2, dimension, max_nodes; planar=false)
branched_sig_to_log_sig(bsig, dimension, degree; planar=false, method=nothing, n_jobs=1)
branched_log_sig(path, degree; planar=false, scalar_term=false, method=nothing, n_jobs=1)
```

Paths use pySigLib's shape convention: `(length, dimension)` for one path and `(batch..., length, dimension)` for batches. Outputs have shape `(coefficients,)` or `(batch..., coefficients)`. Inputs must contain `Float32` or `Float64`; views are accepted. Binary operations convert the second input to the first input's precision. Degree zero and empty batches are supported, but a path must have at least one point and one channel. The scalar signature term is excluded by default. `correction` is currently unsupported.

For ordinary log signatures, method 0 returns tensor logarithms, method 1 extracts Lyndon-word coordinates, method 2 returns Lyndon-basis coordinates, and method 3 computes Lyndon-basis coordinates directly through BCH. Combine/join expect method-2 or method-3 coordinates. Methods 1–3 use compressed outputs and ignore `scalar_term`.

Branched log signatures follow Python's defaults: method 0 for nonplanar paths and method 1 for planar paths. Method 0 returns the expanded Hopf logarithm; planar methods 1 and 2 return extracted weighted Lyndon coordinates and MKW basis coordinates, respectively. Method 3 computes the same coordinates as method 2 directly through BCH. Conversion from a branched signature supports methods 0–2 and infers the scalar term from the input length; method 0 preserves that format, while compressed outputs are scalar-free. `branched_log_sig_length` returns the nonplanar tree count or the compressed planar length. Methods 1–3 require a 4.x native artifact; with the 3.1 bootstrap, use `planar=true, method=0` for expanded planar logs. Non-empty corrections remain unsupported; empty corrections are accepted by `branched_log_sig`.

Load `ChainRulesCore` to enable reverse-mode rules for the numerical operations. Composed log-signature rules reuse the forward signature. `n_jobs=1` runs native computations serially; `-1` requests all available native threads. Cache-dependent operations are synchronized across Julia tasks, including preparation, clearing, and pullbacks. The native operation can still use multiple threads internally. Cache lifetime is shared within the process; external calls into the same library must obey the same synchronization rules.

In-memory caches prepare automatically. Use `set_cache_dir(directory)` before requesting persistent disk caches with `use_disk=true`; SigLib creates the parent directory. The legacy 3.1 binary uses a temporary directory for its in-memory cache bookkeeping. Upstream 4.x does not need that directory.

## CUDA

Install CUDA.jl 5.x and pass `CuArray` inputs to the same functions. All currently implemented numerical operations and their ChainRules pullbacks run through native cuSIG; packing, results, and gradients stay on the GPU. Load ChainRulesCore when derivatives are needed.

```julia
using SigLib, CUDA, ChainRulesCore
path = CuArray([0.0 0.0; 0.5 1.0; 1.0 -0.25])
sig(path, 3)                              # CuVector
branched_log_sig(path, 3; planar=true)     # CuVector
clear_log_sig_cache(; device=:cuda)
```

Multiple inputs must use the same backend and CUDA context; views are packed on the GPU. Calls activate the input's context and synchronize the device before and after native work because cuSIG manages its own streams and accepts no caller stream. Native calls are synchronous. `n_jobs` controls CPU calls and has no effect on CUDA calls. Upstream limits direct planar branched BCH (method 3) to degree 20 on CUDA.

Cache preparation is automatic on the input device. Explicit cache controls accept `device=:cpu` (default), `:cuda`, or `:both`; CUDA preparation uses the active device; clearing affects native caches across devices. CUDA is optional for CPU users, and loading CUDA.jl alone does not load the cuSIG artifact. GPU support requires the upstream 4.x release manifest described below; the legacy bootstrap supports CPU calls.

Run the separate hardware suite on a machine with a working NVIDIA GPU and the matching artifacts:

```sh
julia --project=test/cuda -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=test/cuda --threads=4 test/cuda/runtests.jl
```

## Native artifacts

The root `Artifacts.toml` currently pins the existing, verified 3.1 bootstrap binary. Import the published upstream manifest with `julia scripts/update_artifacts.jl vVERSION`, then run `julia --project -e 'using Pkg; Pkg.test()'` before committing it. The updater checks artifact hashes, download checksums, and release URLs before replacing the file. CUDA calls load the lazy `pysiglib_cuda12` artifact on first use. CPU and CUDA artifacts must come from the same 4.x release.

The loader accepts the upstream `pysiglib/` layout and the bootstrap `lib/` layout, loads Windows TBB dependencies, and selects the 3.1 or 4.x branched-conversion ABI by native capability. cpSIG 4 changes planar branched-signature coordinates, so stored planar coefficients must stay associated with their native version. Julia artifact overrides can select a local build for development without changing package code.

Upstream CPU artifacts cover Linux/Windows x86-64 and macOS arm64. Their minimum requirements are glibc 2.28 and x86-64-v3 on Linux, AVX2 on Windows, and macOS 11 on Apple Silicon. Julia's platform selector does not enforce the CPU/glibc requirements. See the [upstream artifact documentation](https://github.com/daniil-shmelev/pySigLib/blob/main/docs/julia-artifacts.md) for current release details.

## Extending the wrapper

`library.jl` owns loading, `c_api.jl` owns concrete float/double ABI calls, `buffers.jl` owns validation and array layout, and the numerical files own public operations. Add native bindings using the existing typed loop pattern, use the shared buffers, and hold `_cache_lock` across preparation and native cache use. Add CUDA bindings in `ext/cuda/c_api.jl` and derivatives in the optional ChainRules extension. Test algebraic identities, batch/view equivalence, both precisions, and finite-difference pullbacks where applicable.

Full pySigLib coverage still requires ordinary signature combine/join, branched joins and straight-line operations, coefficient and prefix operations, signature kernels and metrics, streaming, corrections, and word/tree utilities. These can extend the existing layers.
