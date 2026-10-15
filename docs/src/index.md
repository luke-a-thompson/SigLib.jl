# SigLib.jl

`SigLib.jl` wraps the native cpSIG/cuSIG libraries using checked, version-pinned Julia artifacts. Install this unregistered package with `Pkg.add(url="https://github.com/luke-a-thompson/SigLib.jl")`, or use `Pkg.develop(path=".")` from a local checkout.

The implemented API covers standard and branched signatures, log signatures, conversions, and selected composition operations:

```julia
sig_length(dimension, degree; scalar_term=false)
sig(path, degree; scalar_term=false, correction=nothing, n_jobs=1)
log_sig_length(dimension, degree; time_aug=false, lead_lag=false)
prepare_log_sig(dimension, degree, method; time_aug=false, lead_lag=false)
set_cache_dir(directory)
clear_log_sig_cache(; use_disk=false)
sig_to_log_sig(sig, dimension, degree; time_aug=false, lead_lag=false, method=1, n_jobs=1)
logsig_to_sig(log_sig, dimension, degree; time_aug=false, lead_lag=false, method=1, scalar_term=false, n_jobs=1)
log_sig(path, degree; time_aug=false, lead_lag=false, end_time=1.0, method=1, scalar_term=false, correction=nothing, n_jobs=1)
log_sig_combine(log_sig1, log_sig2, dimension, degree; n_jobs=1)
log_sig_join(log_sig, displacement, dimension, degree; n_jobs=1)
branched_sig_length(dimension, max_nodes; planar=false, scalar_term=false)
prepare_branched_sig(dimension, max_nodes; planar=false)
branched_log_sig_length(dimension, degree; planar=false)
prepare_branched_log_sig(dimension, degree, method; planar=false)
branched_sig(path, max_nodes; planar=false, scalar_term=false, correction=nothing, n_jobs=1)
branched_sig_combine(bsig1, bsig2, dimension, max_nodes; planar=false)
branched_sig_to_log_sig(bsig, dimension, degree; planar=false, method=nothing, n_jobs=1)
branched_log_sig(path, degree; planar=false, scalar_term=false, method=nothing, correction=nothing, n_jobs=1)
```

Paths use `(dimension, length)` for one path and `(dimension, length, batch...)` for batches. Outputs have shape `(coefficients,)` or `(coefficients, batch...)`. Channels and coefficients come first; batch dimensions come last. This convention applies to CPU and CUDA arrays, conversions, composition operations, displacements, and gradients. Dense inputs go directly to the native library, and outputs share native storage without layout conversions. Inputs must contain `Float32` or `Float64`; views are accepted. Binary operations convert the second input to the first input's precision. Degree zero and empty batches are supported, but a path must have at least one point and one channel. The scalar signature term is excluded by default.

This changes the earlier Python-style axis order. To migrate existing paths or corrections while preserving the order of multiple batch dimensions, use `permutedims(x, (ndims(x), ndims(x) - 1, (1:ndims(x)-2)...))`. For batched coefficients or displacements, use `permutedims(x, (ndims(x), (1:ndims(x)-1)...))`; constant vectors need no change. New code should create arrays in the documented order directly.

`sig`, `log_sig`, `branched_sig`, and `branched_log_sig` accept `correction=nothing`, following [pySigLib's correction API](https://github.com/daniil-shmelev/pySigLib/blob/main/pysiglib/sig.py). Corrections are added to each local path increment before taking its exponential, and the segment signatures are composed along the path. Supply a vector `(C,)` for a constant correction, a matrix `(C, length - 1)` for per-segment corrections shared across the batch, or an array `(C, length - 1, batch...)` for batch-specific corrections. The width must be `C = d^2 + d^3 + ... + d^m`, where `d` is the original path dimension and `2 <= m <= degree` (or `max_nodes`); missing higher levels are zero. Levels are concatenated in increasing order, with words in lexicographic order and the last index varying fastest. Branched corrections use the corresponding root-to-leaf chain trees.

Corrections must have the same eltype, backend, and CUDA context as the path; views are accepted. `nothing` and empty arrays mean no correction. A time channel added by `time_aug=true` receives no correction entries. Non-empty corrections cannot be combined with `lead_lag=true` or direct BCH `method=3`. For non-Lie corrections, use log-signature `method=0` to retain the full tensor or Hopf logarithm. ChainRules returns path gradients with corrections held constant.

Native cpSIG 3.1 and 4.0 have a nonplanar branched-signature discrepancy at degree 3: supplying a non-empty all-zero correction changes some branching-tree coefficients compared with omitting the correction. SigLib forwards the native results, matching pySigLib; regression tests record this discrepancy. Ordinary tensor signatures pass the zero-correction identity and an independent tensor-series reference.

For example, the Ito lift for a 2D Brownian path with identity covariance uses the local level-2 correction `-0.5 * dt * I`, flattened in lexicographic order:

```julia
path = [0.0 0.1 0.05; 0.0 -0.2 0.3]
dt = 0.5
correction = [-0.5dt, 0.0, 0.0, -0.5dt] # constant on every segment
s = sig(path, 3; correction=correction)
l = log_sig(path, 3; correction=correction, method=0)
rows = repeat(reshape(correction, :, 1), 1, size(path, 2) - 1)
sig(path, 3; correction=rows) # same correction supplied per segment
```

For ordinary log signatures, method 0 returns tensor logarithms, method 1 extracts Lyndon-word coordinates, method 2 returns Lyndon-basis coordinates, and method 3 computes Lyndon-basis coordinates directly through BCH. Combine/join expect method-2 or method-3 coordinates. Methods 1–3 use compressed outputs and ignore `scalar_term`.

For repeated CPU or CUDA calls of the same shape and precision, `LogSigWorkspace` reuses the intermediate signature and output buffers; dense paths are passed directly to the native library. It supports `Float32` and `Float64`, views, and methods 0–3 with the usual array shapes. The workspace API computes untransformed paths without corrections or a scalar term; use `log_sig` for those options. Degree and method are fixed at construction, and `n_jobs` can be changed per call.

```julia
path = randn(3, 32, 4)
workspace = LogSigWorkspace(path, 5; method=1)
result = log_sig!(workspace, path)
saved = copy(result)             # retain a result across later calls
path .*= 0.7
log_sig!(workspace, path)         # overwrites result; saved keeps the previous value
```

The returned array belongs to the workspace and is overwritten on each call. Use a separate workspace for each concurrent computation. Native cache preparation and clearing remain synchronized; a workspace remains usable after `clear_log_sig_cache()`. The warmed call allocates no Julia memory for dense rank-3 CPU paths; CUDA calls reuse device storage.

Branched log signatures follow Python's defaults: method 0 for nonplanar paths and method 1 for planar paths. Method 0 returns the expanded Hopf logarithm; planar methods 1 and 2 return extracted weighted Lyndon coordinates and MKW basis coordinates, respectively. Method 3 computes the same coordinates as method 2 directly through BCH. Conversion from a branched signature supports methods 0–2 and infers the scalar term from the input length; method 0 preserves that format, while compressed outputs are scalar-free. `branched_log_sig_length` returns the nonplanar tree count or the compressed planar length. Methods 1–3 require a 4.x native artifact; with the 3.1 bootstrap, use `planar=true, method=0` for expanded planar logs.

Load `ChainRulesCore` to enable reverse-mode rules for the allocating numerical operations. Composed log-signature rules reuse the forward signature. `n_jobs=1` runs native computations serially; `-1` requests all available native threads. Cache-dependent operations are synchronized across Julia tasks, including preparation, clearing, and pullbacks. The native operation can still use multiple threads internally. Cache lifetime is shared within the process; external calls into the same library must obey the same synchronization rules.

In-memory caches prepare automatically. Use `set_cache_dir(directory)` before requesting persistent disk caches with `use_disk=true`; SigLib creates the parent directory. The legacy 3.1 binary uses a temporary directory for its in-memory cache bookkeeping. Upstream 4.x does not need that directory.

## Words, trees, and log bases

The enumeration utilities run in Julia and need no Python installation or native cache preparation. They follow [pySigLib's word utilities](https://github.com/daniil-shmelev/pySigLib/blob/main/pysiglib/words.py) and [tree utilities](https://github.com/daniil-shmelev/pySigLib/blob/main/pysiglib/trees.py). SigLib additionally exposes the standard Lyndon bracketings and the compressed MKW basis labels constructed internally by pySigLib.

```julia
words(dimension, degree)                         # includes the empty word ()
words_of_length(dimension, length)
lyndon_words(dimension, degree)                  # no empty word
lyndon_words_of_length(dimension, length)
is_lyndon(word)
lyndon_bracket(word)
lyndon_brackets(dimension, degree)
word_to_idx(word, dimension; scalar_term=false)
idx_to_word(index, dimension; scalar_term=false)
trees(dimension, degree; planar=false)           # includes nothing for the scalar
trees_of_order(dimension, order; planar=false)
rooted_trees(dimension, degree; planar=false)    # individual trees, excludes empty
tree_to_idx(tree, dimension, degree; planar=false, scalar_term=false)
idx_to_tree(index, dimension, degree; planar=false, scalar_term=false)
log_sig_basis(dimension, degree; method=1, scalar_term=false)
branched_log_sig_basis(dimension, degree; planar=false, method=nothing, scalar_term=false)
```

**Channel labels are zero-based; coefficient indices are one-based Julia indices.** Thus `(0,)` labels the first channel, and `word_to_idx((0,), 2)` returns `1`. For time augmentation or lead-lag, pass the effective dimension used by the numerical output. Utilities return fresh vectors; changing a returned vector does not change later enumerations.

Trees use nested tuples: a leaf is `(label,)`, and an internal node is `(child1, child2, ..., root_label)`. Nonplanar children appear in increasing native tree-ID order; use that order when passing a tree to `tree_to_idx`. With `planar=true`, `trees` enumerates ordered forests `(tree1, tree2, ...)`, matching MKW signature coordinates. `rooted_trees(...; planar=true)` enumerates individual planar rooted trees. The scalar label is `nothing` in both cases, and `tree_to_idx(nothing, ...; scalar_term=true)` returns `1`.

```julia
path = [0.0 0.5 1.0; 0.0 1.0 -0.25]
bs = branched_sig(path, 3)
bs[tree_to_idx(((1,), 0), 2, 3)] # root 0, child 1; tensor word (1, 0)

lyndon_words(2, 3)     # [(0,), (1,), (0, 1), (0, 0, 1), (0, 1, 1)]
lyndon_bracket((0, 0, 1)) # (:bracket, 0, (:bracket, 0, 1))
labels = log_sig_basis(2, 3; method=2)
collect(zip(labels, log_sig(path, 3; method=2)))

# Expanded BCK logs are indexed by trees.
collect(zip(branched_log_sig_basis(2, 3), branched_log_sig(path, 3)))

# Compressed MKW logs: numerical methods 1–3 require the 4.x artifact.
mkw_labels = branched_log_sig_basis(2, 3; planar=true, method=2)
```

A bracket expression `(:bracket, left, right)` means the standard commutator `left * right - right * left`; integers are ordinary letter generators, and tree tuples are MKW generators. Products in these symbolic coordinate labels use word concatenation for ordinary logs and forest concatenation for MKW, matching the native coordinate projection. Ordinary `log_sig_basis` returns tensor words for method 0, Lyndon words for method 1, and bracket expressions for methods 2–3. Its default method is 1, matching `log_sig`; choose the same method for your labels and coefficients.

`branched_log_sig_basis` follows the numerical branched-log defaults: method 0 for BCK and method 1 for MKW. Method 0 labels the expanded Hopf log with trees/forests; planar method 1 labels extracted coordinates with weighted Lyndon forests; methods 2–3 label the corresponding standard bracketings. Each planar rooted tree is a letter ordered by its native tree ID, with weight equal to its node count. The truncation bounds total node count. BCK logs retain one coordinate per nonplanar tree. Compressed basis enumeration works with either native version; computing compressed MKW coefficients requires 4.x.

Enumeration materializes the requested basis, so memory grows with the number of words, trees, or forests. Tree enumerations and index maps are cached in Julia for reuse.

## CUDA

Install CUDA.jl 5.x and pass `CuArray` inputs to the numerical functions or `LogSigWorkspace`. The allocating functions and their ChainRules pullbacks run through native cuSIG; results and gradients stay on the GPU. Load ChainRulesCore when derivatives are needed.

```julia
using SigLib, CUDA, ChainRulesCore
path = CuArray([0.0 0.5 1.0; 0.0 1.0 -0.25])
sig(path, 3)                              # CuVector
branched_log_sig(path, 3; planar=true)     # CuVector
batch = CUDA.randn(Float32, 3, 32, 4)     # 4 paths, 32 points, 3 channels
log_sig(batch, 5)                         # CuMatrix: (coefficients, 4)
workspace = LogSigWorkspace(batch, 5)
log_sig!(workspace, batch)                # reuse intermediate and output storage
clear_log_sig_cache(; device=:cuda)
```

Multiple inputs must use the same backend and CUDA context; noncontiguous views are materialized on the GPU; dense inputs and outputs need no layout copies. Calls activate the input's context and synchronize the device before and after native work because cuSIG manages its own streams and accepts no caller stream. Native calls are synchronous. `n_jobs` controls CPU calls and has no effect on CUDA calls. Upstream limits direct planar branched BCH (method 3) to degree 20 on CUDA.

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

Full pySigLib coverage still requires ordinary signature combine/join, branched joins and straight-line operations, coefficient and prefix operations, signature kernels and metrics, and streaming. These can extend the existing layers.
