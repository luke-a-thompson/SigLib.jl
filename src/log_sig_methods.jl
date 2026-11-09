"""
    SigMethod

Algorithms for tensor signatures and their gradients. Pass a value as
`method=SigMethod.PrefixSuffix`.

- `Horner`: Horner's scheme (native value 0).
- `PrefixSuffix`: prefix-suffix factorisation (1).
- `Auto`: select an algorithm for each pass (2, the default).

Values have type `SigMethod.T`. `ps_split` optionally fixes the maximum prefix
degree of `PrefixSuffix`, from 1 to `degree`.
"""
module SigMethod
    @enum T::Int32 Horner=0 PrefixSuffix=1 Auto=2
end

function _ps_split(method::SigMethod.T, ps_split::Union{Nothing,Integer}, degree::Integer)
    ps_split === nothing && return UInt64(0)
    method == SigMethod.PrefixSuffix || throw(ArgumentError("ps_split requires method=SigMethod.PrefixSuffix"))
    1 <= ps_split <= degree || throw(ArgumentError("ps_split must be between 1 and degree"))
    return UInt64(ps_split)
end

"""
    LogSigMethod

Methods for ordinary log signatures. Pass a value as `method=LogSigMethod.LyndonBasis`.

- `Expanded`: full tensor logarithm (native value 0).
- `LyndonWords`: extracted Lyndon-word coordinates (1, the default).
- `LyndonBasis`: Lyndon-basis coordinates (2).
- `DirectBCH`: compute Lyndon-basis coordinates directly from a path through BCH (3).

Values have type `LogSigMethod.T`. Signature/log-signature conversions do not
accept `DirectBCH`; use `log_sig` or `log_sig!` for that method.
"""
module LogSigMethod
    @enum T::Int32 Expanded=0 LyndonWords=1 LyndonBasis=2 DirectBCH=3
end

"""
    BranchedLogSigMethod

Methods for branched log signatures. Pass a value as
`method=BranchedLogSigMethod.MkwBasis`.

- `Expanded`: full Hopf logarithm (native value 0, the nonplanar default).
- `MkwWords`: extracted weighted Lyndon coordinates (1, the planar default).
- `MkwBasis`: MKW-basis coordinates (2).
- `DirectBCH`: compute MKW-basis coordinates directly from a path through BCH (3).

Values have type `BranchedLogSigMethod.T`. All methods except `Expanded` require
`planar=true`. Conversions from branched signatures do not accept `DirectBCH`.
"""
module BranchedLogSigMethod
    @enum T::Int32 Expanded=0 MkwWords=1 MkwBasis=2 DirectBCH=3
end
