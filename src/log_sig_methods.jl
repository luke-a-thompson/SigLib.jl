"""
    LogSigMethod

Methods for ordinary log signatures. Pass a value as `method=LogSigMethod.LyndonBasis`.

- `Expanded`: full tensor logarithm (native value 0).
- `LyndonWords`: extracted Lyndon-word coordinates (1, the default).
- `LyndonBasis`: Lyndon-basis coordinates (2).
- `DirectBCH`: compute Lyndon-basis coordinates directly from a path through BCH (3).

Values have type `LogSigMethod.T`. Signature/log-signature conversions do not
accept `DirectBCH`; use `log_sig` or `LogSigWorkspace` for that method.
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
