# Enumeration order follows pySigLib's words.py, trees.py, and native
# branched_log_sig_cache.cpp (Daniil Shmelev, Apache-2.0).
# Portions adapted from pySigLib, Copyright 2026 Daniil Shmelev.
# Licensed under the Apache License, Version 2.0; see LICENSE.

function _basis_parameters(dimension::Integer, degree::Integer)
    dimension >= 0 || throw(ArgumentError("dimension must be nonnegative"))
    degree >= 0 || throw(ArgumentError("degree must be nonnegative"))
    dimension <= typemax(Int) && degree <= typemax(Int) ||
        throw(OverflowError("basis parameters exceed Int"))
    return Int(dimension), Int(degree)
end

function _checked_word(word::Tuple, dimension::Union{Nothing,Integer}=nothing)
    all(x -> x isa Integer && x >= 0 && (dimension === nothing || x < dimension), word) ||
        throw(ArgumentError("word letters must be integers in 0:dimension-1"))
    return word
end

function _scalar_basis(basis, empty, scalar_term::Bool)
    scalar_term || return copy(basis)
    result = Vector{Union{typeof(empty),eltype(basis)}}(undef, length(basis) + 1)
    result[1] = empty
    copyto!(result, 2, basis, 1, length(basis))
    return result
end

"""
    words_of_length(dimension, length; scalar_term=false)

Enumerate words of exactly `length` in lexicographic coefficient order.
Letters are zero-based channel labels, as in pySigLib. Length zero returns
`[()]` with `scalar_term=true` and `[]` otherwise.
"""
function words_of_length(dimension::Integer, length::Integer; scalar_term::Bool=false)
    d, n = _basis_parameters(dimension, length)
    n == 0 && return scalar_term ? Tuple[()] : Tuple[]
    result = Tuple[()]
    for _ in 1:n
        result = Tuple[(prefix..., label) for prefix in result for label in 0:d-1]
        isempty(result) && break
    end
    return result
end

"""
    words(dimension, degree; scalar_term=false)

Enumerate signature and expanded log-signature coordinate labels through `degree`,
ordered by length and then lexicographically. `scalar_term=true` adds the empty
word `()` first. Increasing `degree` preserves existing indices.
"""
function words(dimension::Integer, degree::Integer; scalar_term::Bool=false)
    d, n = _basis_parameters(dimension, degree)
    result = scalar_term ? Tuple[()] : Tuple[]
    d == 0 && return result
    for order in 1:n
        append!(result, words_of_length(d, order))
    end
    return result
end

_is_lyndon(word::Tuple) = !isempty(word) && all(i -> isless(word, word[i:end]), 2:length(word))

"""
    is_lyndon(word::Tuple)

Test whether a nonnegative integer word is strictly smaller than every proper suffix.
The empty word is not Lyndon.
"""
is_lyndon(word::Tuple) = _is_lyndon(_checked_word(word))

"""
    lyndon_words_of_length(dimension, length)

Enumerate Lyndon words of exactly `length` in lexicographic order.
"""
function lyndon_words_of_length(dimension::Integer, length::Integer)
    d, n = _basis_parameters(dimension, length)
    result = Tuple[]
    (d == 0 || n == 0) && return result
    # Duval's enumeration avoids constructing all d^n tensor words.
    word = [0]
    while !isempty(word)
        period = Base.length(word)
        period == n && push!(result, Tuple(word))
        while Base.length(word) < n
            push!(word, word[end - period + 1])
        end
        while !isempty(word) && word[end] == d - 1
            pop!(word)
        end
        isempty(word) || (word[end] += 1)
    end
    return result
end

"""
    lyndon_words(dimension, degree)

Return the labels of ordinary `LogSigMethod.LyndonWords` coordinates, ordered by length then
lexicographically. There is no scalar entry.
"""
function lyndon_words(dimension::Integer, degree::Integer)
    d, n = _basis_parameters(dimension, degree)
    result = Tuple[]
    d == 0 && return result
    for order in 1:n
        append!(result, lyndon_words_of_length(d, order))
    end
    return result
end

const _lyndon_word_enumerations = Dict{Tuple{Int,Int},NamedTuple}()

function _lyndon_word_enumeration(dimension::Integer, degree::Integer)
    d, n = _basis_parameters(dimension, degree)
    return lock(_cache_lock) do
        get!(_lyndon_word_enumerations, (d, n)) do
            basis = lyndon_words(d, n)
            return (basis=basis, indices=Dict(w => i for (i, w) in enumerate(basis)))
        end
    end
end

function _lyndon_bracket(word::Tuple, letters)
    length(word) == 1 && return letters(word[1])
    for split in 2:length(word)
        suffix = word[split:end]
        if _is_lyndon(suffix)
            return (:bracket, _lyndon_bracket(word[1:split-1], letters),
                _lyndon_bracket(suffix, letters))
        end
    end
    error("Lyndon word has no standard factorization")
end

"""
    lyndon_bracket(word::Tuple)

Return the standard Lyndon bracketing using the longest proper Lyndon suffix.
A letter is an integer; a commutator is `(:bracket, left, right)`, interpreted
as `left * right - right * left` under word concatenation. For example `(0, 0, 1)` becomes
`(:bracket, 0, (:bracket, 0, 1))`.
"""
function lyndon_bracket(word::Tuple)
    is_lyndon(word) || throw(ArgumentError("word must be a nonempty Lyndon word"))
    return _lyndon_bracket(word, identity)
end

"""
    lyndon_brackets(dimension, degree)

Return the standard bracket expressions labelling ordinary `LyndonBasis`/`DirectBCH` log coordinates.
"""
lyndon_brackets(dimension::Integer, degree::Integer) =
    map(lyndon_bracket, lyndon_words(dimension, degree))

function _basis_index(index::Integer, count::Integer)
    1 <= index <= count ||
        throw(ArgumentError("index is out of range for the requested basis"))
    return Int(index)
end

"""
    word_to_idx(word, dimension, degree; scalar_term=false)

Return the **one-based Julia index** of a zero-labelled word in a signature or
expanded tensor logarithm truncated at `degree`. The empty word requires
`scalar_term=true`. Inverse of [`idx_to_word`](@ref).
"""
function word_to_idx(word::Tuple, dimension::Integer, degree::Integer; scalar_term::Bool=false)
    d, n = _basis_parameters(dimension, degree)
    _checked_word(word, d)
    length(word) <= n || throw(ArgumentError("word exceeds the requested degree"))
    isempty(word) && !scalar_term && throw(ArgumentError("empty word requires scalar_term=true"))
    index = 0
    for letter in word
        index = Base.Checked.checked_add(Base.Checked.checked_mul(index, d), Int(letter) + 1)
    end
    return scalar_term ? Base.Checked.checked_add(index, 1) : index
end

# Number of nonempty words through degree n, saturated at typemax(Int).
function _word_count(d::Int, n::Int)
    count, power = 0, 1
    for _ in 1:n
        power = d == 0 ? 0 : (power > typemax(Int) ÷ d ? typemax(Int) : power * d)
        count = count > typemax(Int) - power ? typemax(Int) : count + power
        count == typemax(Int) && break
    end
    return count
end

"""
    idx_to_word(index, dimension, degree; scalar_term=false)

Return the word at a one-based Julia index; inverse of [`word_to_idx`](@ref) and
equal to `words(dimension, degree; scalar_term)[index]`.
"""
function idx_to_word(index::Integer, dimension::Integer, degree::Integer; scalar_term::Bool=false)
    d, n = _basis_parameters(dimension, degree)
    count = _word_count(d, n)
    code = _basis_index(index, scalar_term ? (count == typemax(Int) ? count : count + 1) : count) - Int(scalar_term)
    code == 0 && return ()
    letters = Int[]
    while code > 0
        code, letter = divrem(code - 1, d)
        push!(letters, letter)
    end
    return Tuple(reverse!(letters))
end

"""
    lyndon_word_to_idx(word, dimension, degree)

Return the **one-based Julia index** of a Lyndon word in compressed
`LogSigMethod.LyndonWords`, `LyndonBasis`, or `DirectBCH` coordinates. Compressed
and full ([`word_to_idx`](@ref)) indices differ. There is no scalar coordinate.
"""
function lyndon_word_to_idx(word::Tuple, dimension::Integer, degree::Integer)
    d, n = _basis_parameters(dimension, degree)
    _checked_word(word, d)
    length(word) <= n && _is_lyndon(word) ||
        throw(ArgumentError("word is not in the requested Lyndon-word basis"))
    return _lyndon_word_enumeration(d, n).indices[word]
end

"""
    idx_to_lyndon_word(index, dimension, degree)

Return the Lyndon word at a one-based compressed index; inverse of
[`lyndon_word_to_idx`](@ref) and equal to `lyndon_words(dimension, degree)[index]`.
"""
function idx_to_lyndon_word(index::Integer, dimension::Integer, degree::Integer)
    basis = _lyndon_word_enumeration(dimension, degree).basis
    return basis[_basis_index(index, length(basis))]
end

# Stable tree IDs are defined by order, child-ID sequence, then root label.
# Comparing nested tree tuples directly would give the wrong Lyndon alphabet.
const _tree_enumerations = Dict{Tuple{Int,Int,Bool},NamedTuple}()
const _rooted_enumerations = Dict{Tuple{Int,Int,Bool},NamedTuple}()

function _foreach_forest(visit, weights, target::Int, planar::Bool)
    current = Int[]
    function enumerate(remaining, minimum)
        if remaining == 0
            visit(Tuple(current))
            return
        end
        for id in (planar ? 1 : minimum):length(weights)
            weights[id] > remaining && break
            push!(current, id)
            enumerate(remaining - weights[id], id)
            pop!(current)
        end
    end
    enumerate(target, 1)
end

function _rooted_enumeration(dimension::Integer, degree::Integer, planar::Bool)
    d, n = _basis_parameters(dimension, degree)
    d <= 255 || throw(ArgumentError("tree dimension must be <= 255"))
    return lock(_cache_lock) do
        get!(_rooted_enumerations, (d, n, planar)) do
            rooted = Tuple[]
            weights = Int[]
            if d > 0
                for order in 1:n
                    # Only trees of smaller order may appear as children.
                    previous_weights = copy(weights)
                    _foreach_forest(previous_weights, order - 1, planar) do children
                        for label in 0:d-1
                            push!(rooted, (map(id -> rooted[id], children)..., label))
                            push!(weights, order)
                        end
                    end
                end
            end
            return (rooted=rooted, weights=weights)
        end
    end
end

function _tree_enumeration(dimension::Integer, degree::Integer, planar::Bool)
    d, n = _basis_parameters(dimension, degree)
    return lock(_cache_lock) do
        get!(_tree_enumerations, (d, n, planar)) do
            rooted, weights = _rooted_enumeration(d, n, planar)
            forest_words = Tuple[]
            if planar
                for order in 1:n
                    _foreach_forest(ids -> push!(forest_words, ids), weights, order, true)
                end
                basis = Tuple[map(id -> rooted[id], ids) for ids in forest_words]
            else
                basis = copy(rooted)
            end
            orders = planar ? Int[sum(id -> weights[id], ids) for ids in forest_words] : copy(weights)
            lyndon = planar ? findall(_is_lyndon, forest_words) : Int[]
            return (rooted=rooted, basis=basis, orders=orders, forest_words=forest_words,
                indices=Dict(t => i for (i, t) in enumerate(basis)), lyndon=lyndon,
                lyndon_basis=basis[lyndon], lyndon_indices=Dict(basis[k] => i for (i, k) in enumerate(lyndon)))
        end
    end
end

"""
    rooted_trees(dimension, degree; planar=false)

Enumerate decorated rooted trees through `degree`, in native tree-ID order.
A leaf is `(label,)`; a tree is `(child1, child2, ..., root_label)` with zero-based
labels. Nonplanar children are sorted by native tree ID; planar children retain order.
Planar rooted trees are the letters of [`forests`](@ref) and [`lyndon_forests`](@ref).
"""
rooted_trees(dimension::Integer, degree::Integer; planar::Bool=false) =
    copy(_rooted_enumeration(dimension, degree, planar).rooted)

"""
    trees(dimension, degree; scalar_term=false)

Enumerate nonplanar BCK trees through `degree` nodes, the coordinate labels of
nonplanar branched signatures and log signatures. The order is node count,
native child-ID sequence, then root label. `scalar_term=true` adds `nothing`
first. Increasing `degree` preserves existing indices. Use [`forests`](@ref)
for planar MKW coordinates.
"""
trees(dimension::Integer, degree::Integer; scalar_term::Bool=false) =
    _scalar_basis(_tree_enumeration(dimension, degree, false).basis, nothing, scalar_term)

"""
    trees_of_order(dimension, order; scalar_term=false)

Return BCK trees of exactly `order` nodes. Order zero returns `[nothing]` with
`scalar_term=true` and `[]` otherwise.
"""
function trees_of_order(dimension::Integer, order::Integer; scalar_term::Bool=false)
    enumeration = _tree_enumeration(dimension, order, false)
    order == 0 && return scalar_term ? Union{Nothing,Tuple}[nothing] : Tuple[]
    return Tuple[t for (t, n) in zip(enumeration.basis, enumeration.orders) if n == order]
end

function _validate_tree_label(tree, dimension)
    tree isa Tuple && !isempty(tree) || throw(ArgumentError("each tree must be a nonempty tuple"))
    tree[end] isa Integer && 0 <= tree[end] < dimension ||
        throw(ArgumentError("tree labels must be integers in 0:dimension-1"))
    for child in tree[1:end-1]
        _validate_tree_label(child, dimension)
    end
    return nothing
end

function _validate_forest(forest, dimension)
    forest isa Tuple || throw(ArgumentError("forest must be a tuple of trees"))
    foreach(t -> _validate_tree_label(t, dimension), forest)
    return nothing
end

function _element_index(indices, element, scalar_term::Bool)
    index = get(indices, element, 0)
    index > 0 || throw(ArgumentError("element is not in the requested coordinate basis"))
    return index + Int(scalar_term)
end

function _index_element(basis, index::Integer, empty, scalar_term::Bool)
    i = _basis_index(index, length(basis) + Int(scalar_term))
    scalar_term && i == 1 && return empty
    return basis[i - Int(scalar_term)]
end

"""
    tree_to_idx(tree, dimension, degree; scalar_term=false)

Return the **one-based Julia index** of a BCK tree in a nonplanar branched
signature or log signature. Children must be in the canonical native order
returned by [`trees`](@ref). `nothing` requires `scalar_term=true`.
"""
function tree_to_idx(tree, dimension::Integer, degree::Integer; scalar_term::Bool=false)
    enumeration = _tree_enumeration(dimension, degree, false)
    if tree === nothing
        scalar_term || throw(ArgumentError("empty tree requires scalar_term=true"))
        return 1
    end
    _validate_tree_label(tree, dimension)
    return _element_index(enumeration.indices, tree, scalar_term)
end

"""
    idx_to_tree(index, dimension, degree; scalar_term=false)

Return the BCK tree at a one-based Julia index; inverse of [`tree_to_idx`](@ref)
and equal to `trees(dimension, degree; scalar_term)[index]`.
"""
idx_to_tree(index::Integer, dimension::Integer, degree::Integer; scalar_term::Bool=false) =
    _index_element(_tree_enumeration(dimension, degree, false).basis, index, nothing, scalar_term)

"""
    forests(dimension, degree; scalar_term=false)

Enumerate ordered planar MKW forests through `degree` total nodes, the
coordinate labels of planar branched signatures and expanded MKW log
signatures. Each forest is a tuple `(tree1, tree2, ...)` of planar trees, ordered
by total node count and then lexicographically by native tree IDs.
`scalar_term=true` adds the empty forest `()` first.
"""
forests(dimension::Integer, degree::Integer; scalar_term::Bool=false) =
    _scalar_basis(_tree_enumeration(dimension, degree, true).basis, (), scalar_term)

"""
    forest_to_idx(forest, dimension, degree; scalar_term=false)

Return the **one-based Julia index** of an ordered planar forest in full MKW
coordinates. A one-tree forest is the singleton tuple `(tree,)`. The empty forest
`()` requires `scalar_term=true`. Full and compressed
([`lyndon_forest_to_idx`](@ref)) indices differ.
"""
function forest_to_idx(forest, dimension::Integer, degree::Integer; scalar_term::Bool=false)
    enumeration = _tree_enumeration(dimension, degree, true)
    _validate_forest(forest, dimension)
    if isempty(forest)
        scalar_term || throw(ArgumentError("empty forest requires scalar_term=true"))
        return 1
    end
    return _element_index(enumeration.indices, forest, scalar_term)
end

"""
    idx_to_forest(index, dimension, degree; scalar_term=false)

Return the ordered forest at a one-based full MKW index; inverse of
[`forest_to_idx`](@ref) and equal to `forests(dimension, degree; scalar_term)[index]`.
"""
idx_to_forest(index::Integer, dimension::Integer, degree::Integer; scalar_term::Bool=false) =
    _index_element(_tree_enumeration(dimension, degree, true).basis, index, (), scalar_term)

"""
    lyndon_forests(dimension, degree)

Enumerate weighted Lyndon forests, the labels of compressed MKW log signatures.
Each planar tree is a letter ordered by native tree ID with weight equal to its
node count; `degree` bounds the total weight. Order is total weight, then
lexicographic tree-ID order. The empty forest is excluded.
"""
lyndon_forests(dimension::Integer, degree::Integer) =
    copy(_tree_enumeration(dimension, degree, true).lyndon_basis)

"""
    lyndon_forest_to_idx(forest, dimension, degree)

Return the **one-based Julia index** of a weighted Lyndon forest in compressed
MKW coordinates. There is no scalar coordinate.
"""
function lyndon_forest_to_idx(forest, dimension::Integer, degree::Integer)
    enumeration = _tree_enumeration(dimension, degree, true)
    _validate_forest(forest, dimension)
    isempty(forest) && throw(ArgumentError("the empty forest has no compressed index"))
    return _element_index(enumeration.lyndon_indices, forest, false)
end

"""
    idx_to_lyndon_forest(index, dimension, degree)

Return the weighted Lyndon forest at a one-based compressed index; inverse of
[`lyndon_forest_to_idx`](@ref) and equal to `lyndon_forests(dimension, degree)[index]`.
"""
function idx_to_lyndon_forest(index::Integer, dimension::Integer, degree::Integer)
    basis = _tree_enumeration(dimension, degree, true).lyndon_basis
    return basis[_basis_index(index, length(basis))]
end

"""
    log_sig_basis(dimension, degree; method=LogSigMethod.LyndonWords, scalar_term=false)

Return coordinate labels for ordinary log signatures using [`LogSigMethod`](@ref):
tensor words for `Expanded`, Lyndon words for `LyndonWords`, standard bracket
expressions for `LyndonBasis` and `DirectBCH`.
The default is `LyndonWords`, matching `log_sig`. Compressed methods ignore
`scalar_term`. Pass the effective (augmented) dimension.
"""
function log_sig_basis(dimension::Integer, degree::Integer;
    method::LogSigMethod.T=LogSigMethod.LyndonWords, scalar_term::Bool=false)
    method == LogSigMethod.Expanded && return words(dimension, degree; scalar_term)
    return method == LogSigMethod.LyndonWords ? lyndon_words(dimension, degree) : lyndon_brackets(dimension, degree)
end

"""
    branched_log_sig_basis(dimension, degree; planar=false, method=planar ? BranchedLogSigMethod.MkwWords : BranchedLogSigMethod.Expanded, scalar_term=false)

Return labels in branched-log output order using [`BranchedLogSigMethod`](@ref).
`Expanded` uses BCK trees or MKW ordered forests. `MkwWords` uses weighted
Lyndon forests (letters are native planar tree IDs, weight is node count);
`MkwBasis` and `DirectBCH` use their standard bracketings
`(:bracket, left, right)`, with tree tuples as generators.
These symbolic expressions use the forest-concatenation commutator of the
native MKW coordinate projection.
The default is `MkwWords` for planar logs and `Expanded` otherwise, matching `branched_log_sig`.
Compressed methods ignore `scalar_term` and require planar=true.
Pass the effective (augmented) dimension.
"""
function branched_log_sig_basis(dimension::Integer, degree::Integer;
    planar::Bool=false, method::BranchedLogSigMethod.T=planar ? BranchedLogSigMethod.MkwWords : BranchedLogSigMethod.Expanded, scalar_term::Bool=false)
    method == BranchedLogSigMethod.Expanded || planar || throw(ArgumentError("compressed branched log bases require planar=true"))
    if method == BranchedLogSigMethod.Expanded
        return planar ? forests(dimension, degree; scalar_term) : trees(dimension, degree; scalar_term)
    end
    method == BranchedLogSigMethod.MkwWords && return lyndon_forests(dimension, degree)
    enumeration = _tree_enumeration(dimension, degree, true)
    return [_lyndon_bracket(enumeration.forest_words[i], id -> enumeration.rooted[id]) for i in enumeration.lyndon]
end
