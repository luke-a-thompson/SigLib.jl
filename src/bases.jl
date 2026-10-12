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

"""
    words_of_length(dimension, length)

Enumerate words of exactly `length` in lexicographic coefficient order.
Letters are zero-based channel labels, as in pySigLib. Length zero returns `[()]`.
"""
function words_of_length(dimension::Integer, length::Integer)
    d, n = _basis_parameters(dimension, length)
    result = Tuple[()]
    for _ in 1:n
        result = Tuple[(prefix..., label) for prefix in result for label in 0:d-1]
        isempty(result) && break
    end
    return result
end

"""
    words(dimension, degree)

Enumerate words through `degree`, including the empty word `()`, in signature order.
"""
function words(dimension::Integer, degree::Integer)
    d, n = _basis_parameters(dimension, degree)
    result = Tuple[()]
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

Return the labels of ordinary method-1 log coordinates, ordered by length then
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

Return the standard bracket expressions labelling ordinary method-2/3 log coordinates.
"""
lyndon_brackets(dimension::Integer, degree::Integer) =
    map(lyndon_bracket, lyndon_words(dimension, degree))

"""
    word_to_idx(word, dimension; scalar_term=false)

Return the **one-based Julia index** of a zero-labelled word in a signature or
expanded tensor logarithm. The empty word requires `scalar_term=true`.
"""
function word_to_idx(word::Tuple, dimension::Integer; scalar_term::Bool=false)
    d, _ = _basis_parameters(dimension, 0)
    _checked_word(word, d)
    isempty(word) && !scalar_term && throw(ArgumentError("empty word requires scalar_term=true"))
    index = 0
    for letter in word
        index = Base.Checked.checked_add(Base.Checked.checked_mul(index, d), Int(letter) + 1)
    end
    return scalar_term ? Base.Checked.checked_add(index, 1) : index
end

"""
    idx_to_word(index, dimension; scalar_term=false)

Invert `word_to_idx` for a one-based Julia coefficient index.
"""
function idx_to_word(index::Integer, dimension::Integer; scalar_term::Bool=false)
    d, _ = _basis_parameters(dimension, 0)
    1 <= index <= typemax(Int) || throw(ArgumentError("index must be a positive Int"))
    code = Int(index) - Int(scalar_term)
    code == 0 && return ()
    d > 0 || throw(ArgumentError("dimension zero has no nonempty words"))
    letters = Int[]
    while code > 0
        code, letter = divrem(code - 1, d)
        push!(letters, letter)
    end
    return Tuple(reverse!(letters))
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
            basis = Any[nothing]
            orders = Int[0]
            forest_words = Tuple[()]
            if planar && d > 0
                for order in 1:n
                    _foreach_forest(weights, order, true) do ids
                        push!(basis, map(id -> rooted[id], ids))
                        push!(orders, order)
                        push!(forest_words, ids)
                    end
                end
            elseif !planar
                append!(basis, rooted)
                append!(orders, weights)
            end
            return (rooted=rooted, basis=basis, orders=orders,
                forest_words=forest_words, indices=Dict(t => i for (i, t) in enumerate(basis)))
        end
    end
end

"""
    rooted_trees(dimension, degree; planar=false)

Enumerate decorated rooted trees through `degree`, excluding the empty tree.
A leaf is `(label,)`; a tree is `(child1, child2, ..., root_label)` with zero-based
labels. Nonplanar children are sorted by native tree ID; planar children retain order.
Unlike `trees(...; planar=true)`, this function returns individual planar trees.
"""
rooted_trees(dimension::Integer, degree::Integer; planar::Bool=false) =
    copy(_rooted_enumeration(dimension, degree, planar).rooted)

"""
    trees(dimension, degree; planar=false)

Enumerate branched-signature coordinate labels, starting with `nothing` for the
scalar. BCK labels are nonplanar rooted trees; MKW labels are ordered forests
`(tree1, tree2, ...)` of planar rooted trees. Order matches pySigLib's native layout.
"""
trees(dimension::Integer, degree::Integer; planar::Bool=false) =
    copy(_tree_enumeration(dimension, degree, planar).basis)

"""
    trees_of_order(dimension, order; planar=false)

Return branched-signature labels of exactly `order` nodes (total forest size for MKW).
Order zero returns `[nothing]`.
"""
function trees_of_order(dimension::Integer, order::Integer; planar::Bool=false)
    enumeration = _tree_enumeration(dimension, order, planar)
    return [t for (t, n) in zip(enumeration.basis, enumeration.orders) if n == order]
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

"""
    tree_to_idx(tree, dimension, degree; planar=false, scalar_term=false)

Return the **one-based Julia index** in a branched signature or expanded Hopf log.
`nothing` requires `scalar_term=true`. Planar inputs may be an ordered forest or
a single tree (shorthand for a one-tree forest). Nonplanar children must be in
the canonical native order returned by `rooted_trees`.
"""
function tree_to_idx(tree, dimension::Integer, degree::Integer;
    planar::Bool=false, scalar_term::Bool=false)
    enumeration = _tree_enumeration(dimension, degree, planar)
    tree === nothing && !scalar_term && throw(ArgumentError("empty tree requires scalar_term=true"))
    if tree !== nothing
        tree isa Tuple && !isempty(tree) || throw(ArgumentError("tree must be a nonempty tuple or nothing"))
        planar && tree[end] isa Integer && (tree = (tree,))
        if planar
            foreach(t -> _validate_tree_label(t, dimension), tree)
        else
            _validate_tree_label(tree, dimension)
        end
    end
    index = get(enumeration.indices, tree, 0)
    index > 0 || throw(ArgumentError("tree is not in the requested basis"))
    return index - Int(!scalar_term)
end

"""
    idx_to_tree(index, dimension, degree; planar=false, scalar_term=false)

Invert `tree_to_idx` for a one-based Julia coefficient index. Planar results are forests.
"""
function idx_to_tree(index::Integer, dimension::Integer, degree::Integer;
    planar::Bool=false, scalar_term::Bool=false)
    basis = _tree_enumeration(dimension, degree, planar).basis
    1 <= index <= length(basis) - Int(!scalar_term) ||
        throw(ArgumentError("index is out of range for the requested basis"))
    return basis[Int(index) + Int(!scalar_term)]
end

"""
    log_sig_basis(dimension, degree; method=1, scalar_term=false)

Return coordinate labels for ordinary log signatures: tensor words for method 0,
Lyndon words for method 1, standard bracket expressions for methods 2 and 3.
The default is method 1, matching `log_sig`. Compressed methods ignore
`scalar_term`. Pass the effective (augmented) dimension.
"""
function log_sig_basis(dimension::Integer, degree::Integer;
    method::Integer=1, scalar_term::Bool=false)
    0 <= method <= 3 || throw(ArgumentError("method must be in 0:3"))
    if method == 0
        basis = words(dimension, degree)
        return scalar_term ? basis : basis[2:end]
    end
    return method == 1 ? lyndon_words(dimension, degree) : lyndon_brackets(dimension, degree)
end

"""
    branched_log_sig_basis(dimension, degree; planar=false, method=nothing, scalar_term=false)

Return labels in branched-log output order. Method 0 uses BCK trees or MKW ordered
forests. Planar method 1 uses weighted Lyndon forests (letters are native planar
tree IDs, weight is node count); methods 2 and 3 use their standard bracketings
`(:bracket, left, right)`, with tree tuples as generators.
These symbolic expressions use the forest-concatenation commutator of the
native MKW coordinate projection.
The default is method 1 for planar logs and 0 otherwise, matching `branched_log_sig`.
Compressed methods ignore `scalar_term` and require planar=true. Enumeration is
available with either artifact; numerical compressed MKW logs require native 4.x.
Pass the effective (augmented) dimension.
"""
function branched_log_sig_basis(dimension::Integer, degree::Integer;
    planar::Bool=false, method::Union{Nothing,Integer}=nothing, scalar_term::Bool=false)
    m = method === nothing ? Int(planar) : method
    0 <= m <= 3 || throw(ArgumentError("method must be in 0:3"))
    m == 0 || planar || throw(ArgumentError("compressed branched log bases require planar=true"))
    enumeration = _tree_enumeration(dimension, degree, planar)
    if m == 0
        basis = copy(enumeration.basis)
        return scalar_term ? basis : basis[2:end]
    end
    selected = findall(_is_lyndon, enumeration.forest_words)
    m == 1 && return [enumeration.basis[i] for i in selected]
    return [_lyndon_bracket(enumeration.forest_words[i], id -> enumeration.rooted[id]) for i in selected]
end
