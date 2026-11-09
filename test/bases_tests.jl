include("basis_fixtures.jl")

@testset "words and standard Lyndon brackets" begin
    @test words(2, 2) == [(0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)]
    @test words(2, 2; scalar_term=true) == [(), (0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)]
    @test words(2, 0; scalar_term=true) == [()]
    @test words_of_length(2, 2) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    @test isempty(words_of_length(2, 0))
    @test words_of_length(0, 0; scalar_term=true) == [()]
    @test isempty(words_of_length(0, 3))
    @test isempty(words(0, 3))
    @test words(0, 3; scalar_term=true) == [()]
    @test lyndon_words(2, 3) == [(0,), (1,), (0, 1), (0, 0, 1), (0, 1, 1)]
    @test lyndon_brackets(2, 3) == Any[0, 1, (:bracket, 0, 1),
        (:bracket, 0, (:bracket, 0, 1)), (:bracket, (:bracket, 0, 1), 1)]
    @test !is_lyndon(())
    @test !is_lyndon((0, 1, 0, 1))
    @test !is_lyndon((1, 0))
    @test is_lyndon((0, 0, 1))
    for d in 1:3, n in 0:4
        @test lyndon_words_of_length(d, n) == filter(is_lyndon, words_of_length(d, n))
        lyndon = lyndon_words(d, n)
        @test length(lyndon) == log_sig_length(d, n)
        for (i, word) in enumerate(lyndon)
            @test lyndon_word_to_idx(word, d, n) == i
            @test idx_to_lyndon_word(i, d, n) == word
        end
        for scalar in (false, true)
            basis = words(d, n; scalar_term=scalar)
            @test length(basis) == sig_length(d, n; scalar_term=scalar)
            for (i, word) in enumerate(basis)
                @test word_to_idx(word, d, n; scalar_term=scalar) == i
                @test idx_to_word(i, d, n; scalar_term=scalar) == word
            end
            @test_throws ArgumentError idx_to_word(length(basis) + 1, d, n; scalar_term=scalar)
        end
        @test_throws ArgumentError idx_to_lyndon_word(length(lyndon) + 1, d, n)
    end
    # Upstream words.rst example, shifted to one-based indices.
    @test lyndon_words(2, 3)[3] == (0, 1)
    @test lyndon_word_to_idx((0, 1), 2, 3) == 3
    @test word_to_idx((0, 1), 2, 3) == 4
    @test idx_to_word(1, 0, 3; scalar_term=true) == ()
    @test_throws ArgumentError word_to_idx((), 2, 2)
    @test_throws ArgumentError word_to_idx((2,), 2, 2)
    @test_throws ArgumentError word_to_idx((-1,), 2, 2)
    @test_throws ArgumentError word_to_idx((0, 0, 0), 2, 2)
    @test_throws ArgumentError idx_to_word(0, 2, 2)
    @test_throws ArgumentError idx_to_word(1, 0, 2)
    @test_throws ArgumentError lyndon_word_to_idx((1, 0), 2, 2)
    @test_throws ArgumentError lyndon_word_to_idx((0, 0, 1), 2, 2)
    @test_throws ArgumentError lyndon_word_to_idx((), 2, 2)
    @test_throws ArgumentError idx_to_lyndon_word(0, 2, 2)
    @test_throws ArgumentError lyndon_bracket((0, 0))
    @test_throws ArgumentError lyndon_bracket(())
    @test_throws ArgumentError is_lyndon((0.0,))
    @test_throws ArgumentError words(-1, 2)
    @test_throws ArgumentError lyndon_words(2, -1)
    @test_throws OverflowError word_to_idx(ntuple(_ -> 1, 64), 2, 64)
    @test idx_to_word(typemax(Int), 2, 100) isa Tuple
end

@testset "BCK trees, MKW forests, and Lyndon forests" begin
    @test trees(2, 3; scalar_term=true) == BCK_BASIS_2_3
    @test trees(2, 3) == BCK_BASIS_2_3[2:end]
    @test forests(2, 3; scalar_term=true) == [(); MKW_BASIS_2_3[2:end]]
    @test forests(2, 3) == MKW_BASIS_2_3[2:end]
    @test rooted_trees(2, 2) == BCK_BASIS_2_3[2:7]
    @test rooted_trees(1, 4; planar=true) != rooted_trees(1, 4)
    for d in 1:3, n in 0:4
        @test vcat(Any[], [trees_of_order(d, k; scalar_term=true) for k in 0:n]...) ==
            trees(d, n; scalar_term=true)
        for (family, inverse, index, empty, planar) in (
            (trees, idx_to_tree, tree_to_idx, nothing, false),
            (forests, idx_to_forest, forest_to_idx, (), true),
        )
            for scalar in (false, true)
                basis = family(d, n; scalar_term=scalar)
                @test length(basis) == branched_sig_length(d, n; planar, scalar_term=scalar)
                scalar && @test basis[1] === empty
                for (i, element) in enumerate(basis)
                    @test index(element, d, n; scalar_term=scalar) == i
                    @test inverse(i, d, n; scalar_term=scalar) == element
                end
                @test_throws ArgumentError inverse(length(basis) + 1, d, n; scalar_term=scalar)
            end
        end
        lyndon = lyndon_forests(d, n)
        @test length(lyndon) == branched_log_sig_length(d, n; planar=true)
        @test lyndon == filter(f -> f in Set(lyndon), forests(d, n))
        for (i, forest) in enumerate(lyndon)
            @test lyndon_forest_to_idx(forest, d, n) == i
            @test idx_to_lyndon_forest(i, d, n) == forest
        end
        @test_throws ArgumentError idx_to_lyndon_forest(length(lyndon) + 1, d, n)
    end
    # Upstream trees.rst example, shifted to one-based indices.
    leaf, chain = (0,), ((0,), 0)
    @test lyndon_forests(1, 2) == [(leaf,), (chain,)]
    @test (leaf, chain) in lyndon_forests(1, 3)
    @test forest_to_idx((chain,), 1, 2) == 3
    @test lyndon_forest_to_idx((chain,), 1, 2) == 2
    @test forests(1, 2) == [((0,),), ((0,), (0,)), (((0,), 0),)]
    @test trees(2, 2) == [(0,), (1,), ((0,), 0), ((0,), 1), ((1,), 0), ((1,), 1)]
    @test isempty(trees(0, 3))
    @test trees(0, 3; scalar_term=true) == [nothing]
    @test forests(2, 0; scalar_term=true) == [()]
    @test trees_of_order(2, 0; scalar_term=true) == [nothing]
    @test isempty(trees_of_order(2, 0))
    @test isempty(trees_of_order(0, 3))
    @test forest_to_idx(((0,), (1,)), 2, 2) != forest_to_idx(((1,), (0,)), 2, 2)
    changed = trees(2, 3; scalar_term=true)
    changed[1] = (99,)
    @test trees(2, 3; scalar_term=true)[1] === nothing
    changed = lyndon_forests(2, 3)
    empty!(changed)
    @test !isempty(lyndon_forests(2, 3))
    @test_throws ArgumentError tree_to_idx(nothing, 2, 2)
    @test_throws ArgumentError tree_to_idx((2,), 2, 2)
    @test_throws ArgumentError tree_to_idx((0.0,), 2, 2)
    @test_throws ArgumentError tree_to_idx((), 2, 2)
    @test_throws ArgumentError tree_to_idx(((0,), 0), 2, 1)
    @test_throws ArgumentError tree_to_idx(((1,), (0,), 0), 2, 3) # noncanonical child order
    @test_throws ArgumentError idx_to_tree(0, 2, 2)
    @test_throws ArgumentError idx_to_tree(7, 2, 2)
    @test_throws ArgumentError forest_to_idx((0,), 2, 2) # a bare tree is not a forest
    @test_throws ArgumentError forest_to_idx(((0.0,),), 2, 2)
    @test_throws ArgumentError forest_to_idx((), 2, 2)
    @test forest_to_idx((), 2, 2; scalar_term=true) == 1
    @test_throws ArgumentError lyndon_forest_to_idx(((1,), (0,)), 2, 2)
    @test_throws ArgumentError lyndon_forest_to_idx((), 2, 2)
    @test_throws ArgumentError idx_to_lyndon_forest(0, 2, 2)
    @test_throws ArgumentError trees(256, 0)
    @test_throws ArgumentError trees(2, -1)
    @test_throws ArgumentError forests(2, -1)

# Tensor words read chronologically from leaf to root along a chain tree.
    path = [0.0 0.2 0.7 0.5; 0.0 -0.3 0.4 0.8]
    tensor = sig(path, 3)
    for planar in (false, true)
        branched = branched_sig(path, 3; planar=planar)
        for word in words(2, 3)
            tree = (word[1],)
            for label in word[2:end]
                tree = (tree, label)
            end
            index = planar ? forest_to_idx((tree,), 2, 3) : tree_to_idx(tree, 2, 3)
            @test branched[index] ≈ tensor[word_to_idx(word, 2, 3)]
        end
    end
end

# Independent associative-algebra expansion of the returned symbolic expressions.
function expand_basis_bracket(expr)
    if expr isa Tuple && length(expr) == 3 && expr[1] === :bracket
        left = expand_basis_bracket(expr[2])
        right = expand_basis_bracket(expr[3])
        result = Dict{Tuple,BigInt}()
        for (u, a) in left, (v, b) in right
            uv, vu = (u..., v...), (v..., u...)
            result[uv] = get(result, uv, big(0)) + a * b
            result[vu] = get(result, vu, big(0)) - a * b
        end
        return result
    end
    return Dict{Tuple,BigInt}((expr,) => big(1))
end

@testset "coordinate bases agree with native logarithms" begin
    path = [0.0 0.2 0.7 0.5; 0.0 -0.3 0.4 0.8]
    @test log_sig_basis(2, 3; method=LogSigMethod.LyndonWords) == lyndon_words(2, 3)
    @test log_sig_basis(2, 3; method=LogSigMethod.DirectBCH, scalar_term=true) == lyndon_brackets(2, 3)
    @test log_sig_basis(2, 0; method=LogSigMethod.Expanded, scalar_term=true) == [()]
    @test isempty(log_sig_basis(2, 0))
    @test_throws TypeError log_sig_basis(2, 3; method=4)
    @test_throws ArgumentError branched_log_sig_basis(2, 3; method=BranchedLogSigMethod.MkwWords)
    @test_throws TypeError branched_log_sig_basis(2, 3; planar=true, method=-1)
    @test branched_log_sig_basis(2, 3) == BCK_BASIS_2_3[2:end]
    @test branched_log_sig_basis(2, 3; method=BranchedLogSigMethod.Expanded, scalar_term=true) == BCK_BASIS_2_3
    @test branched_log_sig_basis(2, 3; planar=true, method=BranchedLogSigMethod.Expanded) == forests(2, 3)
    @test branched_log_sig_basis(2, 2; planar=true) ==
        Any[((0,),), ((1,),), ((0,), (1,)), (((0,), 0),), (((0,), 1),), (((1,), 0),), (((1,), 1),)]
    for planar in (false, true), method in (planar ? instances(BranchedLogSigMethod.T) : (BranchedLogSigMethod.Expanded,))
        @test isempty(branched_log_sig_basis(0, 3; planar=planar, method=method))
        @test isempty(branched_log_sig_basis(2, 0; planar=planar, method=method))
    end
    for degree in 2:5
        coefficients = log_sig(path, degree; method=LogSigMethod.LyndonBasis)
        expanded = zeros(sig_length(2, degree))
        for (coefficient, expr) in zip(coefficients, log_sig_basis(2, degree; method=LogSigMethod.LyndonBasis))
            for (word, multiplicity) in expand_basis_bracket(expr)
                expanded[word_to_idx(word, 2, degree)] += coefficient * multiplicity
            end
        end
        @test expanded ≈ log_sig(path, degree; method=LogSigMethod.Expanded) atol=1e-12
        tensorlog = log_sig(path, degree; method=LogSigMethod.Expanded)
        selected = [word_to_idx(w, 2, degree) for w in log_sig_basis(2, degree; method=LogSigMethod.LyndonWords)]
        @test tensorlog[selected] ≈ log_sig(path, degree; method=LogSigMethod.LyndonWords)
    end
    for degree in 2:4
        extracted = branched_log_sig_basis(2, degree; planar=true, method=BranchedLogSigMethod.MkwWords)
        bracketed = branched_log_sig_basis(2, degree; planar=true, method=BranchedLogSigMethod.MkwBasis)
        @test length(bracketed) == branched_log_sig_length(2, degree; planar=true)
        @test bracketed == branched_log_sig_basis(2, degree; planar=true, method=BranchedLogSigMethod.DirectBCH)
        coefficients = branched_log_sig(path, degree; planar=true, method=BranchedLogSigMethod.MkwBasis)
        expanded = zeros(branched_sig_length(2, degree; planar=true))
        for (coefficient, expr) in zip(coefficients, bracketed)
            for (forest, multiplicity) in expand_basis_bracket(expr)
                expanded[forest_to_idx(forest, 2, degree)] += coefficient * multiplicity
            end
        end
        selected = [forest_to_idx(f, 2, degree) for f in extracted]
        @test extracted == lyndon_forests(2, degree)
        method1 = branched_log_sig(path, degree; planar=true, method=BranchedLogSigMethod.MkwWords)
        @test expanded[selected] ≈ method1 atol=1e-12
        hopflog = branched_log_sig(path, degree; planar=true, method=BranchedLogSigMethod.Expanded)
        @test hopflog[selected] ≈ method1 atol=1e-12
    end
end
