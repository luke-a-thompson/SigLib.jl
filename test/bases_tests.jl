include("basis_fixtures.jl")

@testset "words and standard Lyndon brackets" begin
    @test words(2, 2) == [(), (0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)]
    @test words_of_length(2, 2) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    @test words_of_length(0, 0) == [()]
    @test isempty(words_of_length(0, 3))
    @test words(0, 3) == [()]
    @test lyndon_words(2, 3) == [(0,), (1,), (0, 1), (0, 0, 1), (0, 1, 1)]
    @test lyndon_brackets(2, 3) == Any[0, 1, (:bracket, 0, 1),
        (:bracket, 0, (:bracket, 0, 1)), (:bracket, (:bracket, 0, 1), 1)]
    @test !is_lyndon(())
    @test !is_lyndon((0, 1, 0, 1))
    @test !is_lyndon((1, 0))
    @test is_lyndon((0, 0, 1))
    for d in 1:3, n in 0:4
        @test lyndon_words_of_length(d, n) == filter(is_lyndon, words_of_length(d, n))
        @test length(lyndon_words(d, n)) == log_sig_length(d, n)
        for scalar in (false, true)
            basis = scalar ? words(d, n) : words(d, n)[2:end]
            @test length(basis) == sig_length(d, n; scalar_term=scalar)
            for (i, word) in enumerate(basis)
                @test word_to_idx(word, d; scalar_term=scalar) == i
                @test idx_to_word(i, d; scalar_term=scalar) == word
            end
        end
    end
    @test idx_to_word(1, 0; scalar_term=true) == ()
    @test_throws ArgumentError word_to_idx((), 2)
    @test_throws ArgumentError word_to_idx((2,), 2)
    @test_throws ArgumentError word_to_idx((-1,), 2)
    @test_throws ArgumentError idx_to_word(0, 2)
    @test_throws ArgumentError idx_to_word(1, 0)
    @test_throws ArgumentError lyndon_bracket((0, 0))
    @test_throws ArgumentError lyndon_bracket(())
    @test_throws ArgumentError is_lyndon((0.0,))
    @test_throws ArgumentError words(-1, 2)
    @test_throws ArgumentError lyndon_words(2, -1)
    @test_throws OverflowError word_to_idx(ntuple(_ -> 1, 64), 2)
end

@testset "BCK trees and MKW ordered forests" begin
    @test trees(2, 3) == BCK_BASIS_2_3
    @test trees(2, 3; planar=true) == MKW_BASIS_2_3
    @test rooted_trees(2, 2) == BCK_BASIS_2_3[2:7]
    @test rooted_trees(1, 4; planar=true) != rooted_trees(1, 4)
    for planar in (false, true), d in 1:3, n in 0:4
        basis = trees(d, n; planar=planar)
        @test length(basis) == branched_sig_length(d, n; planar=planar, scalar_term=true)
        @test vcat([trees_of_order(d, k; planar=planar) for k in 0:n]...) == basis
        for scalar in (false, true)
            labels = scalar ? basis : basis[2:end]
            for (i, tree) in enumerate(labels)
                @test tree_to_idx(tree, d, n; planar=planar, scalar_term=scalar) == i
                @test idx_to_tree(i, d, n; planar=planar, scalar_term=scalar) == tree
            end
        end
    end
    @test trees(0, 3) == [nothing]
    @test trees_of_order(0, 0; planar=true) == [nothing]
    @test isempty(trees_of_order(0, 3))
    @test tree_to_idx((0,), 2, 2; planar=true) == 1
    @test tree_to_idx(((0,), (1,)), 2, 2; planar=true) !=
        tree_to_idx(((1,), (0,)), 2, 2; planar=true)
    changed = trees(2, 3)
    changed[1] = (99,)
    @test trees(2, 3)[1] === nothing
    @test_throws ArgumentError tree_to_idx(nothing, 2, 2)
    @test_throws ArgumentError tree_to_idx((2,), 2, 2)
    @test_throws ArgumentError tree_to_idx((0.0,), 2, 2)
    @test_throws ArgumentError tree_to_idx(((0.0,),), 2, 2; planar=true)
    @test_throws ArgumentError tree_to_idx((), 2, 2)
    @test_throws ArgumentError tree_to_idx(((0,), 0), 2, 1)
    @test_throws ArgumentError idx_to_tree(0, 2, 2)
    @test_throws ArgumentError idx_to_tree(7, 2, 2)
    @test_throws ArgumentError trees(256, 0)
    @test_throws ArgumentError trees(2, -1)

    # Tensor words read chronologically from leaf to root along a chain tree.
    path = [0.0 0.0; 0.2 -0.3; 0.7 0.4; 0.5 0.8]
    tensor = sig(path, 3)
    for planar in (false, true)
        branched = branched_sig(path, 3; planar=planar)
        for word in words(2, 3)[2:end]
            tree = (word[1],)
            for label in word[2:end]
                tree = (tree, label)
            end
            @test branched[tree_to_idx(tree, 2, 3; planar=planar)] ≈ tensor[word_to_idx(word, 2)]
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
    path = [0.0 0.0; 0.2 -0.3; 0.7 0.4; 0.5 0.8]
    @test log_sig_basis(2, 3; method=1) == lyndon_words(2, 3)
    @test log_sig_basis(2, 3; method=3, scalar_term=true) == lyndon_brackets(2, 3)
    @test log_sig_basis(2, 0; method=0, scalar_term=true) == [()]
    @test isempty(log_sig_basis(2, 0))
    @test_throws ArgumentError log_sig_basis(2, 3; method=4)
    @test_throws ArgumentError branched_log_sig_basis(2, 3; method=1)
    @test_throws ArgumentError branched_log_sig_basis(2, 3; planar=true, method=-1)
    @test branched_log_sig_basis(2, 3) == BCK_BASIS_2_3[2:end]
    @test branched_log_sig_basis(2, 3; method=0, scalar_term=true) == BCK_BASIS_2_3
    @test branched_log_sig_basis(2, 3; planar=true, method=0) == MKW_BASIS_2_3[2:end]
    @test branched_log_sig_basis(2, 2; planar=true) ==
        Any[((0,),), ((1,),), ((0,), (1,)), (((0,), 0),), (((0,), 1),), (((1,), 0),), (((1,), 1),)]
    for planar in (false, true), method in (planar ? (0:3) : (0:0))
        @test isempty(branched_log_sig_basis(0, 3; planar=planar, method=method))
        @test isempty(branched_log_sig_basis(2, 0; planar=planar, method=method))
    end
    for degree in 2:5
        coefficients = log_sig(path, degree; method=2)
        expanded = zeros(sig_length(2, degree))
        for (coefficient, expr) in zip(coefficients, log_sig_basis(2, degree; method=2))
            for (word, multiplicity) in expand_basis_bracket(expr)
                expanded[word_to_idx(word, 2)] += coefficient * multiplicity
            end
        end
        @test expanded ≈ log_sig(path, degree; method=0) atol=1e-12
        tensorlog = log_sig(path, degree; method=0)
        selected = [word_to_idx(w, 2) for w in log_sig_basis(2, degree; method=1)]
        @test tensorlog[selected] ≈ log_sig(path, degree; method=1)
    end
    if SigLib._modern_abi[]
        for degree in 2:4
            extracted = branched_log_sig_basis(2, degree; planar=true, method=1)
            bracketed = branched_log_sig_basis(2, degree; planar=true, method=2)
            @test length(bracketed) == branched_log_sig_length(2, degree; planar=true)
            @test bracketed == branched_log_sig_basis(2, degree; planar=true, method=3)
            coefficients = branched_log_sig(path, degree; planar=true, method=2)
            expanded = zeros(branched_sig_length(2, degree; planar=true))
            for (coefficient, expr) in zip(coefficients, bracketed)
                for (forest, multiplicity) in expand_basis_bracket(expr)
                    expanded[tree_to_idx(forest, 2, degree; planar=true)] += coefficient * multiplicity
                end
            end
            selected = [tree_to_idx(f, 2, degree; planar=true) for f in extracted]
            method1 = branched_log_sig(path, degree; planar=true, method=1)
            @test expanded[selected] ≈ method1 atol=1e-12
            hopflog = branched_log_sig(path, degree; planar=true, method=0)
            @test hopflog[selected] ≈ method1 atol=1e-12
        end
    end
end
