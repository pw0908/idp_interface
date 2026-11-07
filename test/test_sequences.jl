using Test
using IDPInterface
using Clapeyron
using Random

"""
    _multiset_permutations_kappa_sym_raw_max(n_plus, n_minus, window)

Exhaustive (no external dependency) search over every distinct arrangement
of `n_plus` `+1`s and `n_minus` `-1`s, returning the true maximum
`_kappa_sym_raw` value -- used only as a brute-force cross-check for small
`N=n_plus+n_minus` in tests, never in production code (see
`kappa_sym_max`'s docstring for the constructive heuristic that IS used
there).
"""
function _multiset_permutations_kappa_sym_raw_max(n_plus::Int, n_minus::Int, window::Int)
    N = n_plus + n_minus
    best = Ref(-1)
    charges = Vector{Int}(undef, N)
    remaining = [n_plus, n_minus]
    function recurse(pos)
        if pos > N
            best[] = max(best[], IDPInterface._kappa_sym_raw(charges, window))
            return
        end
        for (k, sign) in ((1, 1), (2, -1))
            if remaining[k] > 0
                remaining[k] -= 1
                charges[pos] = sign
                recurse(pos + 1)
                remaining[k] += 1
            end
        end
    end
    recurse(1)
    return best[]
end

@testset "Sequence generation and connectivity" begin
    @testset "kappa_seq analytic limits" begin
        N = 20
        diblock = block_sequence(N, N ÷ 2)
        alternating = block_sequence(N, 1)
        @test kappa_seq(diblock) ≈ 1.0
        @test kappa_seq(alternating) < 0.05 # well-mixed, not exactly 0 at finite blob size
    end

    @testset "block_sequence properties" for b in (1, 2, 5, 10)
        seq = block_sequence(20, b)
        @test length(seq) == 20
        @test count(==(1), seq) == 10
        @test count(==(-1), seq) == 10
    end

    @testset "count_bond_types vs Clapeyron.LS's own Npm/Nnc agree ($b)" for (b, seq) in default_sequence_family(20)
        Npm_direct, Nnc_direct = count_bond_types(seq)
        model = IDPInterface.LS(seq)
        Npm_groups = model.neutralmodel.params.Npm.values[1]
        Nnc_groups = model.neutralmodel.params.Nnc.values[1]
        @test Npm_direct == Npm_groups
        @test Nnc_direct == Nnc_groups
    end

    @testset "Npm analytic limits (independent of computation path)" begin
        N = 20
        diblock = block_sequence(N, N ÷ 2)
        alternating = block_sequence(N, 1)
        @test count_bond_types(diblock) == (1, 0)          # single block junction
        @test count_bond_types(alternating) == (N - 1, 0)  # every bond is unlike-charge
    end

    @testset "sequence_to_ls_component is SAFTgammaMie-style (2 groups, not N)" begin
        seq = block_sequence(20, 5)
        name, groups, bonds = sequence_to_ls_component(seq)
        @test Dict(groups) == Dict("+" => 10, "-" => 10)
        model = IDPInterface.LS(seq)
        @test model.neutralmodel.groups.flattenedgroups == ["+", "-"]
        @test model.neutralmodel.groups.n_flattenedgroups[1] == [10, 10]
        @test size(model.neutralmodel.groups.n_intergroups[1]) == (2, 2)
    end

    @testset "expand=true: per-bead, path graph, Npm agrees ($b)" for (b, seq) in default_sequence_family(20)
        N = length(seq)
        model = IDPInterface.LS(seq; expand=true)
        groups = model.neutralmodel.groups
        Z = model.charge
        @test length(groups.flattenedgroups) == N
        @test all(==(1), groups.n_flattenedgroups[1]) # one bead per group instance
        @test size(groups.n_intergroups[1]) == (N, N)

        bondmat = groups.n_intergroups[1]
        # path graph: only (i,i+1)/(i+1,i) entries are 1, everything else 0
        expected = zeros(Int, N, N)
        for i in 1:N-1
            expected[i, i+1] = expected[i+1, i] = 1
        end
        @test bondmat == expected

        Npm_expanded = count(i -> bondmat[i, i+1] != 0 && Z[i] != Z[i+1], 1:N-1)
        Npm_direct, _ = count_bond_types(seq)
        @test Npm_expanded == Npm_direct
    end

    @testset "compact vs expanded LS give identical a_res ($b)" for (b, seq) in default_sequence_family(20)
        z = [1e-3]
        V = 0.02
        T = 300.0
        model_compact = IDPInterface.LS(seq; expand=false)
        model_expanded = IDPInterface.LS(seq; expand=true)
        @test length(model_compact.neutralmodel.groups.flattenedgroups) == 2
        @test length(model_expanded.neutralmodel.groups.flattenedgroups) == length(seq)
        @test Clapeyron.a_res(model_compact, V, T, z) ≈ Clapeyron.a_res(model_expanded, V, T, z) rtol=1e-12
    end

    @testset "random_polyampholyte_sequence: composition matches target net_charge" begin
        rng = MersenneTwister(1)
        for net_charge in (-20, -8, -2, 0, 2, 8, 20)
            for _ in 1:10
                seq = random_polyampholyte_sequence(20, net_charge, rng; max_block=4)
                @test length(seq) == 20
                @test sum(seq) == net_charge
                @test all(x -> x == 1 || x == -1, seq)
            end
        end
        # net_charge=0 must reproduce random_blocky_sequence's exact +/- split
        seq = random_polyampholyte_sequence(20, 0, rng)
        @test count(==(1), seq) == 10 && count(==(-1), seq) == 10
    end

    @testset "random_polyampholyte_sequence: rejects infeasible/malformed inputs" begin
        rng = MersenneTwister(1)
        @test_throws ArgumentError random_polyampholyte_sequence(21, 0, rng)      # N odd
        @test_throws ArgumentError random_polyampholyte_sequence(20, 1, rng)      # N+net_charge odd
        @test_throws ArgumentError random_polyampholyte_sequence(20, 22, rng)     # net_charge > N
    end

    @testset "kappa_sym_max net-neutral case unchanged (regression)" begin
        @test kappa_sym_max(20) == 1040                       # previously-validated net-neutral N=20 value
        @test kappa_sym_max(20; net_charge=0) == kappa_sym_max(20)
    end

    @testset "kappa_sym_max net-charged: exact match against exhaustive brute force (small N)" begin
        for N in (8, 12), window in (2, 3)
            window > N && continue
            for net_charge in -N:2:N
                n_plus = (N + net_charge) ÷ 2
                n_minus = N - n_plus
                (n_plus < 0 || n_minus < 0) && continue
                expected = _multiset_permutations_kappa_sym_raw_max(n_plus, n_minus, window)
                @test kappa_sym_max(N; window=window, net_charge=net_charge) == expected
            end
        end
    end

    @testset "kappa_sym_max net-charged: matches/beats random search at N=20,window=6" begin
        rng = MersenneTwister(7)
        N, window = 20, 6
        for net_charge in -20:2:20
            n_plus = (N + net_charge) ÷ 2
            n_minus = N - n_plus
            h = kappa_sym_max(N; window=window, net_charge=net_charge)
            base = vcat(fill(1, n_plus), fill(-1, n_minus))
            best_random = -1
            for _ in 1:20_000
                shuffle!(rng, base)
                best_random = max(best_random, IDPInterface._kappa_sym_raw(base, window))
            end
            @test h >= best_random
        end
    end

    @testset "kappa_sym: net-charged sequences stay in [0,1], reject bad inputs" begin
        rng = MersenneTwister(3)
        for net_charge in (-8, -2, 2, 8)
            for _ in 1:20
                seq = random_polyampholyte_sequence(20, net_charge, rng; max_block=4)
                k = kappa_sym(seq)
                @test 0.0 <= k <= 1.0 + 1e-12
            end
        end
        @test_throws ArgumentError kappa_sym_max(21) # N odd
    end
end
