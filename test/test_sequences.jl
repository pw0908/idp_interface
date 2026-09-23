using Test
using IDPInterface
using Clapeyron

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

    @testset "count_bond_types vs count_bond_types_from_groups agree ($b)" for (b, seq) in default_sequence_family(20)
        Npm_direct, Nnc_direct = count_bond_types(seq)
        _, group_names, _ = build_ls_groups(seq)
        model = IDPInterface.LS(seq)
        Npm_groups, Nnc_groups = count_bond_types_from_groups(group_names, model.neutralmodel.groups.n_intergroups[1])
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

    @testset "build_ls_groups is SAFTgammaMie-style (2 groups, not N)" begin
        seq = block_sequence(20, 5)
        groups, group_names, Z = build_ls_groups(seq)
        @test group_names == ["cation", "anion"]
        @test Z == [1, -1]
        @test groups.flattenedgroups == ["cation", "anion"]
        @test groups.n_flattenedgroups[1] == [10, 10]
        @test size(groups.n_intergroups[1]) == (2, 2)
    end

    @testset "expand_ls_groups: per-bead, path graph, Npm agrees ($b)" for (b, seq) in default_sequence_family(20)
        N = length(seq)
        groups, bead_names, Z = expand_ls_groups(seq)
        @test length(bead_names) == N
        @test groups.flattenedgroups == bead_names
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
end
