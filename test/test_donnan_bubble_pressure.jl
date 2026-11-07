using Test
using IDPInterface
using Clapeyron
using Random
using LinearAlgebra

"""
    _oracle_coexistence(model, T, z, Vd0, Vl0)

Independent (test-only) 2-unknown coexistence solver along a FIXED
composition ray `z`: finds `(V_dense,V_dilute)` such that
`P(V_dense)=P(V_dilute)` and `g(V_dense)=g(V_dilute)` (equal molar Gibbs
energy, using `Clapeyron.VT_gibbs_free_energy` directly since `z` here
always sums to 1). This is the theoretically-derived N=2 closed-form
reduction (see the project plan's phase-equilibrium section) expressed as a
standalone oracle, built from Clapeyron primitives completely independently
of `donnan_bubble_residual!`/`bubble_pressure`'s own implementation --
deliberately NOT reusing `find_coexistence` (which cannot be used here: its
`Clapeyron.x0_sat_pure_crit` call hardcodes a length-1 `z` internally,
confirmed directly -- a genuine, not merely assumed, API restriction).
"""
function _oracle_coexistence(model, T, z, Vd0, Vl0)
    f!(F, u) = begin
        Vd, Vl = exp(u[1]), exp(u[2])
        F[1] = Clapeyron.pressure(model, Vd, T, z) - Clapeyron.pressure(model, Vl, T, z)
        F[2] = Clapeyron.VT_gibbs_free_energy(model, Vd, T, z) - Clapeyron.VT_gibbs_free_energy(model, Vl, T, z)
        return F
    end
    u0 = [log(Vd0), log(Vl0)]
    res = Clapeyron.Solvers.nlsolve(f!, u0)
    u = Clapeyron.Solvers.x_sol(res)
    return exp(u[1]), exp(u[2])
end

@testset "Donnan-augmented bubble_pressure (N=2: chain + counter-ion)" begin
    @testset "component_charges: chain net charge + ion charge, aligned correctly" begin
        rng = MersenneTwister(11)
        seq = random_polyampholyte_sequence(20, 6, rng; max_block=4)
        model = IDPInterface.LS(seq; ion_charge=-1)
        Z = IDPInterface.component_charges(model)
        @test Z == [sum(seq), -1]
    end

    @testset "bubble_pressure at x=x* reproduces the exact N=2 closed form (Znet=$Znet)" for Znet in (2, 4, 6, 8, 10)
        rng = MersenneTwister(Znet)
        seq = random_polyampholyte_sequence(20, Znet, rng; max_block=4)
        model = IDPInterface.LS(seq; ion_charge=-1)
        Z = IDPInterface.component_charges(model)
        @test Z == [Znet, -1]

        z_ion = -1
        xstar = z_ion / (z_ion - Znet)
        xdense = [xstar, 1 - xstar]
        @test isapprox(LinearAlgebra.dot(Z, xdense), 0.0; atol=1e-12) # electroneutral by construction

        Tc_ray, = Clapeyron.crit_pure(model, nothing, xdense)
        T = Tc_ray * 0.7
        result = Clapeyron.bubble_pressure(model, T, xdense, DonnanBubblePressure())

        # 1) y == x* exactly (electroneutrality with exactly 2 species pins
        # BOTH phases to the same unique composition -- no free parameter).
        @test result.y ≈ xdense atol=1e-10

        # 2) (V_dense,V_dilute) match the independent P=P,g=g oracle solver.
        Vd_o, Vl_o = _oracle_coexistence(model, T, xdense, result.V_dense, result.V_dilute)
        @test result.V_dense ≈ Vd_o rtol=1e-8
        @test result.V_dilute ≈ Vl_o rtol=1e-8

        # 3) Psi from the joint solve agrees with BOTH species' independently
        # recoverable closed-form chemical-potential-difference diagnostics.
        RT = Clapeyron.Rgas(model) * T
        μd = Clapeyron.VT_chemical_potential(model, result.V_dense, T, xdense)
        μl = Clapeyron.VT_chemical_potential(model, result.V_dilute, T, result.y)
        psi_chain = (μd[1] - μl[1]) / RT / Z[1]
        psi_ion = (μd[2] - μl[2]) / RT / Z[2]
        @test result.Ψ ≈ psi_chain rtol=1e-8
        @test result.Ψ ≈ psi_ion rtol=1e-8
    end

    @testset "input validation: rejects non-electroneutral or badly-sized x" begin
        rng = MersenneTwister(1)
        seq = random_polyampholyte_sequence(20, 6, rng; max_block=4)
        model = IDPInterface.LS(seq; ion_charge=-1)
        @test_throws ArgumentError Clapeyron.bubble_pressure(model, 300.0, [0.5, 0.5], DonnanBubblePressure()) # not electroneutral
        @test_throws ArgumentError Clapeyron.bubble_pressure(model, 300.0, [1.0], DonnanBubblePressure())      # wrong length
    end

    @testset "crit_mix's answer sits outside its own ray's pressure loop (regression note)" begin
        # Documents a real physics distinction: crit_mix's general
        # multicomponent stability criterion allows composition
        # perturbations, so its Tc can sit ABOVE the temperature where a
        # pressure loop still exists along the FIXED ray x (crit_pure's own
        # density-only criterion) -- confirming DonnanBubblePressure's own
        # warm-start (via `x0_crit_pure`/`x0_crit_mix`, both now defined for
        # LS directly in Clapeyron) uses the fixed-ray criterion rather than
        # crit_mix for exactly this reason.
        rng = MersenneTwister(1)
        seq = random_polyampholyte_sequence(20, 6, rng; max_block=4)
        model = IDPInterface.LS(seq; ion_charge=-1)
        xstar = -1 / (-1 - 6)
        xdense = [xstar, 1 - xstar]
        Tc_ray, = Clapeyron.crit_pure(model, nothing, xdense)
        Tc_mix, _, _ = Clapeyron.crit_mix(model, xdense)
        @test Tc_mix > Tc_ray
        @test has_physical_loop(model, Tc_ray * 0.99, xdense)
        @test !has_physical_loop(model, Tc_mix, xdense)
    end
end
