using Test
using Clapeyron
using ClassicalDFT
using IDPInterface

# Step 2 deliverable: confirm ClassicalDFT's spatial LS-theory functional
# (src/DFT/LSDFTNeutral.jl, LSDFTIon.jl, LSDFTComposite.jl) reduces to Step
# 1's already-validated bulk Clapeyron model in the uniform (bulk) limit.
# Follows the bulk-limit pattern ClassicalDFT's own test suite uses for
# heterosegmented/electrolyte models (test/test_models.jl): build a small
# Uniform1DCart structure at bulk density, compare Clapeyron's
# VT_chemical_potential_res against ClassicalDFT.δFδρ_res at every bead.

const DFT_TEST_SEQUENCES = [
    ("alternating-2", [1, -1]),
    ("alternating-6", [1, -1, 1, -1, 1, -1]),
    ("diblock",       [1, 1, 1, 1, -1, -1, -1, -1]),
    ("alternating",   [1, -1, 1, -1, 1, -1, 1, -1]),
    ("random-ish",    [1, 1, -1, 1, -1, -1, 1, -1]),
]

"""
    dft_mu2(model, N; T=298.15, eta=0.2)

Build a minimal Uniform1DCart LSDFTSystem for `model` (an `LS`, `LSNeutral`,
or `LSIon` composite/sub-model built with `expand=true`) at packing fraction
`eta`, and return `δFδρ_res` evaluated at the bulk-initialized profile
(one value per bead).
"""
function dft_mu2(model::IDPInterface.LS, N; T=298.15, eta=0.2)
    sigma = IDPInterface.LS_SIGMA
    rho_tot_star = eta * 6 / pi
    z = [1e-3]
    V = Clapeyron.N_A * z[1] * N * sigma^3 / rho_tot_star
    rhobulk = z ./ V
    L = ClassicalDFT.length_scale(model)
    structure = ClassicalDFT.Uniform1DCart((1e5, T), rhobulk, [0.0, 3 * L], (3,))
    sys = IDPInterface.LSDFTSystem(model, structure)
    rho = ClassicalDFT.initialize_profiles(sys)
    mu1 = Clapeyron.VT_chemical_potential_res(model, V, T, z) / T / Clapeyron.Rgas()
    mu2 = ClassicalDFT.δFδρ_res(sys, rho)
    return mu1[1], mu2[1, :]
end

@testset "Step 2: ClassicalDFT bulk limit vs Clapeyron LS" begin
    @testset "Full composite (neutral+ion+chain) ($name)" for (name, charges) in DFT_TEST_SEQUENCES
        N = length(charges)
        model = IDPInterface.LS(charges; expand=true)
        for eta in (0.05, 0.2, 0.35)
            mu1, mu2 = dft_mu2(model, N; eta=eta)
            # ClassicalDFT and Clapeyron agree, and every bead agrees with every other
            @test all(x -> isapprox(x, mu1; rtol=1e-8), mu2)
        end
    end

    @testset "Neutral-only functional matches LSNeutral.a_res ($name)" for (name, charges) in DFT_TEST_SEQUENCES
        N = length(charges)
        model = IDPInterface.LS(charges; expand=true)
        neutralmodel = model.neutralmodel
        sigma = IDPInterface.LS_SIGMA
        eta = 0.2
        rho_tot_star = eta * 6 / pi
        z = [1e-3]
        V = Clapeyron.N_A * z[1] * N * sigma^3 / rho_tot_star
        rhobulk = z ./ V
        T = 298.15

        mu1 = Clapeyron.VT_chemical_potential_res(neutralmodel, V, T, z) / T / Clapeyron.Rgas()

        L = ClassicalDFT.length_scale(neutralmodel)
        structure = ClassicalDFT.Uniform1DCart((1e5, T), rhobulk, [0.0, 3 * L], (3,))
        species = ClassicalDFT.get_species(neutralmodel, structure)
        fields = ClassicalDFT.get_fields(neutralmodel, species, structure, ClassicalDFT.CPU(), Float64)
        propagator = ClassicalDFT.get_propagator(neutralmodel, species, structure, ClassicalDFT.CPU(), Float64)
        NF = ClassicalDFT.compute_field_len(fields, ClassicalDFT.dimension(structure))
        options = ClassicalDFT.DFTOptions()
        sys = ClassicalDFT.DFTSystem(neutralmodel, species, structure, fields, nothing, propagator, options, Val{NF}())
        rho = ClassicalDFT.initialize_profiles(sys)
        mu2 = ClassicalDFT.δFδρ_res(sys, rho)

        @test all(x -> isapprox(x, mu1[1]; rtol=1e-8), mu2[1, :])
    end
end
