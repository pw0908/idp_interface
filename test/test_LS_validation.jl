using Test
using Clapeyron
using IDPInterface

# Reference bulk LS-theory implementation (Zhang et al. 2016), kept only in
# resources/ so there is a single source of truth for the benchmark target.
include(joinpath(@__DIR__, "..", "resources", "Helmholtz_LS.jl"))

# The reference file's own design treats each bead TYPE (cation/anion) as a
# separate Clapeyron "component", using per-type parameters N (chain length,
# same for both types), Npm/Nnc (per-chain bond-type totals -- see the
# derivation below), Z (charge), and Znet = N*Z (the charge a hypothetical
# whole-chain-of-this-type would carry, giving ν=|Znet/N|=|Z|=1 for our ±1
# charges). This is the bead-type-as-component design IDPInterface.LS
# deliberately does NOT use (it treats one sequence as one pseudo-pure
# component instead, see src/LS/LSComposite.jl) -- this helper reconstructs
# the reference's own convention purely for cross-validation.
function reference_model(N, Npm, Nnc, chi=0.0)
    Nvec = Clapeyron.SingleParam("N", ["cation", "anion"], [Float64(N), Float64(N)])
    Npmvec = Clapeyron.SingleParam("Npm", ["cation", "anion"], [Float64(Npm), Float64(Npm)])
    Nncvec = Clapeyron.SingleParam("Nnc", ["cation", "anion"], [Float64(Nnc), Float64(Nnc)])
    Zvec = Clapeyron.SingleParam("Z", ["cation", "anion"], [1.0, -1.0])
    Znetvec = Clapeyron.SingleParam("Znet", ["cation", "anion"], [Float64(N), -Float64(N)])
    params = LSParam(Nvec, Npmvec, Nncvec, Zvec, Znetvec, chi)
    return LS(["cation", "anion"], params)
end

const TEST_SEQUENCES = [
    ("alternating", [1, -1, 1, -1, 1, -1, 1, -1]),
    ("diblock", [1, 1, 1, 1, -1, -1, -1, -1]),
    ("random-ish", [1, 1, -1, 1, -1, -1, 1, -1]),
]

@testset "LS bulk validation vs Helmholtz_LS.jl" begin
    @testset "Npm/Nnc per-chain decomposition ($name)" for (name, charges) in TEST_SEQUENCES
        # Validates the algebraic decomposition alone, at matched (lB, ρ★)
        # points -- bypassing the (V,T,z) unit conversion entirely.
        N = length(charges)
        Npm, Nnc = IDPInterface.count_bond_types(charges)
        refm = reference_model(N, Npm, Nnc)

        for lB in (0.0, 0.5, 1.5, 3.0), rho_each in (0.02, 0.05, 0.15)
            rho = [rho_each, rho_each]
            a_ref = a(refm, lB, rho)
            rho_tot = sum(rho)

            η = (π/6) * rho_tot
            yhs = (2 + η) / (2 * (1 - η)^2)
            fhs = 6 * η^2 * (4 - 3η) / (π * (1 - η)^2)
            f0 = sum(rho ./ N .* (log.(rho ./ N) .- 1))
            ρchain = rho_tot / N
            if lB == 0.0
                fch = ρchain * (1 - N) * log(yhs)
                fel = 0.0
            else
                κ = sqrt(4π * lB * sum(rho .* abs.([1.0, 1.0]) .* [1.0, 1.0] .^ 2))
                Γ = (-1 + sqrt(1 + 2κ)) / 2
                fel = -Γ^3 * (2/3 + Γ) / π
                ypp = yhs * exp(-lB / (1 + Γ)^2 + lB)
                ypm = yhs * exp(lB / (1 + Γ)^2 - lB)
                fch = ρchain * ((1 + Npm + Nnc - N) * log(ypp) - Npm * log(ypm) - Nnc * log(yhs))
            end

            @test (f0 + fhs + fel + fch) ≈ a_ref rtol=1e-10
        end
    end

    @testset "Full a_res unit conversion ($name)" for (name, charges) in TEST_SEQUENCES
        # NOTE: this no longer cross-checks against Helmholtz_LS.jl's a()
        # directly -- IDPInterface.LS deliberately uses g_hs (Clapeyron's
        # HeterogcPCPSAFT's own hard-sphere contact-value correlation,
        # exactly what ClassicalDFT's _f_hc_bonds computes for the Step 2 DFT
        # chain term) rather than Helmholtz_LS.jl's yhs=(2+η)/(2(1-η)²) for
        # the neutral (Γ_MSA=0) contact value -- the two differ by a few
        # percent at typical η, which was making the bulk model and the
        # reused DFT kernel subtly inconsistent (see LSNeutral.a_res's
        # docstring). LSIon's Δfch closed form is unchanged (it's already
        # y_hs-agnostic by construction -- see LSIon.a_res's docstring), so
        # this test hand-computes the same g_hs-based reference the source
        # now uses, as a straightforward (V,T,z)-unit-conversion regression
        # check, not an independent cross-validation.
        N = length(charges)
        Npm, Nnc = IDPInterface.count_bond_types(charges)
        mymodel = IDPInterface.LS(charges)
        σ = IDPInterface.LS_SIGMA
        ϵr = 78.38484961 # Clapeyron's ConstRSP default, matches mymodel's default RSPmodel

        for T in (250.0, 298.15, 350.0), η in (0.05, 0.2, 0.4)
            lB = Clapeyron.e_c^2 / (4π * Clapeyron.ϵ_0 * ϵr * σ * Clapeyron.k_B * T)
            rho_tot = η * 6 / π
            ρchain = rho_tot / N

            fhs = 6 * η^2 * (4 - 3η) / (π * (1 - η)^2)
            c1 = 1 / (1 - η)
            c2 = 3η / (1 - η)^2
            c3 = 2η^2 / (1 - η)^3
            g_hs = c1 + 0.5 * c2 + 0.25 * c3
            fch0 = ρchain * (1 - N) * log(g_hs)

            κ = sqrt(4π * lB * rho_tot) # |Z|Z^2=1 for our ±1 charges
            Γ = (-1 + sqrt(1 + 2κ)) / 2
            fel = -Γ^3 * (2/3 + Γ) / π
            Δfch = ρchain * (1 + 2Npm + Nnc - N) * lB * (1 - 1 / (1 + Γ)^2)

            a_ref = (fhs + fch0 + fel + Δfch) / ρchain

            z = [1e-3]
            V = Clapeyron.N_A * z[1] * N * σ^3 / rho_tot
            a_mine = Clapeyron.a_res(mymodel, V, T, z)

            @test a_mine ≈ a_ref rtol=1e-12
        end
    end

    @testset "Neutral/ion decomposition (Γ_MSA=0 collapse)" for (name, charges) in TEST_SEQUENCES
        mymodel = IDPInterface.LS(charges)
        z = [1e-3]
        T = 298.15
        V = Clapeyron.N_A * z[1] * mymodel.neutralmodel.params.N * IDPInterface.LS_SIGMA^3 / 0.2

        a_neutral = Clapeyron.a_res(mymodel.neutralmodel, V, T, z)
        a_ion = Clapeyron.a_res(mymodel.ionmodel, V, T, z)
        a_total = Clapeyron.a_res(mymodel, V, T, z)

        @test a_neutral + a_ion ≈ a_total rtol=1e-12
    end
end
