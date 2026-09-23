# `ClassicalDFT.converge!`'s entry points (`AASol`, `cDFTProblem`,
# `get_new_profile!`) all dispatch on the closed
# `Union{DFTSystem,DGTSystem,ElectrolyteDFTSystem}` -- `LSDFTSystem` is none
# of these, so `converge!(system::LSDFTSystem, ρ)` has no matching method
# without the three extensions below. `converge!` itself (the actual
# Anderson-acceleration driver, `converge!(prob::DFTProblem{S},
# method::AASol, ρ) where S`) is generic over `S`, so it needs no new method
# -- only the three functions that select/build defaults for a *specific*
# system type do.

"""
    ClassicalDFT.AASol(system::LSDFTSystem; kwargs...)

Same defaults as `AASol(::Union{DFTSystem,DGTSystem,ElectrolyteDFTSystem};...)`
(`maxit=10000,beta=1e-2,tol=1e-4,anderson_start=1e-1,anderson_m=5`).
"""
function ClassicalDFT.AASol(system::LSDFTSystem; maxit=10000, beta=1e-2, tol=1e-4,
                             anderson_start=1e-1, anderson_m=5, verbose=false,
                             nan_max_retries=5, nan_beta_factor=0.5,
                             omega_window=0, omega_rtol=1e-6, kwargs...)
    return ClassicalDFT.__AASol(; maxit, beta, tol, anderson_start, anderson_m, verbose,
                                 nan_max_retries, nan_beta_factor, omega_window, omega_rtol)
end

"""
    ClassicalDFT.cDFTProblem(system::LSDFTSystem; kwargs...)

`DFTProblem(system)`'s own constructor is untyped on `system` -- the only
gap is `cDFTProblem`'s own dispatch, which is Union-restricted.
"""
ClassicalDFT.cDFTProblem(system::LSDFTSystem; kwargs...) = ClassicalDFT.DFTProblem(system; kwargs...)

"""
    ClassicalDFT.get_new_profile!(system::LSDFTSystem, ρ, δfδρ_res, caches)

Verbatim translation of
`get_new_profile!(::Union{DFTSystem,DGTSystem,ElectrolyteDFTSystem}, ρ, δfδρ_res, caches)`
(`src/methods/converge.jl`), with the `@comps`/`@chain(i)` macro loop over
components/beads replaced by direct `model.groups.i_groups[1]` iteration --
`LS` is always exactly one Clapeyron component (`length(model)==1`), so this
is an exact, not approximate, translation (and avoids relying on
`@comps`/`@chain`'s macro hygiene resolving `model` correctly across the
module boundary).
"""
function ClassicalDFT.get_new_profile!(system::LSDFTSystem, ρ, δfδρ_res, caches)
    (; cache_model, cache_external, cache_propagator, ln_Gx) = caches
    FP = ClassicalDFT.fptype(system.options)
    nd = ClassicalDFT.dimension(system)
    species = system.species
    model = system.model

    ClassicalDFT.δFδρ_res!(system, ρ, δfδρ_res, cache_model...)
    ClassicalDFT.evaluate_external_field!(system, ρ, δfδρ_res, cache_external)
    ClassicalDFT.propagate!(system, ρ, δfδρ_res, cache_propagator)

    chem_pot_res_dens_1 = FP(species.chempot_res[1] + log(species.bulk_density[1]))

    i_groups = model.groups.i_groups[1]
    n_intergroups_1 = model.groups.n_intergroups[1]
    for k in i_groups
        if species.nbeads[1] != 1
            α = findall(n_intergroups_1[k, :] .== 1 .&& species.levels .> species.levels[k])
        else
            α = k
        end
        selectdim(ln_Gx, nd + 1, k) .= chem_pot_res_dens_1 .- selectdim(δfδρ_res, nd + 1, k)
    end

    if any(typeof.(system.external_field) .<: ClassicalDFT.ElectrostaticPotentialModel)
        ep_model = filter(x -> x isa ClassicalDFT.ElectrostaticPotentialModel, system.external_field)[1]
        Z = model.charge

        psi_c = ClassicalDFT.find_ψ_const(system.structure, ep_model, model, exp.(ln_Gx)) / Clapeyron.k_B / system.structure.conditions[2]
        for k in i_groups
            selectdim(ln_Gx, nd + 1, k) .-= psi_c * Z[k]
        end
    end

    clamp!(ln_Gx, -100, 100)

    return nothing
end
