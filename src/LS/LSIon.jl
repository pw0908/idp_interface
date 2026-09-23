struct LSIon{ϵ} <: Clapeyron.EoSModel
    components::Vector{String}
    groups::Clapeyron.GroupParam
    params::LSIonParam
    RSPmodel::ϵ
end

Base.length(model::LSIon) = 1

"""
    LSIon(charges; RSPmodel=Clapeyron.ConstRSP(), seqname="ls_seq", expand=false)

Charge-induced half of the LS-theory bulk EOS for a single IDP sequence:
the restricted-primitive-model (equal bead size) Blum-MSA electrostatic
free energy, plus the charge-induced correction to the TPT1 chain term.
Shares its bead/bond data with `LSNeutral`; `LS` composes the two. See
`LSNeutral`'s docstring for the `expand` keyword (compact vs. per-bead
`GroupParam` -- `a_res`'s value is identical either way).
"""
function LSIon(charges::AbstractVector{<:Integer}; RSPmodel=Clapeyron.ConstRSP(), seqname::String="ls_seq", expand::Bool=false)
    groups, group_names, Z = expand ? expand_ls_groups(charges; seqname=seqname) : build_ls_groups(charges; seqname=seqname)
    N = length(charges)
    Npm, Nnc = expand ? count_bond_types(charges) : count_bond_types_from_groups(group_names, groups.n_intergroups[1])
    params = LSIonParam(Z, N, Npm, Nnc, LS_SIGMA)
    return LSIon([seqname], groups, params, RSPmodel)
end

function Clapeyron.a_res(model::LSIon, V, T, z)
    p = model.params
    ρ★ = ls_group_densities(model.groups, V, z, p.sigma)
    ρtot = sum(ρ★)

    ϵr = Clapeyron.dielectric_constant(model.RSPmodel, V, T, z)
    lB = Clapeyron.e_c^2 / (4π * Clapeyron.ϵ_0 * ϵr * p.sigma * Clapeyron.k_B * T)

    # Restricted primitive model (all beads the same size): the Blum-MSA
    # screening parameter Γ_MSA has the closed-form solution below -- no
    # fixpoint iteration needed (unlike Clapeyron's general/asymmetric-size
    # MSA.jl), matching Helmholtz_LS.jl exactly. _safe_pos guards against
    # transiently unphysical (T<0 or V<0) trial points from derivative-based
    # solvers (crit_pure, saturation_pressure) -- inert within the physical
    # domain.
    κ_MSA = sqrt(_safe_pos(4π * lB * sum(ρ★ .* abs.(p.Z) .* p.Z.^2)))
    Γ_MSA = (-1 + sqrt(1 + 2κ_MSA)) / 2
    fel = -Γ_MSA^3 * (2/3 + Γ_MSA) / π

    # log(ypp) = log(yhs) - lB/(1+Γ_MSA)^2 + lB, log(ypm) = log(yhs) +
    # lB/(1+Γ_MSA)^2 - lB (Helmholtz_LS.jl); substituting into fch(Γ_MSA)
    # and subtracting fch(Γ_MSA=0) = ρchain*(1-N)*log(yhs), every log(yhs)
    # term cancels exactly, leaving this closed form -- no need to compute
    # yhs/ypp/ypm/log at all. Verified numerically against the direct
    # ypp/ypm-based computation to machine precision.
    ρchain = ρtot / p.N
    N, Npm, Nnc = p.N, p.Npm, p.Nnc
    Δfch = ρchain * (1 + 2Npm + Nnc - N) * lB * (1 - 1 / (1 + Γ_MSA)^2)

    # See LSNeutral.a_res's matching fix: Clapeyron's a_res is per mole of
    # chain (ρchain), not per mole of monomer (ρtot = N*ρchain).
    return (fel + Δfch) / ρchain
end
