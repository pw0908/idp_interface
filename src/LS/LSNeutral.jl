struct LSNeutral <: Clapeyron.EoSModel
    components::Vector{String}
    groups::Clapeyron.GroupParam
    params::LSNeutralParam
end

Base.length(model::LSNeutral) = 1

"""
    LSNeutral(charges; chi=0.0, seqname="ls_seq", expand=false)

Neutral (density-only) half of the LS-theory bulk EOS for a single IDP
sequence: hard-sphere (BMCSL) + the Γ_MSA=0 limit of the TPT1 chain term +
an inert Flory `chi` crowder cross term. See `LSIon` for the charge-induced
half; `LS` composes the two.

`a_res`'s value depends only on `N`/`Npm`/`Nnc` (scalars) and the *total*
reduced density (invariant to how beads are grouped), so it is identical
whether built from the compact (`expand=false`, Step 1's bulk EOS) or
per-bead-resolved (`expand=true`, Step 2/3's spatial DFT, via
`expand_ls_groups`) `GroupParam` -- only `model.groups`/`model.params.Z`'s
*shape* differs between the two.
"""
function LSNeutral(charges::AbstractVector{<:Integer}; chi::Float64=0.0, seqname::String="ls_seq", expand::Bool=false)
    groups, group_names, Z = expand ? expand_ls_groups(charges; seqname=seqname) : build_ls_groups(charges; seqname=seqname)
    N = length(charges)
    Npm, Nnc = expand ? count_bond_types(charges) : count_bond_types_from_groups(group_names, groups.n_intergroups[1])
    params = LSNeutralParam(Z, N, Npm, Nnc, LS_SIGMA, chi)
    return LSNeutral([seqname], groups, params)
end

function Clapeyron.a_res(model::LSNeutral, V, T, z)
    p = model.params
    ρ★ = ls_group_densities(model.groups, V, z, p.sigma)
    ρtot = sum(ρ★)
    η = (π/6) * ρtot
    fhs = 6 * η^2 * (4 - 3η) / (π * (1 - η)^2)

    # g_hs: the same hard-sphere contact-value correlation Clapeyron's
    # HeterogcPCPSAFT itself uses in its bulk a_hc (g_hs[k,l] = c1 + r*c2 +
    # r^2*c3, r=dk*dl/(dk+dl); here r=1/2 since every bead shares one
    # diameter) -- verified algebraically identical to ClassicalDFT's
    # _f_hc_bonds' yᵈᵈ (the DFT chain-term kernel this project reuses
    # unmodified). This is NOT the same closed form as Zhang et al. 2016's
    # own yhs=(2+η)/(2(1-η)²) (Helmholtz_LS.jl) -- those differ by a few
    # percent at typical η, which is exactly what made this bulk model and
    # the reused DFT kernel subtly inconsistent. _safe_pos guards against
    # transiently unphysical (V<0) trial points from derivative-based
    # solvers (crit_pure, saturation_pressure) -- inert within the physical
    # domain.
    c1 = 1 / (1 - η)
    c2 = 3η / (1 - η)^2
    c3 = 2η^2 / (1 - η)^3
    r = 0.5
    g_hs = _safe_pos(c1 + r * c2 + r^2 * c3)

    ρchain = ρtot / p.N
    N, Npm, Nnc = p.N, p.Npm, p.Nnc
    # Γ_MSA=0 limit: every bond's contact value collapses to the same g_hs
    # (Npm/Nnc terms cancel exactly), giving (1-N)*log(g_hs) total.
    fch0 = ρchain * (1 - N) * log(g_hs)

    fchi = 0.0 # inert until a neutral crowder species is introduced

    # Clapeyron's a_res convention is *per mole of Clapeyron component* --
    # here, per mole of whole CHAIN (z[1] counts chains, length(model)==1),
    # not per mole of monomer/bead. fhs/fch0/fchi are energy densities
    # (per unit volume), so the correct normalization is ρchain, not ρtot
    # (=N*ρchain, the aggregate bead density) -- dividing by ρtot as this
    # function did previously returned a value N times too small (caught by
    # Step 2's DFT bulk-limit chemical-potential comparison against
    # ClassicalDFT's δFδρ_res, which showed a clean, uniform N× ratio; not
    # caught by Step 1's own validation since its reference computation used
    # the same wrong ρtot convention on both sides). Coexistence/critical-point
    # equality conditions (p(Vl)=p(Vv), μ(Vl)=μ(Vv)) are invariant to this
    # T,V-independent constant rescaling, so Step 1's phase diagrams are
    # unaffected -- only *absolute* a_res/pressure/chemical-potential values
    # were ever wrong.
    return (fhs + fch0 + fchi) / ρchain
end
