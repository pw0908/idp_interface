# Neutral (density-only) half of the LS-theory spatial DFT functional: FMT
# hard-sphere + the Γ_MSA=0 limit of the per-bond TPT1 chain term. Dispatches
# on `LSNeutral` built with `expand=true` (see `LSGroups.jl`'s `expand_ls_groups`)
# so every real bead is its own group -- mirrors
# `ClassicalDFT`'s `hetero_gcPPCSAFT.jl`, minus the PC-SAFT dispersion term LS
# theory has no equivalent of.

struct LSDFTNeutralSpecies <: ClassicalDFT.DFTSpecies
    nbeads::Vector{Int64}
    size::Vector{Float64}
    levels::Vector{Int64}
    bulk_density::Vector{Float64}
    chempot_res::Vector{Float64}
end

"""
    ClassicalDFT.get_species(model::LSNeutral, structure)

One bead per `model.groups` entry, all diameter `model.params.sigma` (LS theory's
beads are strictly hard, so unlike PCSAFT's Barker-Henderson `d(model,...)`, no
temperature-dependent softening integral is needed -- `size` is just `sigma`
repeated). `chempot_res` is a placeholder overwritten by `LSDFTSystem`'s
constructor with the *composite* `LS` model's chemical potential (mirrors
`ElectrolyteDFTSystem`'s `species.chempot_res .= μres` pattern).
"""
function ClassicalDFT.get_species(model::LSNeutral, structure::ClassicalDFT.DFTStructure)
    nbeads = length.(model.groups.groups)
    nc_groups = sum(nbeads)
    size = fill(model.params.sigma, nc_groups)
    levels = ClassicalDFT.compute_levels(model)
    μres = zeros(length(nbeads))
    return LSDFTNeutralSpecies(nbeads, size, levels, structure.ρbulk, μres)
end

"""
    ClassicalDFT.get_fields(model::LSNeutral, species, structure, backend, FP)

Field layout (5 fields, reduced units -- lengths divided by `L=length_scale(model)`,
matching PCSAFT.jl/hetero_gcPPCSAFT.jl's convention):
  1        : ρ (unweighted)               -- used by the per-bond chain term
  2        : ∫ρdz  with 0.5*d → n₀,n₁,n₂  -- FMT
  3        : ∫ρz²dz with 0.5*d → n₃       -- FMT
  4..3+ND  : ∫ρzdz with 0.5*d → nᵥ        -- FMT
  4+ND     : ∫ρz²dz with d    → ρ̄hc       -- chain term ζ₂/ζ₃
No dispersion/polar field: LS theory has neither.
"""
function ClassicalDFT.get_fields(model::LSNeutral, species::ClassicalDFT.DFTSpecies, structure::ClassicalDFT.DFTStructure, backend::ClassicalDFT.Backend, ::Type{FP}) where FP<:AbstractFloat
    nb = sum(species.nbeads)
    ngrid = structure.ngrid
    L = ClassicalDFT.length_scale(model)
    ω = ClassicalDFT.structure_ω(structure, backend, FP)
    d = species.size ./ L
    return (ClassicalDFT.SWeightedDensity(:ρ, zeros(nb), ω, ngrid, backend, model),
            ClassicalDFT.SWeightedDensity(:∫ρdz, 0.5*d, ω, ngrid, backend, model),
            ClassicalDFT.SWeightedDensity(:∫ρz²dz, 0.5*d, ω, ngrid, backend, model),
            ClassicalDFT.VWeightedDensity(:∫ρzdz, 0.5*d, ω, ngrid, backend, model),
            ClassicalDFT.SWeightedDensity(:∫ρz²dz, d, ω, ngrid, backend, model))
end

"""
    ClassicalDFT.get_propagator(model::LSNeutral, species, structure, backend, FP)

Full `TangentHSPropagator`: bead *position* along the specific sequence is the
entire point of this project, so every real bond is tracked individually
(never `IdealPropagator`/a folded bulk-homopolymer treatment).
"""
function ClassicalDFT.get_propagator(model::LSNeutral, species::ClassicalDFT.DFTSpecies, structure::ClassicalDFT.DFTStructure, backend::ClassicalDFT.Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    return ClassicalDFT.TangentHSPropagator(model, species, structure, backend, FP)
end

ClassicalDFT.length_scale(model::LSNeutral) = model.params.sigma

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────

"""
Pointwise residual free energy for `LSNeutral`: FMT hard-sphere + per-bond
chain term (every bond uses `y_hs`, i.e. the Γ_MSA=0 limit -- see `LSNeutral.a_res`
in `LSGroups.jl`'s bulk model for the closed-form identity this reduces to in
the uniform limit). `f_chi` (inert Flory crowder cross term) is 0 until a
neutral crowder species exists, so it's omitted here entirely rather than
computed as a literal no-op.

Field layout matches `get_fields` above; `F2=2` (n₀/n₁/n₂ source is field 2),
so this is a direct call into `ClassicalDFT.f_hs`/`ClassicalDFT._f_hc_bonds`
(the same functions `HeterogcPCPSAFT` uses) -- no reimplementation.
"""
@inline function ClassicalDFT.f_res(::Type{M}, kk, out, n, params, T,
                                     ::Val{NC}, ::Val{ND}) where {NC, ND, M <: LSNeutral}
    res_hs, = ClassicalDFT.f_hs(n, params.m, params.HSd, kk, Val(NC), Val(ND), Val(2))
    res_bond = f_bond_neutral(M, kk, n, params, T, Val(NC), Val(ND))
    out[kk] = res_hs + res_bond
    return nothing
end

@inline function f_bond_neutral(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{ND}) where {NC, ND, M <: LSNeutral}
    HSd = params.HSd
    m_seg = params.m
    bond_k = params.bond_k
    bond_l = params.bond_l

    FP = eltype(n)
    idx_ζ = 4 + ND
    ζ₃ = zero(FP); ζ₂ = zero(FP)
    @inbounds for i in 1:NC
        mi = m_seg[i]; di = HSd[i]; ρ̄hci = n[kk, idx_ζ, i]
        ζ₃ += mi * ρ̄hci
        ζ₂ += mi * ρ̄hci / di
    end
    ζ₃ /= 8; ζ₂ /= 8
    inv1ζ₃ = 1 / (1 - ζ₃)

    return ClassicalDFT._f_hc_bonds(n, bond_k, bond_l, HSd, kk, ζ₂, inv1ζ₃)
end

"""
    preallocate_params(system::ClassicalDFT.DFTSystem{<:LSNeutral})

Bond list + reduced-units `HSd`/`sigma`, exactly `hetero_gcPPCSAFT.jl`'s
pattern (minus dispersion's `epsilon`/`nbeads_for_group`, which LS theory
has no use for; `m` is all-ones since each group is a single hard sphere).
"""
function ClassicalDFT.preallocate_params(system::ClassicalDFT.DFTSystem{<:LSNeutral})
    backend = system.options.device
    FP = ClassicalDFT.fptype(system.options)
    model = system.model
    nc_groups = sum(system.species.nbeads)

    bond_k_list = Int[]
    bond_l_list = Int[]
    i_groups = model.groups.i_groups[1]
    n_intergroups_1 = model.groups.n_intergroups[1]
    for k in i_groups
        for l in findall(n_intergroups_1[k, :] .== 1)
            push!(bond_k_list, k)
            push!(bond_l_list, l)
        end
    end

    n_bonds = length(bond_k_list)
    bond_k_t = ntuple(ib -> bond_k_list[ib], n_bonds)
    bond_l_t = ntuple(ib -> bond_l_list[ib], n_bonds)

    L = ClassicalDFT.length_scale(model)
    HSd_local = system.species.size ./ L

    params = (;
        HSd = ClassicalDFT.adapt_to_device(backend, FP, HSd_local),
        m = ClassicalDFT.adapt_to_device(backend, FP, ones(nc_groups)),
        bond_k = bond_k_t,
        bond_l = bond_l_t,
    )
    return params, nc_groups
end
