# Charge-induced half of the LS-theory spatial DFT functional: the local
# (restricted-primitive-model) MSA/Blum electrostatic free energy, plus the
# per-bond ±Δ(r) correction to the chain term (see `LSIon.a_res` in
# `LSGroups.jl`'s bulk model for the closed-form identity this is the
# spatially-resolved generalization of). Mirrors
# `ClassicalDFT`'s `models/DFT/Electrolyte/DH.jl`, substituting LS theory's
# closed-form Γ_MSA=(-1+sqrt(1+2κ_MSA))/2 (exact for equal-sized beads) for
# DH's Padé-approximant χ(x) term (needed there for possibly-unequal ion
# sizes) -- no Padé approximant/fixpoint iteration needed here.

const _nti = ClassicalDFT._nti

struct LSDFTIonSpecies <: ClassicalDFT.DFTSpecies
    nbeads::Vector{Int64}
    charges::Vector{Float64}
    size::Vector{Float64}
    levels::Vector{Int64}
    bulk_density::Vector{Float64}
end

"""
    ClassicalDFT.get_species(model::LSIon, neutralmodel::LSNeutral, structure)

One entry per bead (same `model.groups` as `neutralmodel`); charges come
straight from `model.params.Z` (already per-bead for an `expand=true` model)
-- unlike `DH.jl`'s `get_species`, no separate `charges` argument or
`get_sigma` Barker-Henderson call is needed (LS beads are strictly hard, all
diameter `model.params.sigma`).

`levels` (via `compute_levels`, same as `LSDFTNeutralSpecies`) is carried
here too even though `LSDFTComposite`'s constructor never uses it (the
composite always propagates once, via `LSDFTNeutralSpecies.levels`, shared
across neutral+ion fields) -- it's needed for a standalone `DFTSystem{<:LSIon}`
built with a real `TangentHSPropagator` (rather than the default
`IdealPropagator`) to work at all, e.g. for isolated testing/debugging of the
electrostatic term on a bonded (not just single-ion) sequence.
"""
function ClassicalDFT.get_species(model::LSIon, neutralmodel::LSNeutral, structure::ClassicalDFT.DFTStructure)
    Z = Float64.(model.params.Z)
    nc_groups = length(Z)
    size = fill(model.params.sigma, nc_groups)
    levels = ClassicalDFT.compute_levels(model)
    return LSDFTIonSpecies(ones(Int64, nc_groups), Z, size, levels, structure.ρbulk)
end

"""
    ClassicalDFT.get_fields(tup::Tuple{<:LSIon,FP}, species, structure, backend, FP)

One extra `∫ρdz` field (on top of `LSDFTNeutral`'s 5), smoothed over a
`(σ/2 + 1/κ_MSA_bulk)` width -- exactly `DH.jl`'s `get_fields` pattern, with
`κ_MSA_bulk` from LS theory's own (not DH's) screening-length formula
(`LSIon.a_res`, `LSGroups.jl`). `L` must be the *neutral* model's
`length_scale` (shared with `LSDFTNeutral`'s fields) since only one global
`_energy_scale`/`L^3` correction applies to the combined `F_res` -- see
`LSDFTSystem`'s constructor, which passes it explicitly.
"""
function ClassicalDFT.get_fields(tup::Tuple{<:LSIon,FP}, species::ClassicalDFT.DFTSpecies, structure::ClassicalDFT.DFTStructure, backend::ClassicalDFT.Backend, ::Type{FP}) where FP<:AbstractFloat
    ionmodel, L = tup
    (pressure, temperature) = structure.conditions
    ρbulk = structure.ρbulk
    ngrid = structure.ngrid
    p = ionmodel.params
    v = 1 / sum(ρbulk)

    ρ★ = ls_group_densities(ionmodel.groups, v, ρbulk, p.sigma)
    ϵr = Clapeyron.dielectric_constant(ionmodel.RSPmodel, v, temperature, ρbulk)
    lB = Clapeyron.e_c^2 / (4π * Clapeyron.ϵ_0 * ϵr * p.sigma * Clapeyron.k_B * temperature)
    κ_MSA_bulk = sqrt(_safe_pos(4π * lB * sum(ρ★ .* abs.(p.Z) .* p.Z .^ 2)))

    ω = ClassicalDFT.structure_ω(structure, backend, FP)
    # κ_MSA_bulk is dimensionless (built from lB and ρ★, both already reduced),
    # so 1/κ_MSA_bulk is ALREADY a σ-reduced length -- unlike p.sigma/2 (a raw
    # physical length that still needs the /L reduction). Previously both terms
    # were summed then divided by L together, silently re-reducing the already-
    # reduced 1/κ_MSA_bulk term by an extra factor of 1/L (~3e9), inflating the
    # smoothing kernel width to ~1e8 σ -- found via `sys.fields[end].width`
    # printing an absurd O(1e8) value; effectively an all-box-smearing kernel,
    # explaining the box-size-independent chemical-potential mismatch seen in
    # the Step 3 interfacial solve.
    width = fill(p.sigma / (2L) + 1 / κ_MSA_bulk, length(p.Z))
    return (ClassicalDFT.SWeightedDensity(:∫ρdz, width, ω, ngrid, backend, L),)
end

function ClassicalDFT.get_fields(model::LSIon, species::ClassicalDFT.DFTSpecies, structure::ClassicalDFT.DFTStructure, backend::ClassicalDFT.Backend, ::Type{FP}) where FP<:AbstractFloat
    L = FP(ClassicalDFT.length_scale(model))
    return ClassicalDFT.get_fields((model, L), species, structure, backend, FP)
end

"""
Ions carry no independent connectivity of their own -- the beads' chain
connectivity is entirely handled by `LSDFTNeutral`'s `TangentHSPropagator`,
acting on the same physical beads. Mirrors `DH.jl`'s `get_propagator`.
"""
function ClassicalDFT.get_propagator(model::LSIon, species::ClassicalDFT.DFTSpecies, structure::ClassicalDFT.DFTStructure, backend::ClassicalDFT.Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    return ClassicalDFT.IdealPropagator()
end

ClassicalDFT.length_scale(model::LSIon) = model.params.sigma

# ── Enzyme / KernelAbstractions kernel support ──────────────────────────────

"""
    preallocate_params(system::LSDFTSystem, model::LSIon)

Mirrors `DH.jl`'s `preallocate_params(system::ElectrolyteDFTSystem, model::DHModel)`:
precomputes the bulk dielectric constant (`ConstRSP` only, so this is exact --
no per-point recomputation needed), and per-bead `Z`/smoothing-`width` tuples.

Additionally builds the per-bond charge-combination sign tuple
(`ion_bond_sign`) for the closed-form `±Δ(r)` chain-term correction: `+1` for
a same-charge bonded pair, `-1` for unlike-charge, `0` if either bead is
neutral. Mirrors `_f_hc_bonds`'/`LSDFTNeutral.jl`'s `bond_k`/`bond_l`
construction exactly (each physical bond appears twice, once per direction),
so `f_bond_ion`'s `-ρ/2·sign·Δ` sum collapses to the bulk closed form
`ρ_chain·(1+2Npm+Nnc-N)·Δ` in the uniform limit (`test/test_DFT_bulk.jl`).
"""
function _ls_ion_params(model::LSIon, FP, width_vec, L, NF_neutral)
    Z_vec = Float64.(model.params.Z)
    nc = length(Z_vec)

    Z_t = ntuple(i -> FP(Z_vec[i]), nc)
    w_t = ntuple(i -> FP(width_vec[i]), nc)

    bond_k_list = Int[]
    sign_list = Int[]
    i_groups = model.groups.i_groups[1]
    n_intergroups_1 = model.groups.n_intergroups[1]
    for k in i_groups
        for l in findall(n_intergroups_1[k, :] .== 1)
            push!(bond_k_list, k)
            Zk, Zl = Z_vec[k], Z_vec[l]
            push!(sign_list, (Zk == 0 || Zl == 0) ? 0 : (Zk * Zl > 0 ? 1 : -1))
        end
    end
    n_bonds = length(bond_k_list)
    bond_k_t = ntuple(ib -> bond_k_list[ib], n_bonds)
    bond_sign_t = ntuple(ib -> FP(sign_list[ib]), n_bonds)

    return (;
        ls_Z = Z_t,
        ls_width = w_t,
        ls_sigma = FP(model.params.sigma),
        ls_L = FP(L),
        ls_nf_neutral = Val(NF_neutral),
        ion_bond_k = bond_k_t,
        ion_bond_sign = bond_sign_t,
    )
end

function ClassicalDFT.preallocate_params(system::LSDFTSystem, model::LSIon)
    nd = ClassicalDFT.dimension(system)
    FP = ClassicalDFT.fptype(system.options)
    NF_neutral = ClassicalDFT.compute_field_len(Base.front(system.fields), nd)
    temperature = system.structure.conditions[2]
    ρbulk_ion = system.ion_species.bulk_density
    eps_r = FP(Clapeyron.dielectric_constant(model.RSPmodel, 1 / sum(ρbulk_ion), temperature, ρbulk_ion))
    L = ClassicalDFT.length_scale(system.model)
    width_vec = last(system.fields).width

    base = _ls_ion_params(model, FP, width_vec, L, NF_neutral)
    return merge((; ls_eps_r = eps_r), base)
end

"""
    ClassicalDFT.preallocate_params(system::DFTSystem{<:LSIon})

Standalone counterpart to the `LSDFTSystem`-composite method above, for
testing/debugging `LSIon`'s electrostatic term in isolation (no neutral
model, no `TangentHSPropagator` -- `get_propagator(::LSIon,...)` returns
`IdealPropagator`, so a `DFTSystem{<:LSIon}` never invokes chain-bonding
machinery at all). The ion field is field 1 directly (`ls_nf_neutral=Val(0)`),
since there is no neutral field prepended.
"""
function ClassicalDFT.preallocate_params(system::ClassicalDFT.DFTSystem{<:LSIon})
    model = system.model
    FP = ClassicalDFT.fptype(system.options)
    temperature = system.structure.conditions[2]
    ρbulk_ion = system.species.bulk_density
    eps_r = FP(Clapeyron.dielectric_constant(model.RSPmodel, 1 / sum(ρbulk_ion), temperature, ρbulk_ion))
    L = ClassicalDFT.length_scale(model)
    width_vec = system.fields[1].width

    base = _ls_ion_params(model, FP, width_vec, L, 0)
    params = merge((; ls_eps_r = eps_r), base)
    return params, length(model.params.Z)
end

@inline function ClassicalDFT.f_res(::Type{M}, kk, out, n, params, T,
                                     ::Val{NC}, ::Val{ND}) where {NC, ND, M <: LSIon}
    out[kk] += f_el_and_bond_ion(M, kk, n, params, T, Val(NC), params.ls_nf_neutral)
    return nothing
end

"""
GPU/Enzyme-compatible local MSA electrostatic free energy plus the per-bond
`±Δ(r)` chain correction at grid point `kk`.

`NF_NEUTRAL` locates the ion field (`F_ion = NF_NEUTRAL+1`) in `n`, exactly
like `f_dh`. Unlike `f_dh` -- which un-inflates `n[]` by `_NA = N_A*L^3`
because its physically-scaled χ(x)/κ formula runs in true SI units (matching
`DHModel`'s own physical-units convention) -- `LS`'s electrostatic formulas
(`Γ_MSA`, `κ_MSA`, `LSIon.a_res` in `LSGroups.jl`) run entirely in the same
`ρ★=N_Azσ³/V`-reduced units `LSDFTNeutral`'s FMT/chain terms already use
unmodified. `n[kk,F_ion,i]/(wi*2)` and `n[kk,1,k]` already equal that reduced
`ρ★` directly (confirmed numerically against `ls_group_densities` at bulk
density) -- no `_NA` conversion needed at all, matching `LSDFTNeutral`'s
convention. (An earlier version of this function copied `f_dh`'s `_NA`
round-trip verbatim; that inflated the density into physical units before the
*nonlinear* `Γ_MSA`/`κ_MSA` formulas, which does not cancel correctly under
Enzyme differentiation even though the final `*_NA` restored the primal
value's rough scale -- caught by `test/test_DFT_bulk.jl`'s bulk-limit check.)
"""
@inline function f_el_and_bond_ion(::Type{M}, kk, n, params, T, ::Val{NC}, ::Val{NF_NEUTRAL}) where {M, NC, NF_NEUTRAL}
    FP = eltype(n)
    F_ion = NF_NEUTRAL + 1
    ε_r = params.ls_eps_r
    σ = params.ls_sigma
    _ec = FP(Clapeyron.e_c); _kB = FP(Clapeyron.k_B); _ϵ0 = FP(Clapeyron.ϵ_0)
    lB = _ec * _ec / (4 * FP(π) * _ϵ0 * ε_r * σ * _kB * T)

    I = zero(FP)
    @inbounds for i in 1:NC
        Zi = _nti(params.ls_Z, i)
        wi = _nti(params.ls_width, i)
        ρi = n[kk, F_ion, i] / (wi * 2)
        I += ρi * abs(Zi) * Zi * Zi
    end
    κ_MSA = sqrt(max(4 * FP(π) * lB * I, zero(FP)))
    Γ_MSA = (-1 + sqrt(1 + 2κ_MSA)) / 2
    fel = -Γ_MSA * Γ_MSA * Γ_MSA * (FP(2) / 3 + Γ_MSA) / FP(π)

    Δ = lB * (1 - 1 / (1 + Γ_MSA)^2)
    res_bond = f_bond_ion(n, params.ion_bond_k, params.ion_bond_sign, kk, Δ)

    return fel + res_bond
end

@inline function f_bond_ion(n, bond_k::NTuple{NB, Int}, bond_sign::NTuple{NB}, kk, Δ) where NB
    FP = typeof(Δ)
    res = zero(FP)
    @inbounds for ib in 1:NB
        k = _nti(bond_k, ib)
        s = _nti(bond_sign, ib)
        ρk = n[kk, 1, k]
        res -= ρk / 2 * s * Δ
    end
    return res
end
