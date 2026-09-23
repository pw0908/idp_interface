# Composite wiring for `LSDFTSystem` (struct defined in `LSDFTSystem.jl`,
# included before `LSDFTNeutral.jl`/`LSDFTIon.jl` since the latter's
# `preallocate_params(system::LSDFTSystem, model::LSIon)` needs the type).
# Mirrors `ClassicalDFT`'s `models/DFT/Electrolyte/base.jl` throughout.

"""
    LSDFTSystem(model::LS, structure::DFTStructure, options::DFTOptions=DFTOptions())

Build the spatial DFT system for an LS-theory sequence. `model` must be built
with `expand=true` (`LS(charges; expand=true)`, see `LSComposite.jl`) so its
`GroupParam` has one group per real bead -- required for `TangentHSPropagator`
to resolve the sequence's actual bond graph.

The long-range mean-field electrostatic potential external field
(`ClassicalDFT.ElectrostaticPotential`) is wired in via `LSElectrostaticPotential.jl`'s
`model::LS`-dispatched methods -- see that file for why new dispatches
(not a widened `LS` type hierarchy or a `ClassicalDFT` change) are used.
"""
function LSDFTSystem(model::LS, structure::ClassicalDFT.DFTStructure, options::ClassicalDFT.DFTOptions=ClassicalDFT.DFTOptions())
    device = options.device
    FP = ClassicalDFT.fptype(options)

    species = ClassicalDFT.get_species(model.neutralmodel, structure)
    ion_species = ClassicalDFT.get_species(model.ionmodel, model.neutralmodel, structure)

    (pressure, temperature) = structure.conditions
    ρbulk = structure.ρbulk
    μres = Clapeyron.VT_chemical_potential_res(model, 1 / sum(ρbulk), temperature, ρbulk / sum(ρbulk)) / Clapeyron.R̄ / temperature
    species.chempot_res .= μres

    L = FP(ClassicalDFT.length_scale(model))
    fields = ClassicalDFT.get_fields(model.neutralmodel, species, structure, device, FP)
    fields_ion = ClassicalDFT.get_fields((model.ionmodel, L), ion_species, structure, device, FP)
    typed_fields = tuple(fields..., fields_ion...)

    propagator = ClassicalDFT.get_propagator(model.neutralmodel, species, structure, device, FP)

    external_field = [ClassicalDFT.ElectrostaticPotential(model, structure, device, FP)]

    NF = ClassicalDFT.compute_field_len(typed_fields, ClassicalDFT.dimension(structure))
    chunksize = Val{NF}()
    return LSDFTSystem(model, species, ion_species, structure, typed_fields, external_field, propagator, options, chunksize)
end

ClassicalDFT.length_scale(model::LS) = ClassicalDFT.length_scale(model.neutralmodel)

"""
    ClassicalDFT._energy_scale(system::LSDFTSystem)

`ClassicalDFT.F_res` integrates the pointwise residual free-energy density
`f_val` (reduced, σ³-normalized -- the same convention `f_res`'s HS/chain/
electrostatic kernels use throughout this project) over `system.structure`'s
grid, whose bounds are in **physical** (meter) units -- `LSDFTSystem`'s
structures are built as `[-halfbox_L*L, halfbox_L*L]` with `L=length_scale`,
exactly like every other `ClassicalDFT` model's own `surface_tension(model,T,x)`
convenience wrapper. Every existing model with a genuinely reduced `f_res`
(`DFTSystem{<:PCSAFTModel}`, `<:SAFTVRMieModel}`, `<:SAFTgammaMieModel}`,
`<:COFFEEModel}`, `<:PeTSModel}`, and `DGTSystem`) declares
`_energy_scale(system) = length_scale(system.model)^3` for exactly this
reason; the generic `_energy_scale(system) = 1.0` fallback (silently used by
`LSDFTSystem`, since it isn't any of those listed types) leaves `F_res`
under-scaled by a factor of `L^3` (~2.7e-29 for `L=σ=3e-10 m`) -- found via
`ClassicalDFT.surface_tension` returning a spurious, `lB`-growing *negative*
value: with `F_res` ~29 orders of magnitude too small, `surface_tension`'s
`F*k_B*T` term was missing essentially all of its (dominant, comparable to
`F_ideal`) residual contribution.
"""
ClassicalDFT._energy_scale(system::LSDFTSystem) = ClassicalDFT.length_scale(system.model)^3

@inline function ClassicalDFT.f_res(::Type{M}, kk, out, n, params, T,
                                     ::Val{NC}, ::Val{ND}) where {NC, ND, M <: LS}
    MN = fieldtype(M, :neutralmodel)
    MI = fieldtype(M, :ionmodel)
    ClassicalDFT.f_res(MN, kk, out, n, params, T, Val(NC), Val(ND))
    ClassicalDFT.f_res(MI, kk, out, n, params, T, Val(NC), Val(ND))
    return nothing
end

"""
    ClassicalDFT.preallocate_params(system::LSDFTSystem)

Merges `LSNeutral`'s and `LSIon`'s params, exactly as `Electrolyte/base.jl`
does: builds a proxy `DFTSystem` for the neutral part (via the raw struct
constructor, bypassing `expand_model`/`build_DFT_system`) to reuse
`preallocate_params(::DFTSystem{<:LSNeutral})` unchanged, then merges in the
ion params (dispatched on `LSIon` directly since it needs `system.ion_species`).
"""
function ClassicalDFT.preallocate_params(system::LSDFTSystem)
    model = system.model
    n_model = model.neutralmodel
    nd = ClassicalDFT.dimension(system)

    n_fields = Base.front(system.fields)
    NF_neutral = ClassicalDFT.compute_field_len(n_fields, nd)
    n_sys = ClassicalDFT.DFTSystem(n_model, system.species, system.structure,
                                    n_fields, nothing, system.propagator,
                                    system.options, Val{NF_neutral}())
    neutral_params, nc = ClassicalDFT.preallocate_params(n_sys)

    ion_params = ClassicalDFT.preallocate_params(system, model.ionmodel)

    return merge(neutral_params, ion_params), nc
end

"""
    ClassicalDFT.preallocate_model(system::LSDFTSystem, ρ)

Byte-for-byte the same as `ClassicalDFT.preallocate_model(::DFTSystem, ρ)`/
`preallocate_model(::ElectrolyteDFTSystem, ρ)` -- the generic Enzyme/KA
preallocator has no model-specific logic at all, but Julia dispatch needs a
method matching `LSDFTSystem`'s own (necessarily distinct) struct type.
"""
function ClassicalDFT.preallocate_model(system::LSDFTSystem, ρ)
    backend = system.options.device
    FP = ClassicalDFT.fptype(system.options)
    nf = ClassicalDFT.length_fields(system)
    ngrid = system.structure.ngrid
    nd = length(ngrid)
    nb = size(ρ, nd + 1)

    n = ClassicalDFT.allocate(backend, FP, ngrid..., nf, nb)
    δf = ClassicalDFT.allocate(backend, FP, ngrid..., nf, nb)
    fill!(δf, 0)
    fft_buf = ClassicalDFT.allocate(backend, FP, ngrid..., nf, nb)

    CT = ClassicalDFT.transform_eltype(system.structure, FP)
    in_buf = ClassicalDFT.allocate(backend, CT, ngrid...)
    out_buf = similar(in_buf)
    tmp = similar(in_buf)
    plan, iplan = ClassicalDFT.build_transform(system.structure, tmp, nd, backend)

    f_val = ClassicalDFT.allocate(backend, FP, ngrid...)
    δf_val = ClassicalDFT.allocate(backend, FP, ngrid...)
    fill!(δf_val, 1)

    params, nc = ClassicalDFT.preallocate_params(system)

    if system.options.ad_mode === :forward || system.options.ad_mode === :forward_batch
        batch = nf * nc
        dn_seeds = ntuple(Val(batch)) do k
            f_idx = (k - 1) ÷ nc + 1
            c_idx = (k - 1) % nc + 1
            seed = ClassicalDFT.allocate(backend, FP, ngrid..., nf, nc)
            fill!(seed, 0)
            fill!(selectdim(selectdim(seed, nd + 1, f_idx), nd + 1, c_idx), 1)
            seed
        end
        df_outs = ntuple(_ -> ClassicalDFT.allocate(backend, FP, ngrid...), Val(batch))
        fwd_cache = (dn_seeds, df_outs, Val(batch))
    else
        fwd_cache = nothing
    end

    ClassicalDFT._warmup_cpu_kernel!(system, backend, n, δf, f_val, δf_val, params, nc, nd, fwd_cache)

    return n, δf, fft_buf, in_buf, out_buf, plan, iplan, params, f_val, δf_val, nc, nd, fwd_cache
end

"""
    ClassicalDFT.F_res(system::LSDFTSystem, ρ)

Same Enzyme-kernel path as `F_res(::Union{DFTSystem,DGTSystem}, ρ)` (not the
"old scalar `f_res(system,model,n)`" fallback `F_res(::AbstractcDFTSystem,ρ)`
uses -- `LSDFTSystem` never defines that 3-arg method, only the type-dispatched
pointwise kernel `f_res(::Type{M},kk,out,n,params,T,Val(NC),Val(ND))`).
"""
function ClassicalDFT.F_res(system::LSDFTSystem, ρ)
    δfδρ_res, cache_model, _, _ = ClassicalDFT.preallocate(system, ρ)
    ClassicalDFT.δFδρ_res!(system, ρ, δfδρ_res, cache_model...)
    f_val = cache_model[9]
    return ClassicalDFT.:∫(f_val, system.structure) / ClassicalDFT._energy_scale(system)
end
