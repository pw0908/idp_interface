# Long-range mean-field electrostatic potential external field for `LS`,
# needed for Step 3's interfacial potential profiles.
#
# `ClassicalDFT.ElectrostaticPotential`'s constructor and the two functions
# that drive it (`evaluate_external_field!`, `find_ψ_const`,
# `Electrolyte/electrostatic_potential.jl`) are typed on
# `model::Clapeyron.ElectrolyteModel`, which `LS` does not subtype (it's a
# plain `Clapeyron.EoSModel`) -- so they never dispatch for `LS` as written.
# Their bodies, however, only ever touch `model.ionmodel.RSPmodel` and
# `model.charge` via plain property access (no `hasfield`-based type
# introspection, unlike the `@chain`/`propagate!` issue resolved in
# `LSComposite.jl`) -- both of which `LS` now has as genuine fields
# (`ionmodel` directly, `charge` added specifically for this). So rather than
# widening `LS`'s type hierarchy (a much bigger, riskier change -- Clapeyron
# defines a family of generic `ESElectrolyteModel` fallbacks for `lb_volume`,
# `x0_volume_liquid/gas`, `mw`, `p_scale`, `T_scale`, `dielectric_constant`,
# etc. that would suddenly become live candidates for `LS` and would need
# re-auditing against Step 1's already-validated behavior), these three
# methods are reproduced verbatim below with only the type annotation
# changed from `ElectrolyteModel`/`ElectrolyteDFTSystem` context to `LS` --
# exactly the "new functionality via multiple dispatch on new types" pattern
# used throughout this project, not a `ClassicalDFT` modification.

"""
    ClassicalDFT.ElectrostaticPotential(model::LS, structure, backend, FP)

Precomputes the Fourier-space Coulomb Green's-function kernel (scaled by the
solvent's dielectric constant, `model.ionmodel.RSPmodel`) convolved with the
ionic charge density profile during `evaluate_external_field!`. Verbatim copy
of `ClassicalDFT.ElectrostaticPotential(model::ElectrolyteModel,...)` --
see this file's header comment for why a new dispatch, not a `ClassicalDFT`
change or a widened `LS` type hierarchy, is used.
"""
function ClassicalDFT.ElectrostaticPotential(model::LS, structure::ClassicalDFT.DFTStructure, backend::ClassicalDFT.Backend, ::Type{FP}=Float64) where FP<:AbstractFloat
    (_, temperature) = structure.conditions
    ρbulk = structure.ρbulk
    ϵ_r = Clapeyron.dielectric_constant(model.ionmodel.RSPmodel, 1.0, temperature, ρbulk)
    ngrid = structure.ngrid
    nd = length(ngrid)

    ω = ClassicalDFT.structure_ω(structure, backend, FP)

    ω_norm = ClassicalDFT.allocate(ClassicalDFT.CPU(), FP, ngrid...)

    for kk in CartesianIndices(ngrid)
        ω_norm[kk] = LinearAlgebra.norm(@view(ω[Tuple(kk)..., :]))
    end

    ω̄ = ClassicalDFT.allocate(backend, FP, ngrid...)
    copyto!(ω̄, ClassicalDFT.Adapt.adapt(typeof(ω̄), ω_norm))

    _c = FP(Clapeyron.N_A * Clapeyron.e_c^2 / Clapeyron.ϵ_0) / FP(ϵ_r)
    Ω = @. (!iszero(ω̄)) / (FP(4) * π * π * ω̄^2 + iszero(ω̄)) * _c
    return ClassicalDFT.ElectrostaticPotential(ϵ_r, Ω)
end

"""
    ClassicalDFT.evaluate_external_field!(structure, external_field, model::LS, ρ, δfδρ_res, P, iP, Vext)

Verbatim copy of `ClassicalDFT.evaluate_external_field!(...,model::ElectrolyteModel,...)`,
using `model.charge` (now a genuine `LS` field, one entry per bead for an
`expand=true` model -- matches `ρ`'s per-bead layout exactly).
"""
function ClassicalDFT.evaluate_external_field!(structure::ClassicalDFT.DFTStructure, external_field::ClassicalDFT.ElectrostaticPotentialModel, model::LS, ρ, δfδρ_res, P, iP, Vext)
    temperature = structure.conditions[2]
    Z = model.charge
    nd = length(structure.ngrid)
    nbeads = length(Z)

    for i in 1:nbeads
        if i == 1
            Vext .= selectdim(ρ, nd + 1, i) * Z[i]
        else
            Vext .+= selectdim(ρ, nd + 1, i) * Z[i]
        end
    end

    ϵ_r = external_field.ϵ_r
    map = external_field.map

    ClassicalDFT.convolve!(Vext, Vext, map, P, iP, Vext)

    for i in 1:nbeads
        selectdim(δfδρ_res, nd + 1, i) .+= Z[i] * Vext / Clapeyron.k_B / temperature
    end
end

"""
    ClassicalDFT.find_ψ_const(structure, external_field, model::LS, ρ)

Verbatim copy of `ClassicalDFT.find_ψ_const(...,model::ElectrolyteModel,...)`.
"""
function ClassicalDFT.find_ψ_const(structure::ClassicalDFT.DFTStructure, external_field::ClassicalDFT.ElectrostaticPotentialModel, model::LS, ρ)
    Z = model.charge
    nbeads = length(Z)
    nd = length(structure.ngrid)
    ψ0 = 0.0
    while true
        q = 0.0
        dq = 0.0
        for i in 1:nbeads
            q += sum(selectdim(ρ, nd + 1, i) * Z[i]) * exp(-Z[i] * ψ0)
            dq -= sum(selectdim(ρ, nd + 1, i)) * Z[i]^2 * exp(-Z[i] * ψ0)
        end
        ψ0 -= q / dq
        if abs(q) < 1e-6
            break
        end
    end
    return ψ0 * Clapeyron.k_B * structure.conditions[2]
end
