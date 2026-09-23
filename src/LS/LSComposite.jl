"""
    LS{N,I}

`groups` is a genuine struct field (aliasing `neutralmodel.groups`, the same
`GroupParam`/bond graph shared with `ionmodel` by construction -- see
`LSGroups.jl`), not just a forwarded property. This matters specifically
because of `ClassicalDFT`'s `@chain` macro (`src/utils/base.jl`):

```julia
macro chain(component, args...)
    quote
        if hasfield(typeof(model), :groups) && !(typeof(model) <: Clapeyron.HomogcPCPSAFTModel)
            model.groups.i_groups[\$(component)]
        else
            \$(component)
        end
    end |> esc
end
```

`hasfield` is a *declared-field* check (via `fieldnames`), which a
`Base.getproperty` override cannot satisfy -- an earlier version of this
struct forwarded `.groups` via `getproperty` alone, which made *direct*
property access (`model.groups.n_intergroups[1]`, etc., used explicitly
throughout `tangent_hs.jl`'s bottom-up/top-down passes) work correctly, but
left `hasfield(typeof(model),:groups)` `false`. That silently sent every
`@chain(i)` call inside `TangentHSPropagator`'s `propagate!` (the final
chain-bonding-subtraction loop specifically) down the macro's `else` branch,
which returns `component` (`i`, i.e. plain `1` for our single-"component"
composite) instead of `model.groups.i_groups[i]` (all `N` bead indices) --
so that loop only ever iterated bead index `1`, silently leaving every other
bead's `δfδρ_res` completely unpropagated. This was the actual root cause of
the composite (ion-included) bulk-limit validation failure investigated at
length in the project plan; it had nothing to do with `propagate!`'s
algorithm itself. A real field is the only fix `hasfield` will respect
without modifying `ClassicalDFT`.
"""
struct LS{N<:Clapeyron.EoSModel,I<:Clapeyron.EoSModel} <: Clapeyron.EoSModel
    components::Vector{String}
    neutralmodel::N
    ionmodel::I
    idealmodel::Clapeyron.BasicIdeal
    groups::Clapeyron.GroupParam
    charge::Vector{Int}
end

Base.length(model::LS) = 1

"""
    LS(charges; chi=0.0, RSPmodel=Clapeyron.ConstRSP(), seqname="ls_seq", expand=false)

Bulk liquid-state-theory EOS (Zhang et al. 2016) for a single IDP sequence
given as a vector of per-bead charges (e.g. `[1,-1,1,-1,1,-1,1,-1]`, 0 for a
neutral bead). Each sequence is one Clapeyron pseudo-pure component whose
internal bead-by-bead structure is carried by a `GroupParam` (see
`LSGroups.jl`), shared between `neutralmodel::LSNeutral` and
`ionmodel::LSIon`. `a_res(LS,...) = a_res(neutralmodel,...) + a_res(ionmodel,...)`.
`expand` is passed through to `LSNeutral`/`LSIon` (see their docstrings);
Clapeyron's own bulk solvers (Step 1) use `expand=false` (the default),
Step 2/3's spatial DFT functional dispatches on the `expand=true` submodels
directly rather than through this composite.
"""
function LS(charges::AbstractVector{<:Integer}; chi::Float64=0.0, RSPmodel=Clapeyron.ConstRSP(), seqname::String="ls_seq", expand::Bool=false)
    neutralmodel = LSNeutral(charges; chi=chi, seqname=seqname, expand=expand)
    ionmodel = LSIon(charges; RSPmodel=RSPmodel, seqname=seqname, expand=expand)
    return LS([seqname], neutralmodel, ionmodel, Clapeyron.BasicIdeal(), neutralmodel.groups, ionmodel.params.Z)
end

Clapeyron.a_res(model::LS, V, T, z) = Clapeyron.a_res(model.neutralmodel, V, T, z) + Clapeyron.a_res(model.ionmodel, V, T, z)

function Clapeyron.lb_volume(model::LS, z)
    p = model.neutralmodel.params
    return (π/6) * Clapeyron.N_A * z[1] * p.N * p.sigma^3
end

function Clapeyron.T_scale(model::LS, z)
    p = model.ionmodel.params
    ϵr = Clapeyron.dielectric_constant(model.ionmodel.RSPmodel, 1.0, 298.15, z)
    # nominal scale: the temperature at which the Bjerrum length equals one
    # bead diameter (lB=1 in the reduced units used throughout this model).
    return Clapeyron.e_c^2 / (4π * Clapeyron.ϵ_0 * ϵr * p.sigma * Clapeyron.k_B)
end
