"""
Bead diameter, shared by every bead in every sequence (Section "Method" of
idp_interface.md: "only implement the simplified forms of the equations
which assume all beads are of the same size").
"""
const LS_SIGMA = 3e-10 # m

struct LSNeutralParam <: Clapeyron.EoSParam
    Z::Vector{Int}    # charge per group, aligned with groups.flattenedgroups
    N::Int            # chain length (number of beads)
    Npm::Int          # total unlike-charge bonds in the chain
    Nnc::Int          # total neutral-involving bonds in the chain
    sigma::Float64    # bead diameter [m]
    chi::Float64      # Flory chi for a neutral crowder cross term (inert: no crowder species yet)
end

struct LSIonParam <: Clapeyron.EoSParam
    Z::Vector{Int}
    N::Int
    Npm::Int
    Nnc::Int
    sigma::Float64
end

"""
    _safe_pos(x)

Clamp `x` to a tiny positive value instead of letting it go through zero or
negative into a `log`/`sqrt` domain error. Used only to keep `a_res`
evaluable (if physically meaningless) at the unphysical trial points a
derivative-based solver (e.g. `Clapeyron.crit_pure`, `saturation_pressure`)
may transiently visit between Newton steps -- within the physical domain
(`ρ>0`, `T>0`, `η∈(0,1)`) it is the identity and changes nothing.
"""
_safe_pos(x) = max(x, oftype(x, 1e-300))

"""
    ls_group_densities(groups, z, sigma)

Reduced per-group number densities `ρ★[k] = N_A z[1] n_flattenedgroups[1][k] σ³ / V`
for a single-component (one sequence) group-contribution model, given a
volume-carrying context supplied by the caller as `(V,)` via a closure — see
call sites in `LSNeutral`/`LSIon`.
"""
function ls_group_densities(groups::Clapeyron.GroupParam, V, z, sigma)
    n_flat = groups.n_flattenedgroups[1]
    return @. Clapeyron.N_A * z[1] * n_flat * sigma^3 / V
end
