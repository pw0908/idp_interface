#=
Sequence -> Clapeyron.LS construction. Clapeyron's own `LS` constructor
takes already-parsed group/bond data directly (HeterogcPCPSAFT-style, see
its docstring) -- turning a linear bead-charge sequence into that data is
this project's own concern, the same way resolving a SMILES string or a
component name into groups is the *caller's* concern for any other
Clapeyron group-contribution model, not the model itself.
=#

const _CHARGE_TO_GROUP = Dict(1 => "+", -1 => "-", 0 => "0")

"""
    count_bond_types(charges)

Given the per-bead charges of a single linear chain, count `Npm` (bonds
between beads of unlike, opposite-sign charge) and `Nnc` (bonds involving a
neutral bead). Per-chain scalar totals, computed directly from the charge
sequence.
"""
function count_bond_types(charges::AbstractVector{<:Integer})
    N = length(charges)
    Npm = 0
    Nnc = 0
    for i in 1:N-1
        a, b = charges[i], charges[i+1]
        if a == 0 || b == 0
            Nnc += 1
        elseif a != b
            Npm += 1
        end
    end
    return Npm, Nnc
end

"""
    sequence_to_ls_component(charges; name="ls_seq")

Turn a linear bead-charge sequence (`+1`/`-1`/`0` per bead) into one
`Clapeyron.LS`-style `(name, group=>count pairs, (group1,group2)=>bond_count pairs)`
triple: bead `i` is bonded to bead `i+1`, and groups are aggregated by
chemical identity ("+"/"-"/"0"), not bead position -- exactly like an
ordinary group-contribution model (e.g. nonadecanol = CH3×1 + CH2×18 + OH×1
plus a CH2-CH2 bond count, not 20 uniquely-named beads).
"""
function sequence_to_ls_component(charges::AbstractVector{<:Integer}; name::String="ls_seq")
    N = length(charges)
    N >= 2 || throw(ArgumentError("a chain needs at least 2 beads"))
    bead_group = [_CHARGE_TO_GROUP[c] for c in charges]
    group_names = unique(bead_group) # first-occurrence order
    mult = [count(==(g), bead_group) for g in group_names]

    bond_count = Dict{Tuple{String,String},Int}()
    for i in 1:N-1
        a, b = bead_group[i], bead_group[i+1]
        key = a <= b ? (a, b) : (b, a)
        bond_count[key] = get(bond_count, key, 0) + 1
    end
    groups = [group_names[k] => mult[k] for k in eachindex(group_names)]
    bonds = [k => v for (k, v) in bond_count]
    return (name, groups, bonds)
end

"""
    sequence_to_ls_component_expanded(charges; name="ls_seq")

Per-bead-resolved companion to `sequence_to_ls_component`, for a spatially-
resolved (classical-DFT) functional: one uniquely-named group per bead
*position* (not per chemical identity), so every real bead is individually
addressable.
"""
function sequence_to_ls_component_expanded(charges::AbstractVector{<:Integer}; name::String="ls_seq")
    N = length(charges)
    N >= 2 || throw(ArgumentError("a chain needs at least 2 beads"))
    bead_names = ["$(_CHARGE_TO_GROUP[charges[i]])_$(i)" for i in 1:N]
    groups = [bead_names[i] => 1 for i in 1:N]
    bonds = [(bead_names[i], bead_names[i+1]) => 1 for i in 1:N-1]
    return (name, groups, bonds)
end

"""
    LS(charges; ion_charge=nothing, ionnames=nothing, expand=false, seqname="ls_seq", RSPmodel=Clapeyron.ConstRSP())

Convenience constructor matching this project's calling convention: build a
`Clapeyron.LS` model directly from a linear bead-charge sequence, adding
one free-ion component per entry in `ion_charge` (a single `Int` for one
ion, e.g. a stoichiometric counter-ion, or a `Vector{Int}`/`Tuple` for
several, e.g. a symmetric added-salt pair `[1,-1]`), each a bare single-
group component named from `ionnames` or `"ion1"`,`"ion2"`,... by default.

`expand=true` builds the per-bead-resolved (`sequence_to_ls_component_expanded`)
representation instead, for a spatial (classical-DFT) functional.
"""
function Clapeyron.LS(charges::AbstractVector{<:Integer}; ion_charge::Union{Int,AbstractVector{<:Integer},Nothing}=nothing,
            ionnames::Union{AbstractVector{String},Nothing}=nothing, expand::Bool=false,
            seqname::String="ls_seq", RSPmodel=Clapeyron.ConstRSP())
    ions = ion_charge === nothing ? Int[] : (ion_charge isa Int ? [ion_charge] : collect(ion_charge))
    ionnames === nothing || length(ionnames) == length(ions) ||
        throw(ArgumentError("ionnames must have one entry per ion_charge"))
    names = ionnames === nothing ? ["ion$(k)" for k in 1:length(ions)] : collect(ionnames)

    chain = expand ? sequence_to_ls_component_expanded(charges; name=seqname) : sequence_to_ls_component(charges; name=seqname)
    input = Any[chain]
    for (ionname, z) in zip(names, ions)
        push!(input, (ionname, [_CHARGE_TO_GROUP[z] => 1], Pair{Tuple{String,String},Int}[]))
    end
    return Clapeyron.LS(input; RSPmodel=RSPmodel)
end
