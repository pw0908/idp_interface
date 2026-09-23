const _CHARGE_TO_LETTER = Dict(1 => 'P', -1 => 'M', 0 => 'N')
const _LETTER_TO_NAME = Dict("P" => "cation", "M" => "anion", "N" => "neutral") # get_connectivity returns String labels, not Char
const _NAME_TO_CHARGE = Dict("cation" => 1, "anion" => -1, "neutral" => 0)

_sequence_to_structure_string(charges::AbstractVector{<:Integer}) = String([_CHARGE_TO_LETTER[c] for c in charges])

# get_connectivity(::EoSModel, ::CustomStructure) doesn't use its first argument
# at all (see ClassicalDFT.jl's src/utils/connectivity.jl) -- any EoSModel will do.
const _CONNECTIVITY_DUMMY_MODEL = Clapeyron.BasicIdeal()

"""
    _ls_connectivity(charges)

Shared parse step for `build_ls_groups` (aggregate, Step 1) and
`expand_ls_groups` (per-instance, Step 2/3): converts the charge sequence
to a `ClassicalDFT.custom_structure` string and parses it via
`ClassicalDFT.get_connectivity`, so both representations are always built
from the exact same underlying sequence.
"""
function _ls_connectivity(charges::AbstractVector{<:Integer})
    codestring = _sequence_to_structure_string(charges)
    _, letters, bondmat = ClassicalDFT.get_connectivity(_CONNECTIVITY_DUMMY_MODEL, ClassicalDFT.custom_structure(codestring))
    return letters, bondmat
end

"""
    build_ls_groups(charges; seqname="ls_seq")

Build a Clapeyron `GroupParam` for a single linear IDP sequence,
group-contribution style like `SAFTgammaMie`: groups are named by chemical
*identity* ("cation"/"anion"/"neutral"), not by bead position, carrying a
per-type multiplicity and an aggregate `n_intergroups` bond-count matrix
between types -- exactly like a real gc-model (e.g. nonadecanol =
`CH3×1 + CH2×18 + OH×1` plus a CH2-CH2 bond count, not 20 uniquely-named
beads). This is *not* lossy for Step 2/3's later need to resolve individual
bead positions spatially: the exact per-instance sequence is encoded as a
`ClassicalDFT.custom_structure` string, the same mechanism ClassicalDFT
already uses to `expand_model` a compact group-contribution model into a
per-bead-resolved one for pseudo-components like block copolymers (its own
docs name this use case explicitly). Reusing `ClassicalDFT.get_connectivity`
here rather than re-deriving bond counts by hand keeps the bulk (aggregate)
and spatial (per-instance) representations consistent by construction, and
means Step 2 can hand the very same sequence string to
`ClassicalDFT.expand_model` to recover the per-bead structure with no
separate positional bookkeeping in this package at all.

Returns `(groups::GroupParam, group_names::Vector{String}, Z::Vector{Int})`,
where `Z[k]`/`group_names[k]` describe the group at `groups.flattenedgroups[k]`.
"""
function build_ls_groups(charges::AbstractVector{<:Integer}; seqname::String="ls_seq")
    N = length(charges)
    N >= 2 || throw(ArgumentError("a chain needs at least 2 beads"))

    letters, bondmat = _ls_connectivity(charges)

    unique_letters = unique(letters) # first-occurrence order
    group_names = [_LETTER_TO_NAME[l] for l in unique_letters]
    Z = [_NAME_TO_CHARGE[name] for name in group_names]
    ngt = length(group_names)

    mult = [count(==(l), letters) for l in unique_letters]
    input = [(seqname, [group_names[k] => mult[k] for k in 1:ngt])]
    groups = Clapeyron.GroupParam(input)
    @assert groups.flattenedgroups == group_names

    n_intergroups = zeros(Int, ngt, ngt)
    for i in 1:N, j in i+1:N
        bondmat[i, j] == 0 && continue
        a = findfirst(==(letters[i]), unique_letters)
        b = findfirst(==(letters[j]), unique_letters)
        n_intergroups[a, b] += bondmat[i, j]
        a != b && (n_intergroups[b, a] += bondmat[i, j])
    end
    groups.n_intergroups[1] = n_intergroups

    return groups, group_names, Z
end

"""
    expand_ls_groups(charges; seqname="ls_seq")

Per-bead-resolved companion to `build_ls_groups`, for Step 2/3's spatial
DFT functional: one uniquely-named group per bead *position* (not per
chemical identity), with the full `N×N` per-instance path-graph
`n_intergroups` bond matrix. Built from the exact same
`ClassicalDFT.get_connectivity`/`custom_structure` parse as
`build_ls_groups`'s aggregate model (`_ls_connectivity`) — not a separate
hand-derivation — so the compact bulk model (Step 1) and this expanded
spatial model always describe the same sequence by construction.

Deliberately does **not** go through `ClassicalDFT.expand_model`: that
dispatcher reconstructs a model via `MODEL(components, groups, sites,
params, idealmodel, assoc_options, references)`, a 7-field layout matching
`@newmodelgc`-defined CSV-backed association models (`HeterogcPCPSAFT`,
`SAFTgammaMie`). `LSNeutral`/`LSIon` have no CSV data and no association
sites, so routing them through that full pipeline (`build_eosparam`,
`transform_params`, `SiteParam!`, `assoc_mix!`) would add substantial
machinery this model has no use for, just to satisfy a field layout it
doesn't need. Reusing only the connectivity parser directly is simpler and
equally consistent by construction.

Returns `(groups::GroupParam, bead_names::Vector{String}, Z::Vector{Int})`,
one entry per individual bead (length `N`), analogous to `build_ls_groups`'s
return but per-instance rather than per-type.
"""
function expand_ls_groups(charges::AbstractVector{<:Integer}; seqname::String="ls_seq")
    N = length(charges)
    N >= 2 || throw(ArgumentError("a chain needs at least 2 beads"))

    letters, bondmat = _ls_connectivity(charges)
    bead_names = ["$(_LETTER_TO_NAME[letters[i]])_$(i)" for i in 1:N]
    Z = [_NAME_TO_CHARGE[_LETTER_TO_NAME[l]] for l in letters]

    input = [(seqname, [bead_names[i] => 1 for i in 1:N])]
    groups = Clapeyron.GroupParam(input)
    @assert groups.flattenedgroups == bead_names

    groups.n_intergroups[1] = bondmat

    return groups, bead_names, Z
end

"""
    count_bond_types(charges)

Given the per-bead charges of a single linear chain, count:
- `Npm`: total number of bonds between beads of *unlike* (opposite-sign)
  charge along the whole chain
- `Nnc`: total number of bonds involving a *neutral* (charge 0) bead

These are per-chain scalar totals (not per-bead averages). Substituting
these into `(1+Npm+Nnc-N)*log(ypp) - Npm*log(ypm) - Nnc*log(yhs)` (see
`LSNeutral`/`LSIon`'s `a_res`) reproduces `resources/Helmholtz_LS.jl`'s bulk
chain free energy exactly -- verified in `test/test_LS_validation.jl`.
Computed directly from the charge sequence, independent of
`build_ls_groups`'s `ClassicalDFT.get_connectivity`-based aggregation, so
the two can (and are, in `test/test_sequences.jl`) be cross-checked against
each other.
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
    count_bond_types_from_groups(group_names, n_intergroups)

Same quantities as `count_bond_types`, but read directly off a
`build_ls_groups`-style aggregate `n_intergroups` bond-count matrix and its
`group_names` labels, instead of rescanning the raw charge sequence.
"""
function count_bond_types_from_groups(group_names::Vector{String}, n_intergroups::AbstractMatrix{<:Integer})
    ic = findfirst(==("cation"), group_names)
    ia = findfirst(==("anion"), group_names)
    in_ = findfirst(==("neutral"), group_names)
    Npm = (ic !== nothing && ia !== nothing) ? n_intergroups[ic, ia] : 0
    Nnc = 0
    if in_ !== nothing
        ic !== nothing && (Nnc += n_intergroups[ic, in_])
        ia !== nothing && (Nnc += n_intergroups[ia, in_])
    end
    return Npm, Nnc
end
