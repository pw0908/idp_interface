# `LSDFTSystem`: the composite DFT system tying `LSNeutral`'s hard-sphere/chain
# functional together with `LSIon`'s local electrostatics, exactly mirroring
# `ClassicalDFT`'s own `ElectrolyteDFTSystem` (see `models/DFT/Electrolyte/base.jl`).
#
# A dedicated struct (rather than reusing `ClassicalDFT.DFTSystem` via its
# convenience constructor) is required because `DFTSystem`'s public constructor
# unconditionally calls `ClassicalDFT.expand_model`, which reconstructs a
# `@newmodelgc`-style 7-field model layout that `LSNeutral`/`LSIon`'s plain
# 3/4-field structs don't match (and don't need to -- `expand_ls_groups`,
# `LSGroups.jl`, already produces the per-bead `GroupParam` these models are
# built from). `ElectrolyteDFTSystem` bypasses the same problem the same way:
# by calling `get_species`/`get_fields`/`get_propagator` directly and building
# the system struct itself.
struct LSDFTSystem{M<:LS,S<:ClassicalDFT.DFTSpecies,I<:ClassicalDFT.DFTSpecies,
                    T<:ClassicalDFT.DFTStructure,F,EF,P<:ClassicalDFT.DFTPropagator,
                    O<:ClassicalDFT.DFTOptions,C} <: ClassicalDFT.AbstractcDFTSystem
    model::M
    species::S
    ion_species::I
    structure::T
    fields::F
    external_field::EF
    propagator::P
    options::O
    chunksize::Val{C}
end
