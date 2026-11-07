module IDPInterface

import Clapeyron
import ClassicalDFT
import ForwardDiff
import JLD2
import LinearAlgebra
import Random
import PyPlot
using PyPlot: PyDict
const plt = PyPlot

# LS/LSNeutral/LSIon (the model itself) and the general phase-equilibrium/
# critical-point machinery now live upstream in Clapeyron (as of the
# Clapeyron-integration stage) -- brought into this module's namespace and
# re-exported below rather than defined locally, so `using IDPInterface`
# still provides `LS` etc. unqualified, exactly as before. Deriving a
# `Clapeyron.LS` model from a bead-charge *sequence* remains this project's
# own concern (`ls_construction.jl`), the same way turning a SMILES string
# or component name into groups is any other Clapeyron GC model's caller's
# concern, not the model itself.
using Clapeyron: LS, LSNeutral, LSIon, LS_SIGMA,
                 DonnanBubblePressure, donnan_bubble_residual!, component_charges,
                 find_critical_salt_fraction, lB_to_T, has_physical_loop,
                 ls_group_densities

include("ls_construction.jl")
include("sequences.jl")
include("phasediagram.jl")
include("donnan_bubble_pressure.jl")
include("io.jl")
include("interfacial.jl")

export LS, LSNeutral, LSIon
export sequence_to_ls_component, sequence_to_ls_component_expanded, count_bond_types
export kappa_seq, sigma_blob, block_sequence, default_sequence_family, random_blocky_sequence
export high_kappa_asymmetric_sequence, perturbed_diblock_sequence, random_polyampholyte_sequence
export das_pappu_sequences
export kappa_sym_max
export mirror_symmetry, sequence_asymmetry, kappa_sym
export find_coexistence, phase_diagram, has_physical_loop, lB_to_T, coexistence_at_lB
export DonnanBubblePressure, donnan_bubble_residual!, component_charges
export find_critical_salt_fraction, neutral_plus_salt_phase_diagram
export critical_point_from_trace, critical_point_near_trivial_branch, critical_point_min_gap
export donnan_phase_diagram
export save_phase_diagram, load_phase_diagram, plot_phase_diagrams, apply_rcparams!
export save_donnan_phase_diagram, load_donnan_phase_diagram
export save_interfacial, load_interfacial
export run_interfacial_sequence, run_interfacial_point, select_binodal_indices
export run_interfacial_point_charged

end # module IDPInterface
