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

include("LS/LSGroups.jl")
include("LS/LSParams.jl")
include("LS/LSNeutral.jl")
include("LS/LSIon.jl")
include("LS/LSComposite.jl")
include("sequences.jl")
include("phasediagram.jl")
include("io.jl")
include("DFT/LSDFTSystem.jl")
include("DFT/LSDFTNeutral.jl")
include("DFT/LSDFTIon.jl")
include("DFT/LSDFTComposite.jl")
include("DFT/LSElectrostaticPotential.jl")
include("DFT/LSDFTConverge.jl")
include("interfacial.jl")

export LS, LSNeutral, LSIon
export build_ls_groups, expand_ls_groups, count_bond_types, count_bond_types_from_groups
export kappa_seq, sigma_blob, block_sequence, default_sequence_family, random_blocky_sequence
export high_kappa_asymmetric_sequence, perturbed_diblock_sequence
export kappa_sym_max
export mirror_symmetry, sequence_asymmetry, kappa_sym
export find_critical_point, find_coexistence, phase_diagram, has_physical_loop, lB_to_T, coexistence_at_lB
export save_phase_diagram, load_phase_diagram, plot_phase_diagrams, apply_rcparams!
export save_interfacial, load_interfacial
export LSDFTSystem
export initialize_twophase_profile, run_interfacial_sequence, run_interfacial_point, select_binodal_indices

end # module IDPInterface
