#=
Step 1 deliverable: bulk phase diagrams for the default 20-mer sequence
family (spanning kappa_seq from alternating to perfect-diblock). Saves each
sequence's coexistence curve as compressed JLD2 under `results/`, and a
combined overlay figure under `results/phase_diagrams.pdf`.

Usage: julia --project=. scripts/run_phase_diagrams.jl
=#
using IDPInterface

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
mkpath(RESULTS_DIR)

family = default_sequence_family(20)

results = []
labels = Float64[]
for (blocksize, charges) in family
    κ = kappa_seq(charges)
    println("Running blocksize=$blocksize (κ=$(round(κ, digits=3)))...")
    pd = phase_diagram(charges)
    println("  found $(length(pd.lB)) coexistence points, lB in ",
            isempty(pd.lB) ? "none" : extrema(pd.lB))
    save_phase_diagram(joinpath(RESULTS_DIR, "phasediagram_block$(blocksize).jld2"), pd; charges=charges)
    push!(results, pd)
    push!(labels, κ)
end

fig = plot_phase_diagrams(results, labels; path=joinpath(RESULTS_DIR, "phase_diagrams.pdf"))
println("Saved figure to results/phase_diagrams.pdf")
