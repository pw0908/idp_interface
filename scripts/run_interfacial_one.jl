#=
Per-sequence Step 3 interfacial trace, designed to run as one SLURM array
task per sequence (see `submit_interfacial_array.sh`) rather than looping
over all sequences serially in one job.

Usage: julia --project=. scripts/run_interfacial_one.jl <seqname> <npoints>
=#
using IDPInterface

const RESULTS_DIR = joinpath(@__DIR__, "..", "results")

name = ARGS[1]
npoints = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10

pd_path = joinpath(RESULTS_DIR, "phasediagram_$(name).jld2")
isfile(pd_path) || error("no phase diagram at $pd_path")
pd = load_phase_diagram(pd_path)
charges = pd.charges

println("=== $name (κ_seq=$(round(kappa_seq(charges), digits=3))), npoints=$npoints ===")
flush(stdout)

result = run_interfacial_sequence(charges, pd; npoints=npoints)
save_interfacial(joinpath(RESULTS_DIR, "interfacial_$(name).jld2"), result)
println("saved results/interfacial_$(name).jld2 (", length(result.lB), " points)")
